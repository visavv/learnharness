"""Local Whisper dictation server for LearnHarness.

Speaks the subset of Deepgram's streaming protocol that LearnHarness uses,
so Settings -> Dictation -> Server = ws://127.0.0.1:8765 routes dictation
here instead of the Deepgram cloud:

  - client connects to /v1/listen?...&language=<code> and streams binary
    frames of 16 kHz mono linear16 PCM;
  - the server cuts the audio at pauses and sends each transcribed segment
    as {"type": "Results", "is_final": true, "channel": {"alternatives":
    [{"transcript": "..."}]}} while the user keeps talking;
  - {"type": "CloseStream"} transcribes the remaining tail, sends it, and
    closes with code 1000.

Transcription runs on faster-whisper. The model loads on the first
connection and unloads after an idle period, so it holds no VRAM while
dictation is unused.

    python server.py [--host 127.0.0.1] [--port 8765] [--model large-v3-turbo]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np


def _add_cuda_dll_dirs() -> None:
    """pip's nvidia-* wheels ship cuBLAS/cuDNN DLLs that Windows won't find
    on its own; register their bin/ folders before ctranslate2 loads."""
    if sys.platform != "win32":
        return
    for entry in sys.path:
        nvidia = Path(entry) / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in nvidia.glob("*/bin"):
            os.add_dll_directory(str(bin_dir))
            os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


_add_cuda_dll_dirs()

import websockets  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

log = logging.getLogger("whisper-dictation")

SAMPLE_RATE = 16000
FRAME = SAMPLE_RATE // 50  # 20 ms analysis frames
SPEECH_DBFS = -42.0  # a frame louder than this counts as speech
# LearnHarness inserts the transcript only when dictation stops, so early
# cuts only exist to keep the tail short. Long segments give Whisper whole
# sentences (short ones come back as "...fragments...").
MIN_SEGMENT_S = 10.0  # don't cut before this much audio...
PAUSE_S = 0.6  # ...and then cut at the first pause this long
MAX_SEGMENT_S = 28.0  # Whisper's window is 30 s: force a cut before it
FORCED_CUT_SEARCH_S = 3.0  # a forced cut lands on the quietest recent frame
MIN_SPEECH_S = 0.4  # segments with less speech are dropped (clicks, breaths)


class Model:
    """One shared faster-whisper model: lazy load, serialized use, idle unload."""

    def __init__(self, name: str, device: str, compute_type: str, idle_s: float) -> None:
        self.name, self.device, self.compute_type, self.idle_s = name, device, compute_type, idle_s
        self._model: WhisperModel | None = None
        self._lock = threading.Lock()
        self._last_used = time.monotonic()

    def _ensure(self) -> WhisperModel:
        if self._model is None:
            started = time.monotonic()
            self._model = WhisperModel(self.name, device=self.device, compute_type=self.compute_type)
            log.info("loaded %s on %s/%s in %.1fs", self.name, self.device, self.compute_type,
                     time.monotonic() - started)
        return self._model

    def warm(self) -> None:
        with self._lock:
            self._ensure()
            self._last_used = time.monotonic()

    def transcribe(self, audio: np.ndarray, language: str | None, prompt: str) -> str:
        with self._lock:
            model = self._ensure()
            segments, _ = model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=True,
                condition_on_previous_text=False,
                initial_prompt=prompt or None,
            )
            text = " ".join(s.text.strip() for s in segments if s.text.strip())
            self._last_used = time.monotonic()
            # A segment cut mid-thought comes back as "...words..."; the
            # segments are joined with spaces, so the ellipses only clutter.
            return text.strip().removeprefix("...").removesuffix("...").strip()

    def unload_if_idle(self) -> None:
        with self._lock:
            if self._model is not None and time.monotonic() - self._last_used > self.idle_s:
                self._model = None
                log.info("unloaded %s after %.0fs idle", self.name, self.idle_s)


def whisper_language(code: str | None) -> str | None:
    """LearnHarness sends Deepgram codes ("en", "en-GB", "zh-Hant", "multi");
    Whisper takes the bare ISO 639-1 part, or None to auto-detect."""
    if not code or code == "multi":
        return None
    return code.split("-")[0].lower()


class Segmenter:
    """Cuts the incoming PCM stream at pauses so finished sentences can be
    transcribed while the user is still talking."""

    def __init__(self) -> None:
        self.pending = np.zeros(0, dtype=np.float32)  # not yet analysed (< 1 frame)
        self.frames: list[np.ndarray] = []
        self.loudness: list[float] = []  # dBFS per frame
        self.silent_frames = 0

    def feed(self, pcm: bytes) -> list[np.ndarray]:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        self.pending = np.concatenate([self.pending, samples])
        done: list[np.ndarray] = []
        while len(self.pending) >= FRAME:
            frame, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            db = 20 * np.log10(float(np.sqrt(np.mean(frame * frame))) or 1e-9)
            self.frames.append(frame)
            self.loudness.append(db)
            self.silent_frames = 0 if db > SPEECH_DBFS else self.silent_frames + 1
            seconds = len(self.frames) * FRAME / SAMPLE_RATE
            if seconds >= MIN_SEGMENT_S and self.silent_frames * FRAME >= PAUSE_S * SAMPLE_RATE:
                done += self._emit(len(self.frames))
            elif seconds >= MAX_SEGMENT_S:
                window = int(FORCED_CUT_SEARCH_S * SAMPLE_RATE / FRAME)
                recent = self.loudness[-window:]
                done += self._emit(len(self.frames) - window + int(np.argmin(recent)) + 1)
        return done

    def _emit(self, n: int) -> list[np.ndarray]:
        """Cut the first n frames off as a segment; the rest stays."""
        audio, loudness = self.frames[:n], self.loudness[:n]
        self.frames, self.loudness = self.frames[n:], self.loudness[n:]
        self.silent_frames = 0
        speech = sum(db > SPEECH_DBFS for db in loudness) * FRAME
        return [np.concatenate(audio)] if speech >= MIN_SPEECH_S * SAMPLE_RATE else []

    def cut(self) -> np.ndarray | None:
        """End the stream: the remainder as one segment, or None when it
        held too little speech."""
        if len(self.pending):
            self.frames.append(self.pending)
            self.loudness.append(SPEECH_DBFS - 1)
            self.pending = np.zeros(0, dtype=np.float32)
        tail = self._emit(len(self.frames))
        return tail[0] if tail else None


def results(text: str) -> str:
    return json.dumps({
        "type": "Results",
        "is_final": True,
        "channel": {"alternatives": [{"transcript": text}]},
    })


async def handle(ws, model: Model) -> None:
    url = urlparse(ws.request.path)
    if url.path.rstrip("/") != "/v1/listen":
        await ws.close(1008, "expected /v1/listen")
        return
    language = whisper_language(parse_qs(url.query).get("language", [None])[0])
    log.info("session start (language=%s)", language or "auto")

    # Load while the user starts talking; the first segment waits on it anyway.
    warming = asyncio.create_task(asyncio.to_thread(model.warm))
    segmenter = Segmenter()
    queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue()

    async def transcriber() -> None:
        context = ""
        while (audio := await queue.get()) is not None:
            await warming
            text = await asyncio.to_thread(model.transcribe, audio, language, context)
            if text:
                context = (context + " " + text)[-200:]
                await ws.send(results(text))

    worker = asyncio.create_task(transcriber())
    try:
        async for message in ws:
            if isinstance(message, bytes):
                for audio in segmenter.feed(message):
                    queue.put_nowait(audio)
                continue
            try:
                kind = json.loads(message).get("type")
            except (ValueError, AttributeError):
                continue
            if kind == "CloseStream":
                tail = segmenter.cut()
                if tail is not None:
                    queue.put_nowait(tail)
                queue.put_nowait(None)
                await worker
                await ws.close(1000)
                log.info("session end")
                return
            # KeepAlive and anything else: nothing to do.
    except websockets.ConnectionClosed:
        pass
    finally:
        worker.cancel()
        log.info("session closed")


async def idle_unloader(model: Model) -> None:
    while True:
        await asyncio.sleep(30)
        await asyncio.to_thread(model.unload_if_idle)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--idle-unload", type=float, default=300, help="seconds before the model unloads")
    parser.add_argument("--log-file")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename=args.log_file,
    )
    model = Model(args.model, args.device, args.compute_type, args.idle_unload)
    asyncio.create_task(idle_unloader(model))
    async with websockets.serve(lambda ws: handle(ws, model), args.host, args.port, max_size=None):
        log.info("listening on ws://%s:%d (model %s)", args.host, args.port, args.model)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
