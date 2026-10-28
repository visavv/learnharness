# Local Whisper dictation server

A small Deepgram-compatible streaming server that transcribes dictation locally with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). Point LearnHarness at it and
dictation never leaves your machine — no Deepgram key needed.

## Run it

```sh
python -m venv .venv
.venv/Scripts/python -m pip install -r tools/whisper-dictation/requirements.txt   # Windows
# .venv/bin/python -m pip install -r tools/whisper-dictation/requirements.txt     # macOS/Linux
.venv/Scripts/python tools/whisper-dictation/server.py
```

Then in LearnHarness: **Settings → Providers → Dictation → Server** = `ws://127.0.0.1:8765`.
Dictation (the mic button, ⌘D/Ctrl+D) now streams to this server; the Language setting still applies
(**Multilingual** = Whisper's auto-detect).

Options: `--model` (any faster-whisper model, default `large-v3-turbo`), `--device cuda|cpu`,
`--compute-type` (`float16`, `int8_float16`, `int8` for CPU), `--port`, `--idle-unload` (seconds
before the model leaves VRAM, default 300), `--log-file`.

The CUDA wheels in `requirements.txt` cover NVIDIA GPUs on Windows/Linux. On a Mac, or with no
NVIDIA GPU, leave them out and run with `--device cpu --compute-type int8` (slower; a smaller model
such as `small` helps).

## How it works

It implements the part of Deepgram's `/v1/listen` streaming protocol LearnHarness uses: binary
16 kHz linear16 audio frames in, `{"type": "Results", "is_final": true, ...}` messages out,
`{"type": "CloseStream"}` to finish (the server sends the last result and closes with 1000).

LearnHarness inserts the transcript only when dictation stops, so the server transcribes long
stretches in the background while you talk — it cuts at the first pause after 10 s of audio, or at
the quietest moment before Whisper's 30 s window — and only the last stretch is left when you stop.
The model loads on the first dictation and unloads after the idle timeout.
