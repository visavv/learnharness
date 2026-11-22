import React, { act } from "react";
(globalThis as { React?: typeof React }).React = React;
import assert from "node:assert/strict";
import { test } from "node:test";
import { parseHTML } from "linkedom";
import { compactAnswerForModel, type QuestionAnswerData } from "../src/core/index.ts";

const dom = parseHTML("<html><body></body></html>");
Object.assign(globalThis, { window: dom.window, document: dom.document, Element: dom.window.Element,
	HTMLElement: dom.window.HTMLElement, HTMLInputElement: dom.window.HTMLInputElement,
	HTMLTextAreaElement: dom.window.HTMLTextAreaElement, Event: dom.window.Event, CustomEvent: dom.window.CustomEvent,
	IS_REACT_ACT_ENVIRONMENT: true });
// linkedom has no layout: the walked row's scrollIntoView is a no-op here.
(dom.window.Element.prototype as { scrollIntoView?: () => void }).scrollIntoView = () => {};
// linkedom has no focus model either: track the last focused element.
let active: Element = dom.document.body;
Object.defineProperty(dom.document, "activeElement", { get: () => active });
dom.window.HTMLElement.prototype.focus = function (this: Element) { active = this; };
// React only listens for "input" when the document advertises oninput.
Object.assign(dom.document, { oninput: null });

const press = async (key: string) => {
	await act(async () => {
		document.body.dispatchEvent(Object.assign(new window.Event("keydown", { bubbles: true, cancelable: true }), { key }));
	});
};

test("number keys pick the numbered row and Enter submits it", async () => {
	const input = { question: "Pick one", options: [{ label: "alpha" }, { label: "beta" }, { label: "gamma" }], correctAnswer: "beta", explanation: "b" };
	let answer: QuestionAnswerData | undefined;
	// Imported after the DOM globals exist: react-dom decides input-event
	// support when it loads, and static imports run before this file's setup.
	const { createRoot } = await import("react-dom/client");
	const { QuestionDock } = await import("../components/question-dock.tsx");
	const host = document.createElement("div"); document.body.append(host);
	const root = createRoot(host);
	try {
		await act(async () => root.render(<QuestionDock toolName="probe" toolCallId="numkeys" input={input} expanded active
			dictationClaim={false} onExpand={() => {}} onSubmit={a => { answer = a; }} />));
		const radios = () => [...host.querySelectorAll<HTMLButtonElement>('[role="radio"]')];
		assert.deepEqual(radios().map(r => r.querySelector("kbd")?.textContent), ["1", "2", "3"]);
		await press("2");
		assert.equal(radios()[1]!.getAttribute("aria-checked"), "true");
		await press("9"); // past the last row (3 options + "I don't know"): ignored
		assert.equal(radios()[1]!.getAttribute("aria-checked"), "true");
		await press("Enter");
		await act(async () => { await new Promise(r => setTimeout(r, 0)); });
		assert.equal(answer?.answers[0]?.value, "beta");
	} finally { await act(async () => root.unmount()); host.remove(); }
});

test("S focuses the steer line", async () => {
	const input = { question: "Pick one", options: [{ label: "alpha" }, { label: "beta" }], correctAnswer: "beta", explanation: "b" };
	const { createRoot } = await import("react-dom/client");
	const { QuestionDock } = await import("../components/question-dock.tsx");
	const host = document.createElement("div"); document.body.append(host);
	const root = createRoot(host);
	try {
		await act(async () => root.render(<QuestionDock toolName="probe" toolCallId="steer" input={input} expanded active
			dictationClaim={false} onExpand={() => {}} onSubmit={() => {}} />));
		const steer = host.querySelector<HTMLInputElement>('[data-testid="steer-field"] input')!;
		await press("s");
		assert.ok(document.activeElement === steer, "S focuses the steer line");
	} finally { await act(async () => root.unmount()); host.remove(); }
});

test("a steer reaches the teacher as an unanswered question carrying the message", () => {
	const base = { kind: "quiz" as const, questionText: "Q", dontKnow: false, answers: [], answeredAt: "2026-10-28T00:00:00Z" };
	assert.deepEqual(compactAnswerForModel({ ...base, redirect: "teach me && first", thinking: "unsure" }), {
		kind: "quiz", answered: false, learnerMessage: "teach me && first", thinking: "unsure",
	});
	assert.equal(compactAnswerForModel({ ...base, redirect: "x", correct: false, confidence: 70 }).correct, undefined);
});
