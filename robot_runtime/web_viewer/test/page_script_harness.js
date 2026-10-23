// Copyright (c) 2026, Nicholas Palomo. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Runs the <script> of the viewer's page (page/index.html) under node, against a small DOM, a fetch whose answers the
// scenario gives, and a virtual clock that drives setTimeout, setInterval and requestAnimationFrame. Usage:
//   node page_script_harness.js <index.html> <scenario>
// Exits 0 and prints "PASS <scenario>" when the scenario's assertions hold, and exits 1 with the failure otherwise.
// test/test_page_script.py runs every scenario as a test.
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const webcrypto = require("node:crypto").webcrypto;

const FRAME_INTERVAL_MS = 16;  // one animation frame

// ---------------------------------------------------------------------------------------------------------------------
// The DOM: the elements the page reads, by id, and the ones it creates.
// ---------------------------------------------------------------------------------------------------------------------

class FakeElement {
  constructor(tagName, id = "") {
    this.tagName = tagName.toUpperCase();
    this.id = id;
    this.parent = null;
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this.dataset = {};
    this.className = "";
    this.title = "";
    this.hidden = false;
    this.checked = false;
    this.ownText = "";
    this.selectValue = "";
    this.width = 0;
    this.height = 0;
    this.clientWidth = 1280;
    this.clientHeight = 720;
    const element = this;
    this.classList = {
      add(name) { if (!element.className.split(" ").includes(name)) element.className = `${element.className} ${name}`.trim(); },
      remove(name) { element.className = element.className.split(" ").filter((part) => part !== name).join(" "); },
      contains(name) { return element.className.split(" ").includes(name); },
    };
  }

  get textContent() { return this.ownText + this.children.map((child) => child.textContent).join(""); }
  set textContent(text) {
    for (const child of this.children) child.parent = null;
    this.children = [];
    this.ownText = String(text);
  }

  // A <select>'s value is its selected option's; an <input>'s is its own.
  get value() {
    if (this.tagName !== "SELECT") return this.selectValue;
    const selected = this.children.find((option) => option.selected) ?? this.children[0];
    return selected === undefined ? "" : selected.value;
  }
  set value(value) {
    if (this.tagName !== "SELECT") {
      this.selectValue = String(value);
      return;
    }
    for (const option of this.children) option.selected = option.value === String(value);
  }
  get options() { return this.children; }

  get isConnected() {
    let node = this;
    while (node.parent !== null) node = node.parent;
    return node === documentRoot;
  }

  append(...children) {
    for (const child of children) {
      if (child.parent !== null) child.parent.children = child.parent.children.filter((other) => other !== child);
      child.parent = this;
      this.children.push(child);
    }
  }
  replaceChildren(...children) {
    this.textContent = "";
    this.append(...children);
  }
  addEventListener(type, listener) { (this.listeners[type] ||= []).push(listener); }
  dispatch(type, init = {}) {
    let defaultPrevented = false;
    const event = { type, target: this, preventDefault() { defaultPrevented = true; }, ...init };
    for (const listener of this.listeners[type] || []) listener(event);
    return defaultPrevented;
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name] ?? null; }
  focus() { activeElement = this; }
  blur() { if (activeElement === this) activeElement = body; }
  setPointerCapture() {}
  getBoundingClientRect() { return { width: this.clientWidth, height: this.clientHeight }; }
  getContext() { return { drawImage() {} }; }
  toBlob(callback) { callback(null); }
}

const documentRoot = new FakeElement("html");
const body = new FakeElement("body");
documentRoot.append(body);
let activeElement = body;
const elementsById = {};

function addElement(tagName, id, parent = body) {
  const element = new FakeElement(tagName, id);
  parent.append(element);
  elementsById[id] = element;
  return element;
}

// The page's elements (index.html), with the attributes its markup gives them.
const view = addElement("div", "view");
const canvas = addElement("canvas", "frame", view);
canvas.width = 1280;
canvas.height = 720;
for (const id of ["hudFps", "hudSize", "hudCamera", "hudSimTime", "hudAge", "hudClients", "status"]) addElement("span", id);
for (const id of ["resetCamera", "trackCamera", "snapshot", "toggleVisualizations", "toggleSettings"]) addElement("button", id);
addElement("div", "settings").hidden = true;
const visualizationsPanel = addElement("div", "visualizations");
visualizationsPanel.hidden = true;
const visualizationList = addElement("div", "visualizationList", visualizationsPanel);
const presets = [["640", "360"], ["960", "540"], ["1280", "720"], ["1920", "1080"]].map(([width, height]) => {
  const preset = new FakeElement("button");
  preset.dataset = { width, height };
  body.append(preset);
  return preset;
});
addElement("input", "fitWindow");
const fpsSelect = addElement("select", "fps");
for (const rate of ["10", "15", "20", "30", "60"]) {
  const option = new FakeElement("option");
  option.value = rate;
  option.ownText = rate;
  option.selected = rate === "30";
  fpsSelect.append(option);
}
const qualityInput = addElement("input", "quality");
qualityInput.value = "70";
addElement("span", "qualityValue").textContent = "70";

// ---------------------------------------------------------------------------------------------------------------------
// The virtual clock: timers, intervals and animation frames run only when the scenario advances it.
// ---------------------------------------------------------------------------------------------------------------------

let now = 0;
let nextTimerId = 1;
const timers = new Map();  // id -> { due, callback, interval }
let animationFrameCallbacks = [];
let nextAnimationFrame = FRAME_INTERVAL_MS;

function setTimeoutFake(callback, delay = 0) {
  const id = nextTimerId++;
  timers.set(id, { due: now + Math.max(0, delay), callback, interval: null });
  return id;
}
function setIntervalFake(callback, interval) {
  const id = nextTimerId++;
  timers.set(id, { due: now + interval, callback, interval });
  return id;
}
function clearTimer(id) { timers.delete(id); }

// Lets every pending promise reaction run: setImmediate fires after the microtask queue is empty.
function settle() { return new Promise((resolve) => setImmediate(resolve)); }

async function advance(milliseconds) {
  const until = now + milliseconds;
  while (true) {
    let nextDue = Infinity;
    let nextId = null;
    for (const [id, timer] of timers) {
      if (timer.due < nextDue) {
        nextDue = timer.due;
        nextId = id;
      }
    }
    const frameDue = animationFrameCallbacks.length > 0 ? nextAnimationFrame : Infinity;
    const due = Math.min(nextDue, frameDue);
    if (due > until) break;
    now = Math.max(now, due);
    if (frameDue <= nextDue) {
      const callbacks = animationFrameCallbacks;
      animationFrameCallbacks = [];
      nextAnimationFrame = now + FRAME_INTERVAL_MS;
      for (const callback of callbacks) callback(now);
    } else {
      const timer = timers.get(nextId);
      if (timer.interval === null) {
        timers.delete(nextId);
      } else {
        timer.due += timer.interval;
      }
      timer.callback();
    }
    await settle();
  }
  now = until;
  if (nextAnimationFrame < now) nextAnimationFrame = now;
  await settle();
}

// ---------------------------------------------------------------------------------------------------------------------
// fetch: every request waits until the scenario answers it.
// ---------------------------------------------------------------------------------------------------------------------

const requests = [];  // { url, method, resolve, reject, answered }

function fakeFetch(url, init = {}) {
  return new Promise((resolve, reject) => {
    requests.push({ url, method: init.method ?? "GET", headers: init.headers ?? {}, resolve, reject, answered: false });
  });
}

function makeResponse({ status = 200, headers = {}, body = "" }) {
  const lowerCaseHeaders = Object.fromEntries(Object.entries(headers).map(([name, value]) => [name.toLowerCase(), String(value)]));
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: { get: (name) => lowerCaseHeaders[name.toLowerCase()] ?? null },
    text: async () => (typeof body === "string" ? body : JSON.stringify(body)),
    json: async () => (typeof body === "string" ? JSON.parse(body) : body),
    blob: async () => ({ size: 1 }),
  };
}

function pending(predicate) { return requests.filter((request) => !request.answered && predicate(request)); }
function isFrameRequest(request) { return request.method === "GET" && request.url.startsWith("/frame.jpg"); }
function isStateRequest(request) { return request.method === "GET" && request.url.startsWith("/state"); }
function isPost(request) { return request.method === "POST"; }

async function answer(request, response) {
  request.answered = true;
  request.resolve(makeResponse(response));
  await settle();
}
async function fail(request, message) {
  request.answered = true;
  request.reject(new TypeError(message));
  await settle();
}

// Answers the one waiting frame request; fails when none waits.
async function answerFrame(response) {
  const waiting = pending(isFrameRequest);
  assert.equal(waiting.length, 1, `one frame request should wait, not ${waiting.length}`);
  await answer(waiting[0], response);
}
let frameSequence = 0;
async function deliverFrame(ageMs = 10) {
  frameSequence += 1;
  await answerFrame({ status: 200, headers: { "X-Frame-Sequence": frameSequence, "X-Frame-Age-Ms": ageMs } });
}
// Delivers `count` frames, one per animation frame.
async function deliverFrames(count) {
  for (let i = 0; i < count; ++i) {
    await advance(FRAME_INTERVAL_MS);
    await deliverFrame();
  }
  await advance(FRAME_INTERVAL_MS);
}

// Answers every waiting POST with `response` until none is left, letting the page send the next one each frame.
async function answerPosts(response = { status: 204 }) {
  for (let round = 0; round < 50; ++round) {
    await advance(FRAME_INTERVAL_MS);
    const waiting = pending(isPost);
    if (waiting.length === 0) return;
    for (const request of waiting) await answer(request, response);
  }
}

function postUrls() { return requests.filter(isPost).map((request) => decodeURIComponent(request.url)); }
function postParameters() {
  return requests.filter(isPost).map((request) => Object.fromEntries(new URL(request.url, "http://viewer").searchParams));
}

// A /state answer: the server's JSON (README.md, "Routes") with `viewer` fields overridden.
function viewerState({ visualizations = [], render = {}, mode = "tracking", viewers = 1, frame = null } = {}) {
  return {
    clients: 0, viewers, maxClients: 4, droppedInputs: 0, frame,
    viewer: {
      camera: { mode, azimuth: 90, elevation: -20, distance: 3, lookat: [0, 0, 1] },
      render: { width: 1280, height: 720, targetFps: 30, measuredFps: 29.5, quality: 70, ...render },
      geomGroups: [true, true, true, false, false, false],
      visualizations,
    },
  };
}
async function answerState(state) {
  const waiting = pending(isStateRequest);
  assert.ok(waiting.length >= 1, "no /state request waits");
  for (const request of waiting) await answer(request, { status: 200, body: state });
}
// Lets the 500 ms poll ask for /state and answers it.
async function pollState(state) {
  await advance(500);
  await answerState(state);
}

// ---------------------------------------------------------------------------------------------------------------------
// Loading the page.
// ---------------------------------------------------------------------------------------------------------------------

function loadPage(html, search) {
  const script = /<script>([\s\S]*?)<\/script>/.exec(html);
  assert.ok(script !== null, "the page has no <script>");
  const document = {
    hidden: false,
    getElementById: (id) => elementsById[id] ?? null,
    querySelectorAll: (selector) => (selector === "[data-width]" ? presets : []),
    createElement: (tagName) => new FakeElement(tagName),
    get activeElement() { return activeElement; },
  };
  const window = { location: { search }, addEventListener() {} };
  const globals = {
    document, window,
    performance: { now: () => now },
    requestAnimationFrame: (callback) => animationFrameCallbacks.push(callback),
    setTimeout: setTimeoutFake, clearTimeout: clearTimer, setInterval: setIntervalFake, clearInterval: clearTimer,
    fetch: fakeFetch,
    createImageBitmap: async () => ({ width: canvas.width, height: canvas.height, close() {} }),
    WheelEvent: { DOM_DELTA_PIXEL: 0, DOM_DELTA_LINE: 1, DOM_DELTA_PAGE: 2 },
    URL: { createObjectURL: () => "blob:frame", revokeObjectURL() {} },
    URLSearchParams, crypto: webcrypto, console,
  };
  new Function(...Object.keys(globals), script[1])(...Object.values(globals));
}

// ---------------------------------------------------------------------------------------------------------------------
// Input helpers.
// ---------------------------------------------------------------------------------------------------------------------

function pointer(type, x, y, init = {}) {
  view.dispatch(type, { pointerId: 1, button: 0, clientX: x, clientY: y, shiftKey: false, ...init });
}
function key(code) {
  view.dispatch("keydown", { code, repeat: false, ctrlKey: false, altKey: false, metaKey: false });
}
function wheel(deltaY) { view.dispatch("wheel", { deltaMode: 0, deltaY }); }

// The row of the marker `name` in the Markers panel.
function markerRow(name) {
  return visualizationList.children.find((row) => row.children.some((child) => child.className === "name" && child.textContent === name));
}

// A press on `element`, then `between` (which may re-render the page), then the release: as a browser does, the click
// reaches the element only if it is still in the document when the button comes up.
async function clickWith(element, between) {
  await between();
  if (element.isConnected) element.dispatch("click");
  await settle();
}

const ZMP = { name: "zmp", hotkey: "z", enabled: false, description: "zero moment point" };
const DCM = { name: "dcm", hotkey: "d", enabled: true, description: "divergent component of motion" };

// ---------------------------------------------------------------------------------------------------------------------
// Scenarios.
// ---------------------------------------------------------------------------------------------------------------------

const scenarios = {
  // A /state poll between a marker's press and its release must not swallow the click (README.md: "click a marker to
  // toggle it").
  async markerRowsSurviveAPoll() {
    await answerState(viewerState({ visualizations: [ZMP, DCM] }));
    const zmp = markerRow("zmp");
    assert.ok(zmp !== undefined, "no row for zmp");
    assert.ok(!zmp.classList.contains("on"));
    await pollState(viewerState({ visualizations: [ZMP, DCM] }));
    assert.equal(markerRow("zmp"), zmp, "a /state poll replaced the marker rows");

    await clickWith(zmp, () => pollState(viewerState({ visualizations: [ZMP, DCM] })));
    await answerPosts();
    assert.deepEqual(postUrls(), ["/input?type=visualization&name=zmp"]);

    await pollState(viewerState({ visualizations: [{ ...ZMP, enabled: true }, DCM] }));
    assert.equal(markerRow("zmp"), zmp);
    assert.ok(zmp.classList.contains("on"), "the row does not show the marker on");

    // Another robot: other markers, new rows.
    await pollState(viewerState({ visualizations: [DCM] }));
    assert.equal(markerRow("zmp"), undefined);
    assert.ok(markerRow("dcm") !== undefined);
    await pollState(viewerState({ visualizations: [] }));
    assert.equal(visualizationList.textContent, "No markers are configured for this robot.");
    await pollState(viewerState({ visualizations: [ZMP] }));
    assert.ok(markerRow("zmp") !== undefined);
    assert.ok(!visualizationList.textContent.startsWith("No markers"));
  },

  // anInputRefusalOutlivesTheFrameLoop: below, since it loads the page with a query.

  // A key or a button after a drag or a scroll is applied after that motion, not before it.
  async inputsKeepTheirOrderAfterMotion() {
    pointer("pointerdown", 0, 0);
    pointer("pointermove", 72, 0);
    await advance(FRAME_INTERVAL_MS);  // the first drag goes out and stays in flight, as through a tunnel
    assert.equal(pending(isPost).length, 1);
    pointer("pointermove", 792, 0);
    pointer("pointerup", 792, 0);
    elementsById.resetCamera.dispatch("click");
    await answerPosts();
    assert.deepEqual(postParameters().map((parameters) => parameters.type), ["drag", "drag", "reset_camera"]);

    requests.length = 0;
    wheel(-100);
    await advance(FRAME_INTERVAL_MS);
    wheel(-500);
    elementsById.resetCamera.dispatch("click");
    await answerPosts();
    assert.deepEqual(postParameters().map((parameters) => [parameters.type, parameters.notches]),
                     [["scroll", "1.000"], ["scroll", "5.000"], ["reset_camera", undefined]]);

    // A key in the middle of a drag: the motion before it, the key, then the motion after it.
    requests.length = 0;
    pointer("pointerdown", 0, 0);
    pointer("pointermove", 72, 0);
    await advance(FRAME_INTERVAL_MS);
    pointer("pointermove", 432, 0);
    key("KeyK");
    pointer("pointermove", 504, 0);
    pointer("pointerup", 504, 0);
    await answerPosts();
    const sent = postParameters();
    assert.deepEqual(sent.map((parameters) => parameters.type), ["drag", "drag", "key", "drag"]);
    assert.ok(Math.abs(Number(sent[1].dx) - 5 * Number(sent[3].dx)) < 1e-4, JSON.stringify(sent));

    // Nothing in flight, the drag and the key in one animation frame.
    requests.length = 0;
    pointer("pointerdown", 0, 0);
    pointer("pointermove", 144, 0);
    key("KeyK");
    pointer("pointerup", 144, 0);
    await answerPosts();
    assert.deepEqual(postParameters().map((parameters) => parameters.type), ["drag", "key"]);
  },

  // The HUD's frame age is the age of the frame on screen, and a long poll that brings nothing says so.
  async theFrameAgeKeepsGrowingWithoutNewFrames() {
    await advance(FRAME_INTERVAL_MS);
    await answerFrame({ status: 200, headers: { "X-Frame-Sequence": 7, "X-Frame-Age-Ms": 12 } });
    await advance(FRAME_INTERVAL_MS);
    assert.equal(elementsById.status.textContent, "");
    for (let i = 0; i < 2; ++i) {
      await advance(2000);
      await answerFrame({ status: 204 });
      await advance(FRAME_INTERVAL_MS);
    }
    // /state never answers: the age is the page's own.
    await advance(500);
    const age = Number.parseInt(elementsById.hudAge.textContent, 10);
    assert.ok(age >= 4000, `the HUD shows a frame age of '${elementsById.hudAge.textContent}' after 4 s without a frame`);
    assert.match(elementsById.status.textContent, /No new frame for \d+ s/);
    await deliverFrame(5);
    await advance(FRAME_INTERVAL_MS);
    assert.equal(elementsById.status.textContent, "");
  },

  // A page opened before the first frame says it waits instead of showing a blank view and an empty status line.
  async aPageBeforeTheFirstFrameSaysItWaits() {
    await advance(FRAME_INTERVAL_MS);
    await advance(2000);
    await answerFrame({ status: 204 });
    await advance(FRAME_INTERVAL_MS);
    assert.match(elementsById.status.textContent, /Waiting for the first frame/);
    await deliverFrame();
    await advance(FRAME_INTERVAL_MS);
    assert.equal(elementsById.status.textContent, "");
    // A viewer that stopped answers 503: the page says so and retries.
    await advance(FRAME_INTERVAL_MS);
    await answerFrame({ status: 503, body: "the viewer has stopped\n" });
    assert.match(elementsById.status.textContent, /No frames \(503 the viewer has stopped\)/);
  },

  // theSettingsControlsShowTheRenderersSettings: below, since it loads the page with a query.

  // A drag is a fraction of the displayed height on both axes, as MuJoCo's mjv_moveCamera takes it, and a long one is
  // sent in pieces within the server's bound.
  async dragsAreFractionsOfTheDisplayedHeight() {
    pointer("pointerdown", 0, 0);
    pointer("pointermove", 100, 0);
    await advance(FRAME_INTERVAL_MS);
    pointer("pointermove", 100, 100);
    pointer("pointerup", 100, 100);
    await answerPosts();
    const [horizontal, vertical] = postParameters();
    assert.equal(horizontal.dx, (100 / 720).toFixed(5));
    assert.equal(horizontal.dy, "0.00000");
    assert.equal(vertical.dx, "0.00000");
    assert.equal(vertical.dy, (100 / 720).toFixed(5));

    // 3600 px while a POST is on its way: five heights, more than the server's 2 per drag.
    requests.length = 0;
    pointer("pointerdown", 0, 0);
    pointer("pointermove", 10, 0);
    await advance(FRAME_INTERVAL_MS);
    pointer("pointermove", 3610, 0);
    pointer("pointerup", 3610, 0);
    await answerPosts();
    const pieces = postParameters().slice(1).map((parameters) => Number(parameters.dx));
    assert.ok(pieces.length >= 3, `the long drag went out as ${JSON.stringify(pieces)}`);
    for (const piece of pieces) assert.ok(Math.abs(piece) <= 2, `a piece of ${piece} heights`);
    assert.ok(Math.abs(pieces.reduce((sum, piece) => sum + piece, 0) - 5) < 1e-3, JSON.stringify(pieces));
  },

  // Every frame request names this page, so that /state counts it as a viewer, and the HUD shows that count.
  async everyFrameRequestNamesThisPage() {
    await advance(FRAME_INTERVAL_MS);
    await deliverFrames(2);
    const frameUrls = requests.filter(isFrameRequest).map((request) => new URL(request.url, "http://viewer"));
    assert.ok(frameUrls.length >= 3);
    const viewerIds = new Set(frameUrls.map((url) => url.searchParams.get("viewer")));
    assert.equal(viewerIds.size, 1, `the frame requests name ${JSON.stringify([...viewerIds])}`);
    const [viewerId] = viewerIds;
    assert.match(viewerId ?? "", /^[0-9a-f]{16,32}$/);
    await answerState(viewerState({ viewers: 2 }));
    assert.equal(elementsById.hudClients.textContent, "2 of 4");
  },
};

// The query string each scenario loads the page with; "" for those not listed.
const searches = {
  anInputRefusalOutlivesTheFrameLoop: "?quality=high",
  theSettingsControlsShowTheRenderersSettings: "?fps=25&quality=62",
  aClampedOrRefusedSettingStillFollowsTheRenderer: "?fps=100",
};

Object.assign(scenarios, {
  // A refused POST stays on the status line while frames keep arriving, until a POST to the same route succeeds.
  async anInputRefusalOutlivesTheFrameLoop() {
    await advance(FRAME_INTERVAL_MS);
    const [settings] = pending(isPost);
    assert.ok(settings !== undefined && settings.url === "/settings?quality=high", postUrls().join());
    await answer(settings, { status: 400, body: "parameter 'quality' of /settings is not an integer: 'high'\n" });
    await advance(FRAME_INTERVAL_MS);
    await answerFrame({ status: 204 });
    await deliverFrames(5);
    assert.match(elementsById.status.textContent, /Input refused: 400 parameter 'quality'/);

    // A refused key: both messages show, and a successful input clears only its own.
    key("Digit1");
    await advance(FRAME_INTERVAL_MS);
    await answer(pending(isPost)[0], { status: 403, body: "a POST from origin 'https://tunnel.example' is refused\n" });
    await deliverFrames(5);
    assert.match(elementsById.status.textContent, /Input refused: 403 a POST from origin/);
    assert.match(elementsById.status.textContent, /'quality'/);
    key("Digit2");
    await answerPosts();
    assert.doesNotMatch(elementsById.status.textContent, /403/);
    assert.match(elementsById.status.textContent, /Input refused: 400 parameter 'quality'/);

    // A setting that succeeds clears the settings refusal.
    qualityInput.value = "80";
    qualityInput.dispatch("change");
    await answerPosts();
    assert.equal(elementsById.status.textContent, "");

    // A POST that cannot be sent at all shows until any POST succeeds.
    requests.splice(0, requests.length, ...requests.filter((request) => isFrameRequest(request) && !request.answered));
    key("Digit3");
    await advance(FRAME_INTERVAL_MS);
    await fail(pending(isPost)[0], "Failed to fetch");
    await deliverFrames(3);
    assert.match(elementsById.status.textContent, /Input not sent: Failed to fetch/);
    key("Digit4");
    await answerPosts();
    assert.equal(elementsById.status.textContent, "");
  },

  // The settings panel shows what the renderer runs at, which a bookmark or another tab may have set.
  async theSettingsControlsShowTheRenderersSettings() {
    await answerPosts();
    assert.deepEqual(postUrls(), ["/settings?fps=25&quality=62"]);
    await advance(1500);  // the render thread applies the settings
    await answerState(viewerState({ render: { targetFps: 25, quality: 62 } }));
    assert.equal(fpsSelect.value, "25", "the rate select does not show the renderer's rate");
    assert.equal(qualityInput.value, "62");
    assert.equal(elementsById.qualityValue.textContent, "62");

    // Another tab sets 20 fps: this one follows, and picking 30 again sends it.
    await pollState(viewerState({ render: { targetFps: 20, quality: 62 } }));
    assert.equal(fpsSelect.value, "20");
    fpsSelect.value = "30";
    fpsSelect.dispatch("change");
    await answerPosts();
    assert.deepEqual(postUrls().slice(-1), ["/settings?fps=30"]);
    // Until the renderer reports the new rate, a /state of the old one leaves the select alone.
    await pollState(viewerState({ render: { targetFps: 20, quality: 62 } }));
    assert.equal(fpsSelect.value, "30");

    // A control the user holds is not overwritten.
    await advance(1500);
    qualityInput.focus();
    qualityInput.value = "40";
    await pollState(viewerState({ render: { targetFps: 30, quality: 62 } }));
    assert.equal(qualityInput.value, "40");
    assert.equal(fpsSelect.value, "30");
  },

  // A setting the server clamps (a bookmark's fps=100 runs at 60) or refuses does not freeze the controls: once the
  // POST is answered and the renderer had time to apply it, they show what the renderer runs at.
  async aClampedOrRefusedSettingStillFollowsTheRenderer() {
    await answerPosts();
    assert.deepEqual(postUrls(), ["/settings?fps=100"]);
    await advance(1500);
    await pollState(viewerState({ render: { targetFps: 60 } }));
    assert.equal(fpsSelect.value, "60", "the rate select waits for a rate the server clamped away");

    fpsSelect.value = "15";
    fpsSelect.dispatch("change");
    await answerPosts({ status: 400, body: "parameter 'fps' of /settings is refused\n" });
    await pollState(viewerState({ render: { targetFps: 60 } }));
    assert.equal(fpsSelect.value, "60", "the rate select waits for a rate the server refused");
  },
});

async function main() {
  const [htmlPath, name] = process.argv.slice(2);
  const scenario = scenarios[name];
  if (scenario === undefined) {
    console.error(`unknown scenario '${name}'; the scenarios are ${Object.keys(scenarios).join(", ")}`);
    process.exit(2);
  }
  loadPage(fs.readFileSync(htmlPath, "utf8"), searches[name] ?? "");
  await settle();
  await scenario();
  console.log(`PASS ${name}`);
}

main().then(() => process.exit(0), (error) => {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
});
