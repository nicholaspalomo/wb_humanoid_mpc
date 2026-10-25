# web_viewer: the MuJoCo viewer in a browser tab

The HTTP side of the simulator's `web` viewer. The robot process renders the MuJoCo scene offscreen on a thread of its
own (EGL, `robot_runtime/mujoco_sim_interface`), encodes each frame as JPEG and hands it to this package's server,
which serves a page that pulls the frames and sends the mouse and keyboard back to the renderer. There is no X server,
no VNC and no window: open `http://localhost:6090` in a browser.

The package is MuJoCo-free and EGL-free:

| Piece | What it is |
|---|---|
| `WebViewerServer` | The HTTP server ([cpp-httplib](https://github.com/yhirose/cpp-httplib), plain HTTP only). |
| `ViewerFrameMailbox` | The render thread's newest frame and status, the render demand and the client count. |
| `ViewerInputQueue`, `ViewerInput`, `parseViewerInput()` | The browsers' camera and marker input, checked and bounded. |
| `JpegEncoder`, `JpegDecoder` | libjpeg-turbo (TurboJPEG 3, SIMD). The decoder is for tests: it is the testonly `:jpeg_decoder`, outside the library. |
| `placeCurrentThread()` | Pins a viewer thread to the viewer cores and runs it under `SCHED_IDLE` (`ViewerThreadPriority`). |
| `webViewerPage()` | `page/index.html`, compiled into the binary by `tools/embed_page.py`. |

## Opening it

With `--viewer=web` the robot process logs `serving the MuJoCo viewer at http://127.0.0.1:6090/`. On the machine that
runs the simulation, open that address. From another machine, forward the port first, for example
`ssh -L 6090:127.0.0.1:6090 <host>`, or let VS Code forward it; any local port works, because the server checks the
host name and not the port. Open it in a browser tab (VS Code's Ports view: "Open in Browser"): an editor preview that
frames the page shows a blank pane, since the viewer refuses to be framed (see the trust model). Click the view to give
it the keyboard.

| Input | Effect (as in the GLFW window) |
|---|---|
| Left drag | Rotate the camera; with shift, about the vertical axis only |
| Right drag | Move the camera; with shift, horizontally |
| Middle drag, wheel | Zoom |
| `0`-`5` | Toggle the geom groups |
| `k` | Toggle between the tracking and the free camera |
| `a`-`z` | The visualizations' hotkeys |
| `p` | Show the markers panel (the GLFW viewer prints the hotkeys to the log instead) |

The toolbar resets the camera, switches tracking, saves the current frame as a PNG, and opens the markers panel (click
a marker to toggle it) and the settings (frame size, "fit window", frames per second, quality). A bookmark can carry
settings: `http://localhost:6090/?width=960&height=540&fps=20&quality=60`. The HUD shows the frames shown and rendered
per second, the size, the camera mode, the simulation time, the age of the frame on screen and the number of viewers:
the tabs that pulled a frame within the last second (each names itself with a random id) and the open streams.

The camera and the settings are shared: every tab shows the same view, the last input wins, and each tab's settings
controls show what the renderer runs at (the server clamps a value out of range). No input reaches the simulation: see the
trust model.

## Routes

<!-- LINT.IfChange(web_viewer_routes) -->
| Route | Answer |
|---|---|
| `GET /` | The page (`text/html`, `Cache-Control: no-store`). |
| `GET /frame.jpg?after=<sequence>&viewer=<id>` | The newest frame once its sequence exceeds `after`: at once if it already does, else as soon as one is published, or `204 No Content` after 2 s. When that frame is stale (published before this request found the renderer without demand, no client admitted and no frame requested within the last second, and older than two frame periods: the renderer was idle and this request has woken it; a busy renderer's frame is never stale), the next frame instead, within 2 s, or else the stale one. A sequence beyond the newest (a page left open across a restart) counts as 0. `viewer`, optional, is the page's id, 1 to 32 lower-case hex digits, which `/state` counts. Headers: `X-Frame-Sequence`, `X-Sim-Time` (seconds), `X-Frame-Age-Ms` (since the renderer published it). |
| `GET /frame.jpg` | The newest frame unless it is stale, else the next one within 2 s, else `503`. |
| `GET /stream.mjpg` | `multipart/x-mixed-replace; boundary=mujocoframe`, chunked; each part has `Content-Type: image/jpeg`, `Content-Length`, `X-Frame-Sequence` and `X-Sim-Time`. For `curl` and an address typed or bookmarked in the browser; another page's `<img>` or link is refused (see the trust model). |
| `GET /state` | JSON: `clients` (frame requests waiting and streams open at that instant, the count `maxClients` bounds), `viewers` (page ids seen on a frame request within the last second, plus the open streams), `maxClients`, `droppedInputs`, `frame` (`sequence`, `simTime`, `width`, `height`, `ageMs`, or `null`) and `viewer` (`camera`, `render`, `geomGroups`, `visualizations`; `viewerStatusJson()`). |
| `POST /input?type=drag&button=left\|middle\|right&shift=0\|1&dx=&dy=` | A drag, `dx` and `dy` both as fractions of the displayed height, as `mjv_moveCamera` takes them (the GLFW viewer's `cameraDragFractions()`), clamped to [-2, 2]; the page sends a longer drag in pieces. |
| `POST /input?type=scroll&notches=` | Wheel notches with GLFW's sign, clamped to [-10, 10]: positive is the wheel turned away from the user (GLFW's `yoffset` > 0, a negative `WheelEvent.deltaY`) and zooms out, as in MuJoCo's own viewers. |
| `POST /input?type=key&key=` | One of `0`-`5` and `a`-`z`. |
| `POST /input?type=reset_camera` | The camera of start-up. |
| `POST /input?type=camera&mode=free\|tracking` | The camera mode. |
| `POST /input?type=visualization&name=` | Toggles the visualization of that registry name. |
| `POST /settings?width=&height=&fps=&quality=` | Any of the four, clamped: width 160-3840, height 120-2160 (the renderer caps both at its maximum size), fps 1-60, quality 10-95. |
<!-- LINT.ThenChange(//robot_runtime/web_viewer/page/index.html:web_viewer_routes, //robot_runtime/web_viewer/src/WebViewerServer.cpp:web_viewer_routes, //robot_runtime/web_viewer/src/ViewerInputParsing.cpp:web_viewer_routes) -->

A POST answers `204`; a malformed parameter, an unknown or repeated one (also with the same value, or a name spelled
with escapes), a non-finite number, a key outside the set or a request body answers `400` with a message naming it. A frame request beyond the client limit answers `503 too many viewers (N)`. Any
other path answers `404`, another method `405` with `Allow`.

## The pull protocol

The page keeps exactly one `GET /frame.jpg?after=<last sequence>` in flight: it draws the answer on a canvas
(`createImageBitmap`), waits for the next animation frame and asks again. So at most one frame is ever on its way, and
the age of the frame on screen stays below one round trip plus one frame period plus the frame's own transfer, through
any tunnel. A push stream cannot promise that: an SSH or VS Code forward drains the server's socket at once into a
channel buffer of megabytes, which the server cannot see, so a stream of frames faster than the tunnel queues up
seconds of them (`web_viewer_latency_test` shows both, through a 1 MB/s relay with an 8 MiB buffer). A hidden tab
stops pulling, and the renderer idles a second later.

Input is coalesced the same way: at most one POST per animation frame and one in flight. Drag and wheel motion between
two discrete inputs (a key, a button, a setting) keeps adding up while a POST is on its way and goes with the next one;
every input keeps its order relative to the motion made before it, so `Reset camera` right after a drag resets.

## Threads and back-pressure

Nothing here runs on the physics or the realtime thread, and nothing blocks them.

- **The render thread** (in `mujoco_sim_interface`) waits in `ViewerFrameMailbox::waitForDemand()` while nobody
  watches: a frame client is admitted, or a frame was requested less than a second ago (`kDemandLease`). With no client
  it renders nothing. Then it drains the `ViewerInputQueue`, renders, encodes and calls `publish()`, which takes the
  mailbox's mutex only to swap the frame in; the previous frame is freed after it.
- **The listen thread and the request pool** (3 x `maxClients` + 2 threads, 14 for the default 4) are placed with
  `placeCurrentThread()` (the viewer cores, `SCHED_IDLE`) before they start, so the pool inherits the placement. A
  request copies the frame it sends under the mailbox's mutex (about 10 us for 100 KB) and writes it outside; a slow
  client skips frames and holds only its own request thread, never the renderer.
- A waiting request (a long poll, or a stream between two frames) waits on the mailbox's condition variable in slices
  of 250 ms, so that the server's destructor ends it within 250 ms.
- Frames are written in pieces of 4 KiB. A writable TCP socket has at least that much room, so a client that stops
  reading blocks its request thread only in the writability check, for the 2 s write timeout, and never first in a
  `send()` of the rest of a frame; the request thread then drops the client and frees its slot. Between two pieces the
  thread checks whether the server is stopping.
- **Shutdown**: the owner closes the mailbox (every waiter returns at once), then destroys the server, which stops
  accepting, ends the open requests (a write blocked on a stuck client within the 2 s write timeout) and joins every
  thread. The port can be bound again at once.

cpp-httplib's `Server` constructor sets `SIGPIPE` to ignored for the whole process, as servers do; a write to a closed
connection then fails with `EPIPE` instead of killing the process.

## Trust model

The page shows the simulation and moves the viewer's camera and markers, and nothing else. No input reaches the
physics, the controller, the gantry or its weld, or the bus: those stay with the operator GUI on the bus. The server
has no authentication and no TLS.

- **The default bind is 127.0.0.1**: only this machine reaches it. For another machine use an SSH tunnel or VS Code's
  port forwarding, which keep it that way, and open the forwarded address in a browser tab (the Ports view: "Open in
  Browser"). An editor preview that frames the page ("Preview in Editor" or Simple Browser in Antigravity, Cursor,
  vscode.dev or VS Code before its integrated browser) shows a blank pane, because the viewer refuses to be framed and
  refuses cross-site requests; VS Code's desktop integrated browser opens it as a top-level page and works.
- **Another address is an explicit opt-in** (`--web_viewer_address=0.0.0.0`, an interface address or `::`), needed in a
  bridge-networked container (Docker Desktop). The server logs a warning: anyone who can reach the address can watch
  the simulation and move its camera, and can deny the viewer (not the simulation) by holding its request threads (see
  Limits). The checks below stay on.
- **Checks before routing, on every bind and before a client is counted**:
  - **Host**: the host name of the `Host` header, without its port, must be `localhost`, `127.0.0.1`, `::1`, the bound
    address when it is not a wildcard, or one of the allowed hosts (`--web_viewer_allowed_hosts`, a comma list, empty by
    default, of names without ports: the server refuses `robot.lan:6090` or `[fe80::1]:6090` at start-up). An IPv6
    address matches in any spelling, on either side (`2001:0DB8:0:0::1` and the browser's `[2001:db8::1]`). Anything
    else, the rest of 127.0.0.0/8 included, answers `403`. This defeats DNS rebinding: another site's name never passes.
  - **Fetch metadata**: `Sec-Fetch-Site: cross-site` or `same-site` answers `403`, so another site's
    `<img src="http://localhost:6090/stream.mjpg">` takes no client slot and wakes no renderer. `same-origin`, `none`
    (a typed address, a bookmark) and no header (`curl`) pass. A link from a page on another local port (noVNC, Rerun)
    is `same-site` and refused too: type or bookmark the address.
  - **POST** needs `X-MuJoCo-Viewer: 1`, which a form cannot send and a cross-origin script can send only after a CORS
    preflight that the server never grants (it sends no `Access-Control-Allow-*` header, ever), and an `Origin`, when
    there is one, equal to `http://<Host>` or `https://<Host>` (an HTTPS-terminating forward that keeps the Host).
  - Every answer carries `Cross-Origin-Resource-Policy: same-origin`, `X-Frame-Options: DENY`,
    `Content-Security-Policy: frame-ancestors 'none'` and `X-Content-Type-Options: nosniff`.
- **Residual risk of a non-loopback bind**: browsers send no `Sec-Fetch-*` headers to a plain-HTTP origin that is not
  loopback, so with the LAN address among the allowed hosts another site can make a visitor's browser send short
  requests to the viewer. CORP makes the browser discard the answers (and abort a stream), Chrome's Private Network
  Access blocks requests from public sites to private addresses, and the worst a request can do is move the camera.
  Prefer a tunnel to a LAN bind.

## Limits

<!-- LINT.IfChange(web_viewer_limits) -->
| Limit | Value |
|---|---|
| Request body | 1 KiB (`413` beyond); the routes take query parameters only, and a POST with a body answers `400` |
| Request line and headers | cpp-httplib's caps: an 8 KiB request URI, 8 KiB per header line, 100 headers |
| Frame clients at a time | `maxClients`, 4 by default; a waiting `GET /frame.jpg` and an open `GET /stream.mjpg` count |
| Request threads | 3 x `maxClients` + 2; 16 more accepted connections may wait for one, beyond that a connection is closed. A connection holds its thread for its whole life, and a client that trickles its header bytes (or reads an answer a few bytes at a time) holds it for as long as it likes, so that many such connections stop every page's frames, `/state` and input until they close |
| Keep-alive | 1 s idle, 100 requests per connection |
| Timeouts | read 5 s and write 2 s, each bounding one read or write of a socket, not a whole request; a long poll waits 2 s |
| Input queue | 256 events, the oldest dropped first (`droppedInputs` in `/state`); a drag merges into the drag before it while the sum stays within one drag's [-2, 2] |
| Values | `key` in [0-5a-z], drags in [-2, 2], notches in [-10, 10], a visualization name of the registry |
<!-- LINT.ThenChange(//robot_runtime/web_viewer/src/WebViewerServer.cpp:web_viewer_limits) -->

## Building and testing

```bash
bazel test //robot_runtime/web_viewer/...
```

The page is `page/index.html`, a self-contained file with no external scripts, styles or fonts; the genrule
`:web_viewer_page_source` compiles it into the library, so a change to it needs a rebuild of the robot binary. The
tests use synthetic frames and raw sockets, no MuJoCo and no EGL; `web_viewer_latency_test` is timing-sensitive and runs
alone (`exclusive`).
