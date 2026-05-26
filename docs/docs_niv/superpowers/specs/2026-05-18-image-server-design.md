# Image RGB-to-Grayscale TCP Server — Design Spec

**Date:** 2026-05-18  
**Status:** Approved

---

## Overview

A Python TCP server that accepts image data from remote clients, converts each image from RGB to grayscale using OpenCV, and returns the result. Uses the existing `opencv-python` dependency — no new packages required.

---

## Files

| File | Purpose |
|------|---------|
| `image_server.py` | Server: listens for connections, receives images, converts, sends back |
| `image_client.py` | Client: sends a color image, receives and saves the grayscale result |
| `test_server.py` | Tests: integration tests using a real server on a free port |

---

## Architecture

`socketserver.ThreadingTCPServer` with a `BaseRequestHandler` subclass. Each client connection runs in its own thread, allowing multiple concurrent clients. Image conversion is handled by `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`.

```
Client                          Server (one thread per connection)
------                          ----------------------------------
encode image as JPEG
send [4-byte length][JPEG]  -->  recv length header
                                 recv N bytes of JPEG
                                 decode JPEG → BGR numpy array
                                 cvtColor BGR → GRAY
                                 encode GRAY as JPEG
                            <--  send [4-byte length][JPEG]
recv length header
recv N bytes of JPEG
decode → grayscale image
```

A single connection can carry multiple request/response pairs before closing.

---

## Wire Protocol

Every message in both directions:

```
[4 bytes, big-endian uint32: payload length][N bytes: JPEG image data]
```

- **Length header:** `struct.pack('>I', len(jpeg_bytes))` / `struct.unpack('>I', header)[0]`
- **Image encoding:** JPEG (OpenCV default quality)
- **Direction:** identical framing for both client→server and server→client

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Client disconnects mid-transfer | Handler catches `ConnectionResetError` / partial read, logs, closes socket |
| Corrupt or non-image bytes | `cv2.imdecode` returns `None`; handler logs a warning and closes connection |
| `KeyboardInterrupt` on server | `server.shutdown()` called in `finally` block; exits cleanly |

---

## Testing (`test_server.py`)

Uses Python `unittest` — no new dependencies. Each test class starts a real `ThreadingTCPServer` on a random free port (port `0`) in a `setUpClass` background thread and shuts it down in `tearDownClass`.

| Test | What it verifies |
|------|-----------------|
| `test_synthetic_image` | 100×100 synthetic RGB → response decodes as valid grayscale (shape `(100, 100)`) |
| `test_multiple_images_same_connection` | 3 images sent sequentially on one socket, all responses valid |
| `test_real_image` | Sends `sample/left.png` (known to exist); verifies grayscale response |

---

## Non-Goals

- No authentication or TLS (local / trusted network assumed)
- No image format negotiation (JPEG only)
- No retry logic on the client
