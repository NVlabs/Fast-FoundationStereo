# Image RGB-to-Grayscale TCP Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a threaded TCP server that receives JPEG images, converts RGB to grayscale with OpenCV, and returns the result — plus a client module and integration tests.

**Architecture:** `socketserver.ThreadingTCPServer` with one handler thread per client. Both directions use the same framing: 4-byte big-endian length prefix followed by JPEG bytes. Protocol helpers live in `image_server.py` and are imported by `image_client.py`.

**Tech Stack:** Python 3, `socketserver` (stdlib), `socket` (stdlib), `struct` (stdlib), `threading` (stdlib), `opencv-python` (`cv2`), `numpy`, `unittest` (stdlib).

---

## File Map

| File | Role |
|------|------|
| `image_server.py` | `send_msg`, `recv_msg`, `_recv_exact`, `ImageHandler`, `main()` |
| `image_client.py` | `ImageClient` class, `main()` CLI entry point |
| `test_server.py` | `TestProtocolHelpers`, `TestImageServer`, `TestImageClient` |

---

### Task 1: Protocol framing helpers

**Files:**
- Create: `image_server.py`
- Create: `test_server.py`

- [ ] **Step 1: Write the failing tests**

Create `test_server.py`:

```python
import socket
import unittest
from image_server import send_msg, recv_msg


class TestProtocolHelpers(unittest.TestCase):
    def _pair(self):
        return socket.socketpair()

    def test_roundtrip_small(self):
        a, b = self._pair()
        try:
            payload = b'hello world'
            send_msg(a, payload)
            result = recv_msg(b)
            self.assertEqual(result, payload)
        finally:
            a.close()
            b.close()

    def test_roundtrip_binary(self):
        a, b = self._pair()
        try:
            payload = bytes(range(256)) * 100
            send_msg(a, payload)
            result = recv_msg(b)
            self.assertEqual(result, payload)
        finally:
            a.close()
            b.close()

    def test_recv_returns_none_on_closed_socket(self):
        a, b = self._pair()
        a.close()
        result = recv_msg(b)
        self.assertIsNone(result)
        b.close()


if __name__ == '__main__':
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/adiroha/repos/niv/d1_min
python -m pytest test_server.py::TestProtocolHelpers -v
```

Expected: `ModuleNotFoundError: No module named 'image_server'`

- [ ] **Step 3: Implement the protocol helpers**

Create `image_server.py`:

```python
import socket
import socketserver
import struct
import logging
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def send_msg(sock: socket.socket, data: bytes) -> None:
    header = struct.pack('>I', len(data))
    sock.sendall(header + data)


def recv_msg(sock: socket.socket) -> bytes | None:
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    length = struct.unpack('>I', header)[0]
    return _recv_exact(sock, length)


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest test_server.py::TestProtocolHelpers -v
```

Expected:
```
PASSED test_server.py::TestProtocolHelpers::test_roundtrip_small
PASSED test_server.py::TestProtocolHelpers::test_roundtrip_binary
PASSED test_server.py::TestProtocolHelpers::test_recv_returns_none_on_closed_socket
```

- [ ] **Step 5: Commit**

```bash
git add image_server.py test_server.py
git commit -m "feat: add TCP framing helpers with tests"
```

---

### Task 2: Server handler and main

**Files:**
- Modify: `image_server.py` — add `ImageHandler`, `main()`
- Modify: `test_server.py` — add `TestImageServer` with synthetic image test

- [ ] **Step 1: Add the failing integration test**

Append to `test_server.py` (before `if __name__ == '__main__':`):

```python
import threading
import socketserver
import cv2
import numpy as np
from image_server import send_msg, recv_msg, ImageHandler


def _make_server() -> socketserver.ThreadingTCPServer:
    server = socketserver.ThreadingTCPServer(('127.0.0.1', 0), ImageHandler)
    server.allow_reuse_address = True
    return server


class TestImageServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _make_server()
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=2)

    def _connect(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('127.0.0.1', self.port))
        return sock

    def _send_image(self, sock: socket.socket, img: np.ndarray) -> None:
        _, jpeg = cv2.imencode('.jpg', img)
        send_msg(sock, jpeg.tobytes())

    def _recv_gray(self, sock: socket.socket) -> np.ndarray:
        data = recv_msg(sock)
        self.assertIsNotNone(data, 'Server returned no data')
        gray = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertIsNotNone(gray, 'Response could not be decoded as image')
        return gray

    def test_synthetic_image_becomes_grayscale(self):
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        with self._connect() as sock:
            self._send_image(sock, img)
            gray = self._recv_gray(sock)
        self.assertEqual(gray.shape, (100, 100))
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
python -m pytest test_server.py::TestImageServer::test_synthetic_image_becomes_grayscale -v
```

Expected: `ImportError: cannot import name 'ImageHandler' from 'image_server'`

- [ ] **Step 3: Implement ImageHandler and main()**

Append to `image_server.py`:

```python
class ImageHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        logger.info('Connection from %s', self.client_address)
        try:
            while True:
                data = recv_msg(self.request)
                if data is None:
                    break
                img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    logger.warning('Could not decode image, closing connection')
                    break
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, jpeg = cv2.imencode('.jpg', gray)
                send_msg(self.request, jpeg.tobytes())
        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            logger.warning('Connection error: %s', exc)
        finally:
            logger.info('Connection closed: %s', self.client_address)


def main() -> None:
    host, port = '0.0.0.0', 9999
    with socketserver.ThreadingTCPServer((host, port), ImageHandler) as server:
        server.allow_reuse_address = True
        logger.info('Listening on %s:%d', host, port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info('Shutting down')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
python -m pytest test_server.py::TestImageServer::test_synthetic_image_becomes_grayscale -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add image_server.py test_server.py
git commit -m "feat: add threaded TCP image server with grayscale conversion"
```

---

### Task 3: Multiple-image and real-image tests

**Files:**
- Modify: `test_server.py` — add two more test methods to `TestImageServer`

- [ ] **Step 1: Add the tests**

Add these two methods inside the `TestImageServer` class in `test_server.py`:

```python
    def test_multiple_images_same_connection(self):
        with self._connect() as sock:
            for _ in range(3):
                img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                self._send_image(sock, img)
                gray = self._recv_gray(sock)
                self.assertEqual(gray.shape, (64, 64))

    def test_real_image(self):
        img = cv2.imread('sample/left.png')
        self.assertIsNotNone(img, 'sample/left.png must be readable')
        h, w = img.shape[:2]
        with self._connect() as sock:
            self._send_image(sock, img)
            gray = self._recv_gray(sock)
        self.assertEqual(gray.shape, (h, w))
```

- [ ] **Step 2: Run the new tests**

```bash
python -m pytest test_server.py::TestImageServer -v
```

Expected: all three tests in `TestImageServer` pass.

- [ ] **Step 3: Commit**

```bash
git add test_server.py
git commit -m "test: add multi-image and real-image integration tests"
```

---

### Task 4: Client module

**Files:**
- Create: `image_client.py`
- Modify: `test_server.py` — add `TestImageClient`

- [ ] **Step 1: Add the failing client test**

Append to `test_server.py` (before `if __name__ == '__main__':`):

```python
class TestImageClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = _make_server()
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.thread.join(timeout=2)

    def test_client_process_returns_grayscale(self):
        from image_client import ImageClient
        img = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
        with ImageClient('127.0.0.1', self.port) as client:
            gray = client.process(img)
        self.assertEqual(gray.shape, (80, 80))

    def test_client_process_multiple_images(self):
        from image_client import ImageClient
        with ImageClient('127.0.0.1', self.port) as client:
            for size in [32, 64, 128]:
                img = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
                gray = client.process(img)
                self.assertEqual(gray.shape, (size, size))
```

- [ ] **Step 2: Run to verify failure**

```bash
python -m pytest test_server.py::TestImageClient -v
```

Expected: `ModuleNotFoundError: No module named 'image_client'`

- [ ] **Step 3: Implement image_client.py**

Create `image_client.py`:

```python
import socket
import sys
import cv2
import numpy as np
from image_server import send_msg, recv_msg


class ImageClient:
    def __init__(self, host: str, port: int) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((host, port))

    def process(self, img: np.ndarray) -> np.ndarray:
        _, jpeg = cv2.imencode('.jpg', img)
        send_msg(self._sock, jpeg.tobytes())
        data = recv_msg(self._sock)
        if data is None:
            raise RuntimeError('Server closed the connection unexpectedly')
        gray = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError('Server response could not be decoded as an image')
        return gray

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> 'ImageClient':
        return self

    def __exit__(self, *_) -> None:
        self.close()


def main() -> None:
    if len(sys.argv) != 3:
        print(f'Usage: {sys.argv[0]} <input_image> <output_image>')
        sys.exit(1)
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f'Cannot read image: {sys.argv[1]}')
        sys.exit(1)
    with ImageClient('127.0.0.1', 9999) as client:
        gray = client.process(img)
    cv2.imwrite(sys.argv[2], gray)
    print(f'Saved: {sys.argv[2]}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Run the client tests**

```bash
python -m pytest test_server.py::TestImageClient -v
```

Expected:
```
PASSED test_server.py::TestImageClient::test_client_process_returns_grayscale
PASSED test_server.py::TestImageClient::test_client_process_multiple_images
```

- [ ] **Step 5: Run the full test suite**

```bash
python -m pytest test_server.py -v
```

Expected: all 9 tests pass, no failures.

- [ ] **Step 6: Commit**

```bash
git add image_client.py test_server.py
git commit -m "feat: add ImageClient with integration tests"
```

---

## Usage

**Run the server:**
```bash
python image_server.py
# Listening on 0.0.0.0:9999
```

**Run the client (in another terminal):**
```bash
python image_client.py sample/left.png output_gray.png
# Saved: output_gray.png
```

**Run all tests:**
```bash
python -m pytest test_server.py -v
```
