import os
import socket
import threading
import unittest
import socketserver
import cv2
import numpy as np
from core_niv.niv.image_server import send_msg, recv_msg, ImageHandler

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_FIXTURE_LEFT = os.path.join(_TEST_DIR, 'sample', 'left.png')


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


def _make_server() -> socketserver.ThreadingTCPServer:
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    server = socketserver.ThreadingTCPServer(('127.0.0.1', 0), ImageHandler)
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

    def test_multiple_images_same_connection(self):
        with self._connect() as sock:
            for _ in range(3):
                img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                self._send_image(sock, img)
                gray = self._recv_gray(sock)
                self.assertEqual(gray.shape, (64, 64))

    @unittest.skipUnless(os.path.exists(_FIXTURE_LEFT), 'sample/left.png fixture not present')
    def test_real_image(self):
        img = cv2.imread(_FIXTURE_LEFT)
        self.assertIsNotNone(img, 'sample/left.png must be readable')
        h, w = img.shape[:2]
        with self._connect() as sock:
            self._send_image(sock, img)
            gray = self._recv_gray(sock)
        self.assertEqual(gray.shape, (h, w))


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
        from core_niv.niv.image_client import ImageClient
        img = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
        with ImageClient('127.0.0.1', self.port) as client:
            gray = client.process(img)
        self.assertEqual(gray.shape, (80, 80))

    def test_client_process_multiple_images(self):
        from core_niv.niv.image_client import ImageClient
        with ImageClient('127.0.0.1', self.port) as client:
            for size in [32, 64, 128]:
                img = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
                gray = client.process(img)
                self.assertEqual(gray.shape, (size, size))


class TestImageClientWebcam(unittest.TestCase):
    """Interactive webcam test.

    Disabled by default because it requires a physical camera and a display.
    Enable by setting the environment variable NIV_WEBCAM_TEST=1.
    Press 'q' in the display window to finish the test.
    """

    @classmethod
    def setUpClass(cls):
        #if os.environ.get('NIV_WEBCAM_TEST') != '1':
        #    raise unittest.SkipTest('Set NIV_WEBCAM_TEST=1 to run the webcam test')
        cls.server = _make_server()
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        #if os.environ.get('NIV_WEBCAM_TEST') != '1':
        #    return
        cls.server.shutdown()
        cls.thread.join(timeout=2)

    def test_webcam_roundtrip_display(self):
        from core_niv.niv.image_client import ImageClient
        cam_index = int(os.environ.get('NIV_WEBCAM_INDEX', '0'))
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            self.skipTest(f'Cannot open webcam at index {cam_index}')

        frames_processed = 0
        try:
            with ImageClient('127.0.0.1', self.port) as client:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        self.fail('Failed to capture frame from webcam')
                    gray = client.process(frame)
                    self.assertEqual(gray.shape, frame.shape[:2])
                    cv2.imshow('Webcam (original)', frame)
                    cv2.imshow('Server response (grayscale)', gray)
                    frames_processed += 1
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()

        self.assertGreater(frames_processed, 0)


if __name__ == '__main__':
    #unittest.main()
    t = TestImageClientWebcam()
    t.setUpClass()
    t.test_webcam_roundtrip_display()
    t.tearDownClass()
