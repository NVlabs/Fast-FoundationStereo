import socket
import sys
import cv2
import numpy as np
from niv.image_server import send_msg, recv_msg


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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open webcam', file=sys.stderr)
        sys.exit(1)
    with ImageClient('127.0.0.1', 9999) as client:
        while True:
            ret, frame = cap.read()
            if not ret:
                print('Failed to capture frame', file=sys.stderr)
                break
            gray = client.process(frame)
            cv2.imshow('Original', frame)
            cv2.imshow('Grayscale (server)', gray)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release() 
    cv2.destroyAllWindows()

 
if __name__ == '__main__':
    main()
