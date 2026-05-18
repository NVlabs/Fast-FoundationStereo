import socket
import struct
import logging
import socketserver
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
            # Note: returning None here cannot distinguish a clean close from a
            # truncated frame mid-read.  Callers should treat None as
            # "connection gone" and terminate the session.
            return None
        buf += chunk
    return buf


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
    socketserver.ThreadingTCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer((host, port), ImageHandler) as server:
        logger.info('Listening on %s:%d', host, port)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            logger.info('Shutting down')


if __name__ == '__main__':
    main()
