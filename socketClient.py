"""Sends random json packets to server over port 5000"""

import socket
import json
from random import randint
from threading import Thread
from time import time, sleep
from typing import Dict, Any
import cv2
import base64
import numpy as np

IP = "127.0.0.1"
PORT = 5001


def generate_json_message(data: Dict) -> Dict[str, Any]:
    # """Generate random json packet with hashed data bits"""
    return {
        "id": randint(1, 100),
        "timestamp": time(),
        **data
    }


def send_json_message(
        sock: socket.socket,
        json_message: Dict[str, Any],
) -> None:
    """Send json packet to server"""
    message = (json.dumps(json_message) + '\n').encode()
    sock.sendall(message)
    # print(f'{len(message)} bytes sent')


def encode_image(image: np.ndarray) -> str:
    encoded = cv2.imencode(".jpg", image)[1].tobytes()
    base64_bytes = base64.b64encode(encoded)
    decoded = base64_bytes.decode('utf-8')
    return decoded


# class Receiver(Thread):
#     def __init__(self, sock: socket.socket):
#         super().__init__(daemon=True, name='Receiver')
#         self.sock = sock
#         self.sock_file = sock.makefile()
#         self.start()
#
#     def run(self) -> None:
#         while True:
#             data = self.sock_file.readline()
#             yield json.loads(data)

def receiver(sock: socket.socket):
    sock_file = sock.makefile()
    while True:
        data = sock_file.readline()
        yield json.loads(data)


def main() -> None:
    cap = cv2.VideoCapture(0)
    frame_count = 0

    with socket.socket() as sock:
        sock.connect((IP, PORT))
        start_time = time()
        reader = receiver(sock)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of input stream")
                break
            frame_count += 1
            fps = frame_count / (time() - start_time)
            print(f"FPS: {fps}")

            encoded = encode_image(frame)
            data = {
                    "image": encoded,
                    }
            # json_message = generate_json_message(data)
            # json_message["frame_count"] = frame_count
            # json_message["frame"] = cv2.imencode(".jpg", frame)[1].tostring()
            send_json_message(sock, data)
            print(f"Received: {next(reader)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
