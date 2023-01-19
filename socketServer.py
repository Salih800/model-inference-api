"""
Server-Side that receives json packets from client over the network using port
5000. Then it saves the json packet to a file with the filename of current time.
"""
import random
import string
import base64
import time
import logging
from socketserver import StreamRequestHandler, TCPServer, ThreadingTCPServer

import json
from threading import Thread

from typing import Dict, Any

import cv2
import numpy as np

from src.detection_models import FPBlurring

from PIL import Image


def generate_json_message(data: Any) -> bytes:
    """Generate random json packet"""
    json_message = (json.dumps(data) + "\n").encode()
    return json_message


def decode_image(image_string: str) -> Image:
    recovered = base64.b64decode(image_string.encode())

    jpg_as_np = np.frombuffer(recovered, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    return img[:, :, ::-1]


class DumpHandler(StreamRequestHandler):
    def __str__(self):
        return f"{self.client_address} | FPS: {self.get_fps()}"

    def init(self):
        clients.append(self)
        print('connection from {}:{}'.format(*self.client_address))
        self.frame_count = 0
        self.start_time = time.time()

    def handle(self) -> None:
        """receive json packets from client"""
        self.init()
        try:
            # frame_count = 0
            # start_time = time.time()
            while True:
                data = self.rfile.readline()
                if not data:
                    break
                json_string = json.loads(data.decode().rstrip())
                image = decode_image(json_string["image"])
                # try:
                #     result = fp_model.get_pandas_list(image)
                # except RuntimeError as e:
                #     logging.warning(f"Image shape: {image.shape} | size: {image.size}")
                #     logging.error(e, exc_info=True)
                #     continue
                self.frame_count += 1
                # if frame_count == 1:
                #     start_time = time.time()
                # fps = frame_count / (time.time() - start_time)
                # print(f"{self.client_address} | FPS: {fps}")
                # print(result)
                # print(f"Type: {type(result)}")
                # json_dict = {"result": result}
                # self.wfile.write(generate_json_message(json_dict))

        except Exception as e:
            logging.error(f"{self.client_address}: {e}", exc_info=True)
        finally:
            print('disconnected from {}:{}'.format(*self.client_address))
            clients.remove(self)

    def get_fps(self):
        return round(self.frame_count / (time.time() - self.start_time), 2)


def main() -> None:
    server_address = ('localhost', 5003)
    print('starting up on {}:{}'.format(*server_address))
    with ThreadingTCPServer(server_address, DumpHandler) as server:
        print('waiting for a connection')
        server.serve_forever()


clients = []


class Checker(Thread):
    def __init__(self):
        super().__init__(daemon=True, name="Checker")
        self.start()

    def run(self):
        while True:
            print(f"Number of clients: {len(clients)} | {list(map(str, clients))}")
            time.sleep(5)


Checker()


if __name__ == '__main__':
    try:
        # fp_model = FPBlurring()
        main()
    except KeyboardInterrupt:
        pass