import logging
import json
import time
import torch

from constants.models import FP_BLURRING_MODEL
from constants.folders import WEIGHTS_FOLDER
from src.singleton import Singleton
import cv2


class DetectionModel:
    def __init__(self, model_config):
        self.model_name = model_config.model_name
        self.img_size = model_config.img_size
        self.conf_thres = model_config.conf_thres
        self.max_det = model_config.max_det

        self.total_detection_count = 0
        self.total_detect_time = 0

        self.model = None

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        if self.model is None:
            self.load_model()

    def get_model_config(self):
        return {
            "model_name": self.model_name,
            "img_size": self.img_size,
            "conf_thres": self.conf_thres,
            "max_det": self.max_det,
            "device": self.device
        }

    def load_model(self):
        self.model_name += ".pt" if self.device == "cuda:0" else ".onnx"
        # self.model_name += ".onnx"
        # self.device = "cpu"
        self.model = torch.hub.load("./yolov5", 'custom', source='local',
                                    path=WEIGHTS_FOLDER + self.model_name,
                                    device=self.device)
        self.model.conf = self.conf_thres
        self.model.max_det = self.max_det
        logging.info(f"Loaded Model Config: {self.get_model_config()}")

    def detect(self, image_path):
        detect_time = time.time()
        result = self.model(image_path, self.img_size)
        self.total_detect_time += time.time() - detect_time
        self.total_detection_count += 1
        return result

    def detect_buffer(self, buffer):
        # images = []
        # for image in buffer:
        #     images.append(image[..., ::-1])
        results = self.detect(buffer)
        results.ims = buffer
        return results

    def detect_buffer_onnx(self, image):
        result = self.detect(image[..., ::-1])
        result.ims = image
        return result

    def get_json_result(self, image_path):
        result = self.detect(image_path)
        json_results = []
        for res in result.pandas().xyxy:
            if len(result.pandas().xyxy) > 1:
                json_results.append(json.loads(res.to_json(orient="records")))
            else:
                json_results = json.loads(res.to_json(orient="records"))

        return json_results

    def get_pandas_list(self, images):
        result = self.detect(images)
        return result.pandas().xyxy[0].to_dict(orient="records")

    def get_pandas_result(self, image_path):
        result = self.detect(image_path)
        return result.pandas().xyxy[0]

    def get_average_detection_time(self):
        return round(self.total_detection_count / self.total_detect_time,
                     2) if not self.total_detection_count <= 0 else 0


class FPBlurring(DetectionModel, metaclass=Singleton):
    def __init__(self):
        super().__init__(model_config=FP_BLURRING_MODEL)

    def make_blur(self, image_path):
        image = cv2.imread(image_path)
        df = self.get_pandas_result(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        width, height = image.shape[:2]
        kernel_width = (width // 7) | 1
        kernel_height = (height // 7) | 1
        for index, row in df.iterrows():
            x, y, xMax, yMax = row[0], row[1], row[2], row[3]
            x, y, xMax, yMax = int(x), int(y), int(xMax) + 1, int(yMax) + 1

            item = image[y: yMax, x: xMax]
            blur = cv2.GaussianBlur(item, (kernel_width, kernel_height), 0)
            image[y: yMax, x: xMax] = blur
        return image

    def make_live_blur(self, image):
        df = self.get_pandas_result(image[..., ::-1])
        width, height = image.shape[:2]
        kernel_width = (width // 7) | 1
        kernel_height = (height // 7) | 1
        for index, row in df.iterrows():
            x, y, xMax, yMax = row[0], row[1], row[2], row[3]
            x, y, xMax, yMax = int(x), int(y), int(xMax) + 1, int(yMax) + 1

            item = image[y: yMax, x: xMax]
            blur = cv2.GaussianBlur(item, (kernel_width, kernel_height), 0)
            image[y: yMax, x: xMax] = blur
        return image

    @staticmethod
    def blur_result(detection_result):
        image, result = detection_result

        result_width, result_height = image.shape[:2]
        kernel_width = (result_width // 7) | 1
        kernel_height = (result_height // 7) | 1
        for index, row in result.iterrows():
            x, y, xMax, yMax = row[0], row[1], row[2], row[3]
            x, y, xMax, yMax = int(x), int(y), int(xMax) + 1, int(yMax) + 1

            item = image[y: yMax, x: xMax]
            blurred = cv2.GaussianBlur(item, (kernel_width, kernel_height), 0)
            image[y: yMax, x: xMax] = blurred
        return image

    @staticmethod
    def blur_result_onnx(detection_result):
        image, result = detection_result.ims, detection_result.pandas().xyxy[0]

        result_width, result_height = image.shape[:2]
        kernel_width = (result_width // 7) | 1
        kernel_height = (result_height // 7) | 1
        for index, row in result.iterrows():
            x, y, xMax, yMax = row[0], row[1], row[2], row[3]
            x, y, xMax, yMax = int(x), int(y), int(xMax) + 1, int(yMax) + 1

            item = image[y: yMax, x: xMax]
            blurred = cv2.GaussianBlur(item, (kernel_width, kernel_height), 0)
            image[y: yMax, x: xMax] = blurred
        return image
