import requests
import time
import json
import cv2


url = "http://127.0.0.1:5000/"
# file = open("test.jpg", "rb").read()
files = {}
# for i in range(5):
#     files["file" + str(i)] = file

cap = cv2.VideoCapture(0)
frame_count = 0
with requests.session() as s:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of input stream")
            break
        frame_count += 1
        files[f"frame_{frame_count}"] = cv2.imencode(".jpg", frame)[1].tobytes()

        # my_img = {'image': open('test.jpg', 'rb')}
        if frame_count == 5:
            start_time = time.time()
            r = s.post(url + "inference", files=files)
            print(f"FPS: {frame_count/(time.time() - start_time)}")
            frame_count = 0
            files = {}
        # res = s.post(url+"inference", files=files, json={"image_path": "test.jpg"})
        # response_time = res.elapsed.total_seconds()
        # # print(res.json())
        # print(f"Response Time: {response_time}")
