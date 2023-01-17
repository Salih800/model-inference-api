import requests
import time
import json


url = "http://127.0.0.1:5000/"
file = open("test.jpg", "rb").read()
with requests.session() as s:
    while True:
        my_img = {'image': open('test.jpg', 'rb')}
        res = s.post(url+"inference", files=my_img, json={"image_path": "test.jpg"})
        response_time = res.elapsed.total_seconds()
        # print(res.json())
        print(f"Response Time: {response_time}")
