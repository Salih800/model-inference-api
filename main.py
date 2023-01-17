# using flask_restful
import time

import numpy as np
from flask import Flask, jsonify, request
from flask_restful import Resource, Api

from src.detection_models import FPBlurring

import cv2

# creating the flask app
app = Flask(__name__)
# creating an API object
api = Api(app)


# making a class for a particular resource
# the get, post methods correspond to get and post requests
# they are automatically mapped by flask_restful.
# other methods include put, delete, etc.
class Inference(Resource):
    # corresponds to the GET request.
    # this function is called whenever there
    # is a GET request for this resource
    # fp_model = FPBlurring()

    def get(self):
        return jsonify({'message': 'hello world'})

    # Corresponds to POST request
    def post(self):
        files = request.files

        print(f"Files: {type(files)}, {type(files)}")
        # img = request.files['image']
        # image = cv2.imdecode(np.frombuffer(img.read(), np.uint8), cv2.IMREAD_COLOR)
        # print(f"Image shape: {image.shape}")
        result_time = time.time()
        # result = fp_model.get_pandas_l/ist(image)
        # print(f"Result Type: {type(result[0].to_dict(orient='records'))}")
        result_time = time.time() - result_time
        # j_data = request.json  # status code
        return jsonify({'message': "Hello World", 'time': result_time})
        # return jsonify({'data': data}), 201


# adding the defined resources along with their corresponding urls
api.add_resource(Inference, '/inference')

# driver function
if __name__ == '__main__':
    fp_model = FPBlurring()
    app.run(debug=True, threaded=True)
