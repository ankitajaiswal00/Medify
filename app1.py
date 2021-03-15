import os
from flask import Flask
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__ )
app.config["TEST_IMGS"] = os.path.join('static','color_blind')

	
@app.route('/')
def home():
	test_imgs = ['t1.jpg', 't2.jpg','t3.jpg','t4.jpg']
	return render_template('sample.html')

@app.route('/static/color_blind/<test_imgs>')
def display_test_imgs(test_imgs):
    return send_from_directory(app.config['TEST_IMGS'], test_imgs)

@app.route('/predict', methods=['POST'])
def predict():
	user_inputs = [int(x) for x in request.form.values()]
	print(user_inputs)
	return render_template('sample.html')




if __name__ == "__main__":
    app.run()


# from flask import Flask, render_template, redirect, request, jsonify
# import os
# import subprocess, base64
# import cv2
# from PIL import Image
# from utils import img2heatmap
# from alzheimer.classification.classification import predict
# from aptos.inference_aptos import predict_aptos
# from melanoma.inference_melanoma import predict_melanoma
# from pneumonia.inference_pneumonia import predict_pneumonia
# from breast_cancer.inference_brest import predict_breast
# import numpy as np
# import pickle

# app = Flask(__name__)

# @app.route('/')
# def home():
# 	return render_template('sample_form.html')

# @app.route('/formpp', methods=['POST'])
# def sample():
# 	int_features = [x for x in request.form.values()]
# 	print(int_features)
# 	ls = [type(x) for x in int_features]
# 	print(ls)
# 	print(len(int_features))
# 	return render_template('sample_form.html')

# if __name__ == "__main__":
# 	app.run(debug=True)
