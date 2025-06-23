from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
from recommendation import cnv, dme, drusen, normal
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = tf.keras.models.load_model(r".\notebook\trained_retinal_model.h5")

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def model_prediction(image_path):
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    return np.argmax(predictions)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            result_index = model_prediction(filepath)
            classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
            diagnosis = classes[result_index]

            recommendations = [cnv, dme, drusen, normal]
            return render_template("result.html", prediction=diagnosis, img_path=filepath, details=recommendations[result_index])

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)
