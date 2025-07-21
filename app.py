from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename

# Define the F1Score custom metric
import tensorflow.keras.backend as K

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to class labels
        y_pred = K.argmax(y_pred, axis=-1)
        y_true = K.argmax(y_true, axis=-1)

        # Calculate true positives, false positives, and false negatives
        tp = K.sum(K.cast(K.equal(y_true, y_pred), K.floatx()))
        fp = K.sum(K.cast(K.equal(y_true, 0) & K.not_equal(y_pred, y_true), K.floatx()))
        fn = K.sum(K.cast(K.equal(y_pred, 0) & K.not_equal(y_true, y_pred), K.floatx()))

        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1

    def reset_states(self):
        # Reset all variables at the end of each epoch
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)

# Initialize Flask app
app = Flask(__name__)

# Load the model with the custom metric (F1Score)
model = tf.keras.models.load_model(r".\notebook\trained_retinal_model_v2.h5", custom_objects={"F1Score": F1Score})

# Set the folder for uploaded files
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the model prediction function
def model_prediction(image_path):
    # Load the image and preprocess it
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # Preprocess the image as needed for the model
    predictions = model.predict(x)  # Get model predictions
    return np.argmax(predictions)  # Return the class with the highest probability

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
            # Save the uploaded file securely
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Get the prediction index from the model
            result_index = model_prediction(filepath)
            
            # Define the class labels
            classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
            diagnosis = classes[result_index]  # Get the diagnosis based on the prediction
            
            # Define recommendations based on the diagnosis
            recommendations = {
                'CNV': "Recommended Treatment for CNV: Anti-VEGF Therapy",
                'DME': "Recommended Treatment for DME: Anti-VEGF Therapy or Steroids",
                'DRUSEN': "Recommended Treatment for DRUSEN: Monitoring and Lifestyle Changes",
                'NORMAL': "No Treatment Required: Keep up with regular eye check-ups"
            }

            # Return the result page with the prediction and recommendation
            return render_template("result.html", prediction=diagnosis, img_path=filepath, details=recommendations[diagnosis])

    # Render the prediction page if GET request
    return render_template("predict.html")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
