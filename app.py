from flask import Flask, jsonify, request, render_template
import os
import numpy as np
import tensorflow as tf
import keras

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

# Load your model (adjust path if necessary)
brain_tumour_model = keras.models.load_model("Brain_Tumour.keras")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("app.html")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = "img1.jpg"
        file.save(os.path.join(UPLOAD_FOLDER, filename))

        # Preprocess the image
        test_image = tf.keras.utils.load_img(os.path.join(UPLOAD_FOLDER, filename), target_size=(64, 64))
        test_image = tf.keras.utils.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Predict
        result = brain_tumour_model.predict(test_image)
        prediction = "You are Safe. Enjoy!(99% Accurate)🤩" if result[0][0] == 0 else "Tumour Alert. Contact Doctor (99% Accurate)😥"
        
        return render_template("app.html", prediction=prediction)

    return jsonify({'message': 'Invalid file type'}), 400

if __name__ == "__main__":
    app.run(debug=True)
