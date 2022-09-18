from load import *
from fileinput import filename
from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf

import numpy as np
import sys
import os
sys.path.append(os.path.abspath("./model"))


global graph, model

model = init()

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index_view():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict_digit():
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        flash(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded and displayed below')
        x = tf.keras.preprocessing.image.load_img('static/uploads/' + filename, color_mode='grayscale', target_size=(28,28))
        x = tf.keras.preprocessing.image.img_to_array(x)
        out = model.predict(np.array([x]))
        response = np.array(np.argmax(out,axis=1))

        return render_template('index.html', prediction_text ='Model predicted the digit as: {}'.format(response[0]), filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    pass

if __name__ == '__main__':
    app.run(debug=True, port=8000)