from flask import Flask, render_template, request
from keras.models import load_model
from keras.backend import set_session
from skimage.transfrom import resize
import tensorflow as tfH
import numpy as np

global sess
sess = tf.Session()
set_session(sess)
global model
model = load_model('<<model.h5>>')
global graph
graph = tf.get_default_graph()


app = Flask('CovidX')

@app.route('/')
def index():
    #css(flips cards, style), home.html
    return render_template('home.html')

@app.route('/mri_scan', methods=['POST'])
def request_fisier():
    form = request.form
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template("<<fisier_request.html>>")
