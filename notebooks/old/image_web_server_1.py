# T81-558: Applications of Deep Neural Networks
# Modified by Marco Berta
# Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# 
# Deploy simple Keras tabular model with Flask only.
from flask import Flask, request, jsonify,send_from_directory
import uuid
import pickle
import os
from tensorflow.keras.models import load_model
import numpy as np
import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import MobileNet
from PIL import Image, ImageFile
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions

UPLOAD_FOLDER = '/media/marco/DATA/OC_Machine_learning/section_6/DATA/dogs'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
IMAGE_WIDTH = 299
IMAGE_HEIGHT = 299
IMAGE_CHANNELS = 3


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

pickle_in = open("labels_dict.pickle","rb")
label_maps_rev = pickle.load(pickle_in)


model = load_model("my_model.h5")

@app.route('/', methods=['GET'])
def send_index():
    return send_from_directory('./www', "index.html")

@app.route('/<path:path>', methods=['GET'])
def send_root(path):
    return send_from_directory('./www', path)

@app.route('/api/image', methods=['POST'])
def upload_image():
  # check if the post request has the file part
  if 'image' not in request.files:
      return jsonify({'error':'No posted image. Should be attribute named image.'})
  file = request.files['image']

  # if user does not select file, browser also
  # submit a empty part without filename
  if file.filename == '':
      return jsonify({'error':'Empty filename submitted.'})
  if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      x = []
      ImageFile.LOAD_TRUNCATED_IMAGES = False
      img = Image.open(BytesIO(file.read()))
      img.load()
      img = img.resize((IMAGE_WIDTH,IMAGE_HEIGHT),Image.ANTIALIAS)    
      x = image.img_to_array(img)
      x = np.expand_dims(x, axis=0)
      x = preprocess_input(x)
      x = x[:,:,:,0:3]
      probs = model.predict(np.expand_dims(img, axis=0))
      for idx in probs.argsort()[0][::-1][:5]:
        print("{:.2f}%".format(probs[0][idx]*100), "\t", label_maps_rev[idx].split("-")[-1])

      
      return jsonify("{:.2f}%".format(probs[0][idx]*100), "\t", label_maps_rev[idx].split("-")[-1])
  else:
      return jsonify({'error':'File has invalid extension'})

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True)
