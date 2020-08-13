import os
import pandas as pd


import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as image
import matplotlib.image as mpimg


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import h5py
import pickle
from keras.applications.xception import Xception, preprocess_input

# fonction de prediction

def file_predict(filename):
    # download and save
    img = Image.open(filename)
    img = img.convert('RGB')
    img = img.resize((299, 299))
    img.save(filename)
    # show image
    plt.figure(figsize=(4, 4))
    plt.imshow(img)
    plt.axis('off')
    # predict
    img = image.imread(filename)
    img = preprocess_input(img)
    probs = model1.predict(np.expand_dims(img, axis=0))
    for idx in probs.argsort()[0][::-1][:5]:
        print("{:.2f}%".format(probs[0][idx]*100), "\t", label_maps_rev[idx].split("-")[-1])

#fonction principale
def main(filename):
    pickle_in = open("./labels_dict.pickle","rb")
    label_maps_rev = pickle.load(pickle_in)
    model1 = tf.keras.models.load_model("./my_model.h5")
    file_predict(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict breed')
    parser.add_argument('--filename', metavar='path', required=True,
                        help='the path to the picture')
    #parser.add_argument('--height', metavar='int', required=True,
    #                    help='your height')
    
    args = parser.parse_args()
    print(args)
    main(filename=args.filename)


#Pour éxecuter la commande : 
# 1) se placer dans le repertoir qui contient le main.py
# 2) Lancer la commande : python main.py --filename "C://marco..."
