# Dogs breed imagerecognition CNN 

## 1. Background

A dog shelter needs an automatic system to identify dog breeds and possibly match hybrids to the most probable breeds.

The project goal is to build a ML pipeline that predicts the breeds and associated probabilities from a photo of the dog, and deploy it to an API.

<img width="250" alt="Cane" src="https://github.com/opsabarsec/dogs_breed_imagerecognition/blob/master/predicted.png">

In a further step an android app can be built using Tensorflow Lite.

## 2. The data

Having a good training dataset is a huge step towards the robust model. There is [Stanford Dogs Dataset](https://www.kaggle.com/jessicali9530/stanford-dogs-dataset) with ~20K images of dogs of 120 breeds. Every image in the dataset is annotated with the breed of a dog displayed on it. 

## 3 The model
[Full code](https://github.com/opsabarsec/Dog-pictures-classification-by-Deep-Learning/blob/master/notebooks/Stanford_dogs_classifier_part2.ipynb)
Based on Xception convolutional neural network. 

![training](CNNtraining.png)

provides 85% accuracy. 

The model has been saved to a file and can be loaded in a matter of seconds, then run on [Google Colabs](https://colab.research.google.com/drive/1kcAFOSreOd_68WF5gdvoQOceld7cAwrX) to identify the breed of any uploaded dog picture. The same has been deployed to an API, with the result illustrated in this [video](https://www.youtube.com/watch?v=1YKuf0ddEGE). 

Full explanation is provided in a [presentation](https://github.com/opsabarsec/Dog-pictures-classification-by-Deep-Learning/blob/master/documentation/presentation/P6_presentation.pdf).

The Android app needs further work to improve its accuracy since it runs on camera signal rather than on a picture.


