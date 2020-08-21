# Final Project - Facial Keypoints Detection

By Ernesto Oropeza and Michael Hubert


## Introduction 

### Problem Statement
The goal of this project is to accurately predict the x and y coordinates for several facial features based on the image of a person's face. These features include multiple coordinates for the eye, mouth, eyebrow, and nose. Successfully predicting these coordinates with little error is the basis for tracking faces and facial recognition technologies. We will build, train, and test a model to predict these features with the lowest Root Mean Square Error (RMSE) possible. Our goal is to develop a model that will have an RMSE that is in the top 100 for Kaggle Competition leaderboard.

### Model Overview
The model we will use for the predicting the facial features is a Convolutional Neural Network (CNN). CNNs have proven to be very successful at identifying patterns in images and deliver state of the art performance. In addition to identifying which types of layers (2D convolutional, dense, etc.) we will need in the network, we will need to determine the number of layers as well as the number of neurons and activation function for each layer. Moreover, for the convolutional layers we need to evaluate the size of our filters and understand if zero padding will be needed to maintain the dimensions of the input image throughout the network.

The key parameter for the model will be the weights for each neuron connection of the network which we will optimize using an optimization algorithm similar to stochastic gradient descent to minimize our root mean squared error loss function. We will fit the model using training data and test the model's performance on unseen data.  Based on the model's performance on training and test data we then determine if we need to increase the model's complexity. We will also conduct error analysis after each model we build and review our worst predictions and our loss function for specific features to identify ways to improve our model.   

We will then improve the model by tuning hyperparameters such as the number of neurons for each layer in the CNN and explore techniques such as Max Pooling and Dropout to prevent our model from overfitting and improve the model's ability to generalize to the test data. We will then update the model, refit the model, and test the model again on our test data.  One of the challenges we will encounter during this project is building a CNN with missing data.  We will explore using a model with a masked loss function to allow us to train the model using all of the training dataset. At the end our analysis we then compare the performance across all of our models and also discuss additional feature engineering or data augmentation techniques that could be explored to improve the performance of future models. 

## File System

### Data

The original data is downloaded from Kaggle under the following link:\
https://www.kaggle.com/c/facial-keypoints-detection/data?select=training.zip. \
The name of the file is ***training.zip*** for the Kaggle's competition. Otherwise, it also can be downloaded from our drive in the following link:\
https://drive.google.com/drive/folders/176D4a34yfzZvf7ETIx9ST2oih2l6XZCy?usp=sharing. \
Please be sure that the data (***training.csv***) have the following path form the location of this notebook: \
***/data/training/training.csv***


### Models and Loss 

After running the notebook for the first time, the CNN models are saved in a folder named ***models*** with three subfolders that correspond to each model. The following paths are from the location of this notebook: 
* /models/model_base 
* /models/model_base_extended
* /models/model_Mask

The loss and validation loss calculated during the training phase are also saved in the following files with JSON format: 
* models/model_base_loss.json
* models/model_base_extended_loss.json
* models/model_mask.json

***IMORTANT***: If any model is going to be trained, the corresponding model's folder must be removed from its location. Otherwise, it is going to be loaded from that location.