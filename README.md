# Image_Classification_with_Pre-trained_Models

## **Pretrained Image Classifier**

This repository contains code for building an image classifier using pre-trained models in Keras. The code demonstrates how to use the ResNet50 and VGG16 models to classify images. Pretrained models are pre-trained on large datasets and have optimized parameters, making them a powerful tool for image classification tasks.

## **Table of Contents**
•	Introduction

•	Data Preparation

•	Pre-training and Comparing Models

•	Results

## **Introduction**

This repository provides code and instructions to build an image classifier using pre-trained models in Keras. Pretrained models are beneficial when computational resources are limited or to achieve high performance with few epochs. The repository covers the following concepts:
1.	Loading and preparing the dataset
2.	Using the Keras ImageDataGenerator class for data augmentation and batch training
3.	Training and evaluating an image classifier using the ResNet50 pre-trained model
4.	Building an image classifier using the VGG16 pre-trained model
5.	Comparing the performance of both models on a test set of images
   


## **Data Preparation**

In this section, we explore how to prepare the dataset for image classification. The dataset contains images of concrete surfaces, some with cracks (positive samples) and some without cracks (negative samples). We demonstrate how to load and visualize the images, avoiding memory inefficiencies when working with large datasets. To handle the resource limitations effectively, we use the Keras ImageDataGenerator class to augment the training data and train the deep learning model in batches.

## **Pre-training and Comparing Models**

In this section, we delve into using pre-trained models for image classification. Specifically, we focus on the ResNet50 and VGG16 pre-trained models. We show how to build a classifier by excluding the output layer from the pre-trained model and adding a new output layer for our image dataset. By training the new output layer using the pre-trained model's hidden layers, we can achieve impressive performance with minimal epochs.

The code includes step-by-step instructions for creating training and validation dataset objects, reshaping images to fit the model, creating a softmax classifier, setting the random seed, training the model, and evaluating its performance.


## **Results**

The code in the notebook trains and evaluates the image classifiers using the ResNet50 and VGG16 pre-trained models. It provides performance metrics such as loss and accuracy for each model and generates predictions for a test set of images. The results can be used to compare the performance of both models.

