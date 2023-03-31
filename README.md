# Trainings-and-Projects
This repository contains two AI related mini projects.

# LSTM Chatbot:
## Project Summary
In this project, you will learn how to build an AI chatbot using LSTMs, Seq2Seq, and pre-trained word embeddings for increased accuracy. You will be provided with a dataset of conversational dialogue. You will use this dataset to build your chatbot using Pytorch, train it on the dataset, and tune your network hyperparameters.

At the end of the project, you will demonstrate your proficiency in deep learning by conversing with their chatbot at the command line.

### Project Steps Overview and Estimated Duration

Below you will find each of the components of the project, and estimated times to complete each portion. These are estimates and not exact timings to help you expect the amount of time necessary to put aside to work on your project.

### Prepare data (~2 hours)

Build your vocabulary from a corpus of language data. The Vocabulary object is described in Lesson Six: Seq2Seq.

### Build Model (~4 hours)

Build your Encoder, Decoder, and larger Sequence to Sequence pattern in PyTorch. This pattern is described in Lesson Six: Seq2Seq.

### Train Model (~3 hours)

Write your training procedure and divide your dataset into train/test/validation splits. Then, train your network and plot your evaluation metrics. Save your model after it reaches a satisfactory level of accuracy.

### Evaluate & Interact w/ Model (~1 hour)

Write a script to interact with your network at the command line.

# Face Generation => GAN:
This is the second project and here you'll build a custom generative adversarial network to generate new images of faces.

Open the notebook file, dlnd_face_generation_starter.ipynb and follow the instructions. This project is organized as follows:

### Data Pipeline: 
implement a data augmentation function and a custom dataset class to load the images and transform them.
### Model Implementation: 
build a custom generator and a custom discriminator to make your GAN
### Loss Functions and Gradient Penalty: 
decide on loss functions and whether you want to use gradient penalty or not.
### Training Loop: 
implement the training loop and decide on which strategy to use

Each section requires you to make design decisions based on the experience you have gathered in this course. Do not hesitate to come back to a section to improve your model or your data pipeline based on the results that you are getting.

Building a deep learning model is an iterative process, and it's especially true for GANs!
