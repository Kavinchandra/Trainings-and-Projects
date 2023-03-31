# Trainings-and-Projects
This repository contains three AI related mini projects.

# 1. LSTM Chatbot:
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

# 2. CNN based LandMark Classification:

Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.

If no location metadata for an image is available, one way to infer the location is to detect and classify a discernible landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgment to classify these landmarks would not be feasible.

In this project, you will take the first steps towards addressing this problem by building models to automatically predict the location of the image based on any landmarks depicted in the image. You will go through the machine learning design process end-to-end: performing data preprocessing, designing and training CNNs, comparing the accuracy of different CNNs, and deploying an app based on the best CNN you trained.

# 3. Face Generation => GAN:
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
