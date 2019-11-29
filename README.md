# Recyclops

An end to end application for classifying recycling using tensorflow 2

A critical part of the recycling process is sorting recyclable items into the
correct categories. Incorrectly sorted or contaminated items cause huge losses
to the productivity of recycling processing facitities and huge losses in the 
amount of recyclable items that are actually recycled.
See: https://fivethirtyeight.com/features/the-era-of-easy-recycling-may-be-coming-to-an-end/

This project aims to aid in the classification of recyclable items. There are 
4 main parts to this image classification workflow: data gathering, data cleaning,
neural network model training, and model deployment.

The programs are intentionally fairly agnostic to the categories so that the same methodologies
and workflow can be applied to other image classification applications.

## Gather

![GatherScreenShot](https://raw.githubusercontent.com/BrianOfrim/recyclops/master/doc/assets/gatherSample_480.jpg)

The program /gather/gather.py will overlay the categories for image classification
on the live camera image stream. Enter the key for one of the category options to 
classify the current live stream image and have it sent to s3 storage.

## Clean
![CleanScreenShot](https://raw.githubusercontent.com/BrianOfrim/recyclops/master/doc/assets/cleanSample_480.jpg)

The accuracy of the image classification model is dependent upon the accuracy
of the training data so before using the gathered data for training we need to 
verify the accuracy of the images. To do this we will use /clean/clean.py
The image category options will be overlaid on the image along with the category
it was assigned at the gathering stage. Press the corresponding character key to
classify the image. Press 's' to skip the current image, press 'w' to return to
a previous image. Note, that going to the previous image with 'w' will remove 
the "clean" category given to the current image and it must be re-classified 
when the image is returned to.

Images to be cleaned are downloaded from s3. After cleaning, the list of verified
items is uploaded to s3 and any incorrectly classified items from the gathering
stage are uploaded to the correct s3 location. 

## Train

### Setup
To obtain the classified images that will be used for training the 
/train/setup.py program will be used. This downloads the most recent version of
the verification list from s3 for each category and ensures that all verified 
images are downloaded prior to training. The /train/setup.py will also download
the newest trained model that was uploaded to s3 by a previous run of the
/train/train.py program which will be described later.

### Hyperparameter Search
In order to increase validation and test accuracy we must find the optimal
hyperparameters for training the model. Hyperparameter serach is done with 
the /train/hParamSearch.py program. We will do a grid search of hyperparameters
for this project, meaning that we will train the model with every combination of
hyperparameters and compare the final validation accuracy for each set of 
hyperparameters.
For this project the hyperparameters that we are primarly interested in are:
 * Training Batch Size (e.g. 4, 8, 16, 32)
 * Dropout Rate (e.g. 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
 * Optimizer (e.g. adam, RMSprop)
 * Base Learning Rate (e.g. 0.0001, 0.00001)
 * Fine Tuning (e.g. True, False)

For each run (i.e. set of hyperparameters) we will log the training statistics
(training accuracy/loss, validation accuracy/loss).
We will then use tensorboard (https://www.tensorflow.org/tensorboard) to view 
the logged data and determine which set of hyperparameters are best suited for
our application.

For the base model we will use MobileNetV2 pretrained on imagenet. We will 
exclude the top layer of the network and add our own global pooling and 
classification layers on top. We will take advantage of the pretrained network's
feature extraction vector layer by freezing the headless base model. We will then
train the layers we added in order to classify images into our categories. When 
the hyperparameter "Fine Tuning" is True, we will reduce the learning rate 
by 10x, unfreeze part of the initial pretrained network and perform fine tuning 
training on the unfrozen portion of the network in addition to the added layers.

### Training
Once we have found the most optimal hyperparameters from our grid search we
will use /train/train.py to train, save and upload the model that we will
deploy. We will input the hyperparameters we wish to use to train the model
via command line flag arguments. We can also customize the training run with other
command line flag arguments such as the number of epochs, validation split, etc.
To see the full list of arguments, use:
```
$ python3 train.py --help
```
The --help flag can be given to any of the scripts in this repo to get a list of
all command line flag options.
Once the training script is completed the trained model will be zipped and uploaded
to s3 for us to download and deploy as part of the deployment stage.

The /train/train.py can also take advantage of GPU acceleration to increase the 
speed of training. Running /train/train.py on an AWS p3.2xlarge EC2 instance 
(has NVIDIA Tesla V100 GPU) with the GPU enabled version of tensorflow 2 
(https://www.tensorflow.org/install/gpu) it was observed that the training
speedup was around 6x for batch size = 32 when compared to CPU based training 
on a workstation computer. The GPU speedup relative to CPU increases as batch size 
increases as there will be less frequent CPU to GPU memory transfer.

When training is complete the saved model folder will be zipped and sent to s3.
As mentioned earlier the saved model can be obtained on any computer with access
to the recyclops s3 bucket using the /train/setup.py program.

Similar to /train/hParamSearch.py, the training accuracy/loss and validation
accuracy/loss will be logged in a format that makes it viewable with tensorboard.

## Deploy
![DeployScreenShot](https://raw.githubusercontent.com/BrianOfrim/recyclops/master/doc/assets/deploySample_480.jpg)

The /deploy/classifier.py script can be run once the latest trained model has been
obtained with /train/setup.py
This program will classify items into different categories.
There are command line options for this program that can be observed with:
```
$ python classifier.py --help
```
