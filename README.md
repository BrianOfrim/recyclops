# Recyclops

An end to end application for classifying recycling using tensorflow

A critical part of the recycling process is sorting recycleable into the correct catagories.
Incorrectly sorted or contaminated items cause huge losses to the productivity of recycling 
processing facitities and huge losses in the amount of reycleable items that are actually recycled.
See: https://fivethirtyeight.com/features/the-era-of-easy-recycling-may-be-coming-to-an-end/

This project aims to aid in the classification of recyclable items. There are 4 main parts to an image classification
workflow: data gathering, data cleaning, neural network model training, and model deployment.  

## Gather

![GatherScreenShot](https://raw.githubusercontent.com/BrianOfrim/recyclops/master/doc/assets/gatherSample_480.jpg)

Using the program /gather/gather.py catagories for image classification are overlaid on a live image stream.
Click one of the catagagories to classify the current live stream image and have it sent to s3 storage.

## Clean
![CleanScreenShot](https://raw.githubusercontent.com/BrianOfrim/recyclops/master/doc/assets/cleanSample_480.jpg)

The accuracy of the image classification model is dependant upon the accuracy of the training data so before using
the gathered data for training we need to verify the accuracy of the images. To do this we will use /clean/clean.py
The image catagory options will be overlaid on the image along with the catagory is was assigned at the gathering stage.
Press the corresponding character key to classify the image. Press 's' to skip the current image, press 'w' to return to
a previous image. Note, that going to the previous image with 'w' will remove the "clean" catagory given to the current image 
and it must be re-classified when the image is returned to.

Images to be cleaned are downloaded from s3. After cleaning the list of verified items is uploaded to s3 and any 
incorretly classified items from the gathering stage are uploaded to the correct s3 location. 

## Train

### Setup
To obtain the classified images that will be used for training the /train/setup.py program will be used. This downloads the 
most recent version of the verification list from s3 for each catagory and enures that all verified images are download prior 
to training.

### Hyperparameter Search
In order to increase validation and test accuracy we must find the optimal hyperparameters for trainging the model.
Hyperparameter serach is done with the /train/hParamSearch.py program.
We will do a grid search of hyper parameters for this project, meaning that we will train the model with every combination of
hyperparameters and compare the final validation accuracy for each set of hyper parameters.
For this project the hyperparamers that we are primarly interested in are:
Training Batch Size (eg 4, 8, 16, 32)
Dropout Rate (eg 0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
Optimizer (eg adam, RMSprop)
Base Learning Rate (eg 0.0001, 0.00001)
Fine Tuning (eg True, False)

For each run (set of hyperparameters) we will log the training statistics (trainig accuracy/loss, validation accuracy/loss)
We will then use tensorboard (https://www.tensorflow.org/tensorboard) to view the logged data and determine which set of hyperparameters are best suited for our applicaiton.

For the base model we will use MobileNetV2 pretrained on imagenet. We will exclude the top layer of the network and add our own global pooling and classification layers on top. We will take advantage of the pretrained network's feature extraction vector layer by freezing the headless base model. We will then train the layers we added in order to classify images into our catagories. When the hyperparameter "Fine Tuning" is True, after we have finished the initial training of the added classification layers we will reduce the learning rate by 10x, unfreeze part of the initial pretrained network and preform fine tuning trainig on the unfrozen portion of the network in addition to the added layers.


## Deploy
![DeployScreenShot](https://raw.githubusercontent.com/BrianOfrim/recyclops/master/doc/assets/deploySample_480.jpg)
