# Recyclops
An end to end application for classifying recycling using tensorflow

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

## Train

## Deploy
![DeployScreenShot](https://raw.githubusercontent.com/BrianOfrim/recyclops/master/doc/assets/deploySample_480.jpg)
