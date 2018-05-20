# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image-data-viz]: ./images/dis-graph.png "Visualization"
[image4]: ./images/groad-1.png "Traffic Sign 1"
[image5]: ./images/groad-3.png "Traffic Sign 2"
[image6]: ./images/groad-17.png "Traffic Sign 3"
[image7]: ./images/groad-33.png "Traffic Sign 4"
[image8]: ./images/groad-38.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/waleoyediran/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training dataset is distributed acrosses the different classes of traffic signs

![Visualization][image-data-viz]

### Design and Test a Model Architecture

#### 1. Data Preproceessing techniques

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I chose to generate additional augmented data because tests showed that augmenting the dataset achieved better results than tuning the hyperparameters of the network

To add more data to the the data set, I used basically generated additional 2 rotated versions of the original images, by rotating the images st +5 and -% degrees.



#### 2. Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| Max pooling			| Reduces the size to 14x14x6					|
| RELU					|												|
| Convolution 5x5     	| Input 14x14x6, outputs 10x10x6 				|
| Fully Connected + RELU| None-Linear - output - 267					|
| Fully Connected + RELU| None-Linear - output - 178					|
| Fully Connected + RELU| None-Linear - output - 43		  				|
|						|												|
|						|												|
 


#### 3. Training Approach

To train the model, I used an the LetNet architecture
I used the configuration
EPOCHS = 20
BATCH_SIZE = 256

#### Training Discussion

My final model results:
* Test set accuracy of .91

 

### Test a Model on New Images From the web

#### 1. Choice of new Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Predictions

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h Zone   	  	| End of all speed and passing limits   		| 
| 60 km/h ZOne 			| U-turn 										|
| No Entry				| Yield											|
| Turn right ahead  	| Bumpy Road					 				|
| Keep Right			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 92% from the training

#### 3. Softmax Probabilities 
The model is provides a very high degree of probabilities when predicting new images.

1:
 32: 97.97%
 1: 1.99%
 38: 0.04%
 6: 0.00%
 36: 0.00%
3:
 3: 100.00%
 2: 0.00%
 10: 0.00%
 5: 0.00%
 1: 0.00%
17:
 17: 100.00%
 14: 0.00%
 22: 0.00%
 1: 0.00%
 26: 0.00%
33:
 33: 100.00%
 28: 0.00%
 39: 0.00%
 12: 0.00%
 24: 0.00%
38:
 38: 100.00%
 34: 0.00%
 13: 0.00%
 36: 0.00%
 0: 0.00%



