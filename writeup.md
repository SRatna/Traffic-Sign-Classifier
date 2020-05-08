# **Traffic Sign Recognition** 

## Writeup

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[hist]: ./images/hist.png "Visualization"
[gray]: ./images/gray.png "Gray scaling"
[norm]: ./images/norm.png "Normalization"
[few_signs]: ./images/few_signs.png "Few traffic signs"
[my_signs]: ./images/my_images.png "My chosen traffic signs"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

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

Here is an exploratory visualization of the data set. It is a bar chart showing how number of training images per labels.

![alt text][hist]

And here are few images of traffic signs which I used for training purpose.

![alt text][few_signs]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to convert the images to grayscale because through my research I found that color images don't provide much value to the accuracy of the model but it will make the training process slow for sure.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][gray]

As a last step, I normalized the image data because normalized data works better and all the pixels value will be between -1 and 1.

Here is an example of an original image and an normalized image:

![alt text][norm]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 10x10x32	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32   				|
| Flattening    		| outputs 800 									|
| Dropout       		| 0.75 keep probability							|
| Fully connected		| outputs 512  									|
| RELU					|												|
| Dropout       		| 0.75 keep probability							|
| Fully connected		| outputs 256  									|
| RELU					|												|
| Dropout       		| 0.75 keep probability							|
| Fully connected		| outputs 43  									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an adam optimizer and cross entropy loss function. I chose following hyperparameters:
* Learning rate: 0.0005
* Number of epoch: 60
* Batch size: 128
* Variables were initialized using the truncated normal distribution with mu = 0.0 and sigma = 0.1
* keep probability of the dropout layer: 0.75

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* testing set accuracy of 0.997 at 60th epoch.
* validation set accuracy of 0.949 at 60th epoch.
* testing set accuracy of 0.94

Of course I went through an iterative process to get the above architecture and hyperparameters:
* Initially I followed LeNet architecture and perform only normalization and using learning rate of 0.001, I trained model for 80 epochs. I reached about 88% of accuracy on validation set.
* After that I used dropout layers and also increase the size of hidden units but I was not able to get more than 90% accuracy.
* Then, I did some reasearch and performed grayscaling which along with reduction in learning rate to 0.0005 brought the model to afore mentioned accuracy.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][my_signs]

The second image might be difficult to classify because the model has to detect a human figure to classify it as a pedestrian as for other signs this model did pretty good job in classifying them.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Pedestrians  			| Pedestrians                       			|
| Bumpy Road			| Bumpy Road									|
| 60 km/h	      		| 60 km/h    					 				|
| Road work 			| Road work          							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94% accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the 2nd last cell of the Ipython notebook.

For the first image, the model is totally sure that this is a road work sign, and the image does contain a road work sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Road work   									| 
| .00	      			| Dangerous curve to the right 	 				|
| .00					| General caution                   			|
| .00				    | Right-of-way at the next intersection 		|
| .00     				| Children crossing								|

For the second image, the model is totally sure that this is a 60km/h speed limit sign, and the image does contain a 60km/h speed limit sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Speed limit (60km/h)  						| 
| .00     				| Keep right                    				|
| .00	      			| Speed limit (30km/h)       	 				|
| .00					| Speed limit (20km/h)    						|
| .00				    | Road work                  					|

For the third image, the model is totally sure that this is a bumpy road sign, and the image does contain a bumpy road sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Bumpy road   									| 
| .00     				| Bicycles crossing								|
| .00					| No vehicles   								|
| .00	      			| Road narrows on the right 	 				|
| .00				    | Road work                 					|


For the fourth image, the model is totally sure that this is a stop sign, and the image does contain a stop sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Stop      									| 
| .00     				| Yield                  						|
| .00					| Speed limit (120km/h)							|
| .00	      			| Keep left                 	 				|
| .00				    | Speed limit (50km/h)        					|

For the fifth image, the model is totally sure that this is a pedestrians sign, and the image does contain a pedestrians sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Pedestrians									| 
| .00					| Right-of-way at the next intersection     	|
| .00     				| Traffic signals           					|
| .00	      			| General caution           	 				|
| .00				    | Road narrows on the right  					|






