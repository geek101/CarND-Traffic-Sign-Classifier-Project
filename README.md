# **Traffic Sign Recognition** 

---


[//]: # (Image References)

[image1]: ./examples/img_o.png "Visualization"
[image2]: ./examples/img_y.png "Y Channel with CLAHE"
[image3]: ./examples/img_a.png "Affine Transformation random rotation,sheer and scale"
[image4]: ./examples/img_g.png "Gaussian blur"
[image5]: ./examples/img_m.png "Motion blur"
[image6]: ./examples/img_p.png "Projetive transformation random"
[image7]: ./examples/41_end_of_no_passing_crop.jpg "End of no passing Class 41"
[image8]: ./examples/23_slippery_road_crop.jpg "Slippery road Class 23"
[image9]: ./examples/17_no_entry_crop.jpg "No Entry Class 17"
[image10]: ./examples/28_children_crossing.jpeg "Children Crossing Class 28"
[image11]: ./examples/24_road_narrows_on_the_right_crop.jpg "Road Narrows On The Right Class 24"
[image12]: ./examples/12_priority_road.jpg "Priority Road Class 12"
[image13]: ./examples/1_speed_limit_30.jpeg "Speed Limit 30 Class 1"
[image14]: ./examples/13_yield.jpg "Yield Class 13"
[image15]: ./examples/38_keep_right.jpg "Keep Right Class 38"
[image16]: ./examples/train_set.png "Training Set Bar Plot"
[image17]: ./examples/valid_set.png "Validation Set Bar Plot"
[image18]: ./examples/test_set.png "Test Set Bar Plot"
[image19]: ./examples/smax-41.png "End of no passing Class 41"
[image20]: ./examples/smax-23.png "Slippery road Class 23"
[image21]: ./examples/smax-17.png "No Entry Class 17"
[image22]: ./examples/smax-28.png "Children Crossing Class 28"
[image23]: ./examples/smax-24.png "Road Narrows On The Right Class 24"
[image24]: ./examples/smax-12.png "Priority Road Class 12"
[image25]: ./examples/smax-1.png "Speed Limit 30 Class 1"
[image26]: ./examples/smax-13.png "Yield Class 13"
[image26]: ./examples/smax-38.png "Keep Right Class 38"
[image27]: ./examples/activ_img_test.png "Activation image"
[image28]: ./examples/conv1_layer_visual.png "Conv first layer activation output"
[image29]: ./examples/conv1_layer_weights.png "Conv first layer weights after training"

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Summary of given data and after agumentation.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples given = 34799, after aug = 464868
* Number of validation examples given = 4410, after aug = 116217
* Number of testing examples given = 12630, after aug = 12630
* Image data shape given = (32, 32, 3), after processing = (32, 32, 1)
* Number of classes = 43

I combined both the training and validation data and then balanced and agumented it as described below and split this data 80/20 for train and validation sets.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed in the original given set and after
agumentation.

![alt text][image16]       ![alt text][image17]     ![alt text][image18]

### Design and Test a Model Architecture

#### 1. Processing of given Image set

As a first step, I decided to convert the images to grayscale because because both the papers ([Sermanet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) and [IDSIA](http://people.idsia.ch/~juergen/ijcnn2011.pdf)) seem to indicate that grayscale image out performed color images given the model size and the size of the data set.

Here is an example of a traffic sign image before and after grayscaling with contrast limited adaptive histogram equalization (CLAHE) applied as suggested by .

![alt text][image1] ![alt text][image2] 

I generated a balanced data set with repliation of each class set to 1.2 of the maximum sized class by applying various techniques like 
Affine transformation with random rotation (+/- 0.15), shear (+/1 0.15) and scale (0.9 to 1.1), Gaussuan blur (sigma = 1.0) and motion blur with random kernel size between (4, 6).

This set is again taken and to each image projective transformation as specified by this [blog](https://navoshta.com/traffic-signs-classification/) 

Here are the examples of augmented images of the above image(order affine transformation, gaussian blur, motion blur and projective transformation):

![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6] 

The difference between the original data set and the augmented data set is also mentioned above in the previous section.

#### 2. Final model architecture.

The best performing final model closely follows the model mentiond in [IDSIA](http://people.idsia.ch/~juergen/ijcnn2011.pdf) with couple of changes. The three convolutional layers use 5x5 kernels and two fully connected layers with first fully connected layer implementing dropout with probability of 0.5. Both the fully connected layers also undergo l2 regularization with beta set to 0.0001. Mini batch size is 128 and epoch length tried is from 100 to 210.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Y-Channel with CLAHE image   			| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x100 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, same padding,  outputs 16x16x100 	|
| Convolution 5x5	    | 1x1 stride, same padding,  outputs 16x16x150  |
| RELU					|												|
| Max pooling	      	| 2x2 stride, same padding,  outputs 8x8x250 	|
| Fully connected 1	    | 1024 , with dropout 0.5 and l2 reg of .0001   |
| RELU                  |                                               |
| Fully connected 2	    | 100 , with l2 reg of .0001                    |
| RELU                  |                                               |
| Output layer			| 43 classes        							|
| Softmax cross entropy |                                               |

 
#### 3. Model training

To train the model, I used an Adam optimizer with the weights and biases initialized using Xavier Initialization. Number of Epochs are 210 with best performing model saved and restored later. 
Learning rate is set to .0001 and L2 loss beta set to .0001 and the mini batch size is set to 128.

#### 4. Approach

Initally I have taken LeNet for MNIST and scaled it by 4 times at all layers
and trained on given training set converted to grayscale with CLAHE applied. I was able to quickly obtain above 0.93 test set accuracy.

* What was the first architecture that was tried and why was it chosen?

   Used LeNet for MNIST since it is a CNN and easy to train.

* What were some problems with the initial architecture?

   It was underfitting the data and had high error for validation and 
   test accuracy.

* How was the architecture adjusted and why was it adjusted?

   Initially the simple LeNet was scaled 4 times at all layers. I was able to achieve 93% accuracy on test set with this. 

   Then to achieve higher accuracy I have implemented both 
   [sermanet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) and
   [idsia](http://people.idsia.ch/~juergen/ijcnn2011.pdf)

   This submission is for IDSIA with 98.73% accuracy.

* Which parameters were tuned? How were they adjusted and why?

   I tried various kernel sizees from [3, 3] to [5,5]. Initially 
   had no L2 regularization then tried .001 and .0001. No dropout and
   dropout all all layers and finally settled with dropout only for
   the first fully connected layer.

* What are some of the important design choices and why were they chosen?

   How to agument data and what transformations to apply to increase the training set size.


I read and implement [sermanet](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) which does mention agumenting the dataset with more examples. I have used Affine transformations along with Gaussian blur and Motion blur. I have tried the model with various both 108-108 and 108-200 filters, with one and two fully connected layers of various sizes. This model seems to achieve close to .98% test accuracy and I was unable to get it past 0.9881 (human level performance).

I then read and implemented [idsia](http://people.idsia.ch/~juergen/ijcnn2011.pdf) and this model with fewer filters and seems to give better test accuracy with fewer epochs of training.

The final model is the idsia implementation with noted differences mentioned in the previous section.

Both the models tried are based on CNNs and this is image classification I thought it was best to try few parameters within the contraints of this model.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 99.99%
* test set accuracy of 98.73%

### Test a Model on New Images

#### 1. German traffic signs from web analysis

Here are eight German traffic signs that I found on the web:

![alt text][image7]
![alt text][image8]
![alt text][image9] 
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]

The model for first image ("End of No Passing") has recall rate of 87%  and this is the top 3rd least performing class. The model fails have this classification in the top 5 possibilites and attempts to classify the speed limit sign in the image.

####2. Model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			             |     Prediction	        				   |
|:--------------------------:|:-------------------------------------------:|
| End of no passing          | Speed limit 120km/h   					   |
| Slippery road     	     | Slippery road							   |
| No Entry				     | No Entry									   |
| Children Crossing	         | Children Crossing					 	   |
| Road narrows on the right  | Slippery Road      					       |
| Priority road              | Priority road                               |
| Speed limit (30km/h)       | Speed limit (30km/h)                        |
| Yield                      | Yield                                       |
| Keep Right                 | Keep Right                                  |


The model was able to correctly guess 8 out of the 9 traffic signs, which gives an accuracy of 88.89%. This compares favorably to the accuracy on the test set of 98.73%

#### 3. Softmax probabilities for each prediction. 

The code for making predictions on my final model is located in the 58th cell of the notebook and 61st cell contains the top 5 softmax probabilities and predicted
classes for the web images. You can find the same below.

This image is deliberately cropped two contain two signs. One of the signs is similar to speed limit signs but none of the classes match it.
The expectation here is to see if the model can correctly classify this image as "End of no passing". However the model does not have the right prediction in top 5 predictions.
![alt text][image19]

This image also contains additional artifacts other than the sign like snow symbols. However here the model correctly predicts it as "Slippery road" with not with high confidence. 
![alt text][image20]


When the model was higher test error it seems to be confused by the artifacts in the image introduced by the CLAHE transformation. But a model with higher accuracy does classify it correctly.

![alt text][image21]

This image has lower brightness overall and seems to be captured at night time but it is still classified correctly.

![alt text][image22]

road narrows on the right sign was previously wrongly classified as other signs like traffic light etc but seems to have performed well here.
![alt text][image23]

This image was selected for being slightly sheared and also the CLAHE transformation introduced many artifacts in the background however the image is still classified accurately.

![alt text][image24]

This image was again captured at night time but has a lot of white pixes around the sign it did classify accurately regardless.

![alt text][image25]

This is a straight forward sign at day time but the CLAHE transformation did add many artifacts around the sign.

![alt text][image26]


### Visualizing the Neural Network 

#### 1. Discuss the visual output of your trained network's feature maps.

The following Beware of ice/snow (class 30) image is used to output the activation maps from the first convolution layer.

![alt text][image27]

The 100 convolution layer 1 activation maps for the above image is shown below.

![alt text][image28]

It seems like edges of the two triangles at various intensities awith different background along with distored variations of activated pixes of the snow flake image in the center play role in the various outputs from the first layer.

#### 2. Weights of the first covolution layer.

The following shows the weights of the first covolution layer of the fully training network. Clearly many edges with various intensities are acting as filters.

![alt text][image29]

