# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the [dataset](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic-signs-data.zip). 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

You're reading it! and here is a link to my [project code](https://github.com/jingr1/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

[//]: # (Image References)

[image1]: ./writeup/Histogram_of_Traffic_Sign_distribution.png "Visualization"
[image2]: ./writeup/image_samples.png "Visualization"
[image3]: ./writeup/grayscaled.png "grayscale"
[image4]: ./writeup/normalized.png "normalized"
[image5]: ./writeup/accuracy_normalized.png "accuracy_normalized"
[image6]: ./writeup/accuracy_dropout0.5.png "accuracy_dropout0.5"
[image7]: ./writeup/new_images.png "new_images"
[image8]: ./writeup/softmax_1.png "Traffic Sign"
[image9]: ./writeup/softmax_2.png "Traffic Sign"
[image10]: ./writeup/softmax_3.png "Traffic Sign"
[image11]: ./writeup/softmax_4.png "Traffic Sign"
[image12]: ./writeup/softmax_5.png "Traffic Sign"
[image13]: ./writeup/softmax_6.png "Traffic Sign"
[image14]: ./writeup/softmax_7.png "Traffic Sign"


---
Histogram of Traffic Sign distribution.png
### Data Set Summary & Exploration

#### 1. Basic summary of the dataset
I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 (width) x 32 (height) x 3 (RGB color channels)
* The number of unique classes/labels in the data set is 43 (see file [signnames.csv](./signnames.csv))

#### 2. Exploratory visualization of the dataset
Here is an exploratory visualization of the training data set. It is a bar chart showing the count of each sign. Here I use pandas pivot table and matplot library to plot the histogram of Traffic Sign distribution.

![alt text][image1]

You also can see below a sample of the images from the dataset.

![alt text][image2]


### Design and Test a Model Architecture

#### Pre-Pocessing the image data
As a first step, I decided to convert the images to grayscale because gray images only has one channel for each pixels, which can reduce the use of the computational load and storage cost.

Here is a sample of traffic sign images after grayscaling.

![alt text][image3]

As a last step, I normalized the image data with Feature Standardization because it refers to (independently) setting each dimension of the data to have zero-mean and unit-variance. it will have better training accuracy and results.

Here is a sample of traffic sign images after normalized.

![alt text][image4]

#### Final model architecture
My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 gray image                            | 
| Convolution 3x3       | 1x1 stride, valid padding, outputs 28x28x6    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 14x14x6                   |
| Convolution 3x3       | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x16                    |
| Dropout               | 0.5 keep probability                          |
| Flatten               | outputs 400                                   |
| Fully connected       | outputs 120                                   |
| RELU                  |                                               |
| Fully connected       | outputs 84                                    |
| RELU                  |                                               |
| Fully connected       | outputs 43                                    |
| Dropout               | 0.5 keep probability                          |
| Softmax               | etc.                                          |
|                       |                                               |

#### How to trained the model
To train the model, I calculate the cross entropy with tf.nn.softmax_cross_entropy_with_logits(), and use tf.reduce_mean(cross_entropy) as the loss function.
Then I use AdamOptimizer as the optimizer to reduce the loss and adjust the weights in the model.
Finally, I run the model to find the final solutions with 128 as the batch size, 0.001 as the learning rate and 100 as the epochs.

In order to improve the model reliability, I use Dropout to prevents the model from overfitting.
Refer to the [paper](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf),
I defined two levels of dropout, one (p-conv) for convolutional layers, the other (p-fc) for fully connected layers and p-conv >= p-fc.

I tried different paratemers but ultimately settled on p-conv=0.5 and p-fc=0.5, which enabled us to achieve a test accuracy of 94.7% on normalised grayscale images.
#### Training Approach

My final model results were:
* training set accuracy of 99.9%
* validation set accuracy of 96.4% 
* test set accuracy of 95.2%

I use the classical model structure LeNet5 with Tensorflow. The LeNet consists of a convolutional layer with 32x32x1 normalised gray image input and 28x28x6 as output, followed by relu activation layer and max-pooling layer. Then we have same three layers following, the output is 5x5x16. Then a Flatten layer was used to compress the 5x5x16 to 400. and then followed by two full connected layer that convert 400 to number of classes 43. 

 I compared grayscaled and normalised images, and saw that normalised gray image tended to outperform the only grayscaled. I got the Training Accuracy = 1.000, Validation Accuracy = 0.948 and Test Accuracy = 0.934. 

 ![alt text][image5]

 That's not bad, but as the graph above shows that the model is not smooth, which actually meant our model was overfitting on the training set and not generalising.

so I added two levels of dropout, one (p-conv = 0.5) for convolutional layers, the other (p-fc = 0.5) for fully connected layers. I got the Training Accuracy = 0.999, Validation Accuracy = 0.964 and Test Accuracy = 0.952. 

 ![alt text][image6]

### Test a Model on New Images

#### Load the new German traffic signs
Here are Seven German traffic signs that I found on the web:

![alt text][image7]

They represent different traffic signs that we currently classify, the "unknow" image is out of the classes, so it certainly can't be classify correctly.

The "Priority Road" image might be difficult to classify because it is incomplete. 

#### Predictions Result

The model predicted 5 out of 7 signs correctly, it's 71.4% accurate on these new images.
Exclusive of the "unknow" sign, it gives an accuracy of 85.7%. This compares favorably to the accuracy on the test set of 95.2%.

Here are the results of the prediction:

| Image                 |     Prediction                                | 
|:---------------------:|:---------------------------------------------:| 
| Keep Right            | Keep Right                                    | 
| Stop                  | Stop                                          |
| Yield                 | Yield                                         |
| Speed Limit 30 km/h   | Speed Limit 30 km/h                           |
| Priority Road         | Traffic Signals                               |
| No Entry              | No Entry                                      |
| unknow                | Turn Left Ahead                               |

#### Softmax probabilities

The output results of top_5 softmax probabilities is listed below:

```
TopKV2(values=array([[  1.00000000e+00,   9.40376828e-25,   3.56742623e-27,
          1.61195630e-28,   1.68107812e-31],
       [  1.00000000e+00,   7.55503151e-21,   1.21045371e-28,
          3.35769100e-31,   2.79243914e-32],
       [  9.99995947e-01,   3.91783442e-06,   8.00379283e-08,
          2.16840110e-08,   9.48551904e-09],
       [  9.67969537e-01,   1.39265228e-02,   1.02096582e-02,
          7.19060469e-03,   3.54055665e-04],
       [  9.99946713e-01,   3.88235858e-05,   1.43164370e-05,
          1.01156807e-07,   5.58384201e-08],
       [  2.87382632e-01,   2.29388788e-01,   1.44465432e-01,
          1.24148376e-01,   5.41160889e-02],
       [  1.00000000e+00,   1.51023500e-16,   2.25728009e-20,
          7.47513083e-22,   1.03131863e-22]], dtype=float32), indices=array([[13, 12,  1, 35, 18],
       [38, 34, 40, 25, 12],
       [ 1,  2,  0,  4,  5],
       [34, 17, 22,  9, 26],
       [17, 10,  9, 34, 23],
       [26, 40, 35,  8, 12],
       [14, 13,  3, 38, 33]], dtype=int32))
```

This is the distribution histogram of softmax probabilities:
![alt text][image8] ![alt text][image9] ![alt text][image10] 
![alt text][image11] ![alt text][image12] ![alt text][image13] 
![alt text][image14]

We can clearly see that our model quite confident in its predictions. Except the "Priority Road" image and the "unknow" image.

