# Writeup Template--Behavioral Cloning P3

# *Rubric Points*
## All required files necessary run the simulator in autonomous mode are included.
The included files are:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 is a video illustrating a successful lap of the course in autonomous mode
* writeup_report.md summarizing the results

## *Model Architecture and Training Strategy*
### 1. Model architecture
Two models were constructed and evaluated.  The first model was based on the LeNet architecture and the second model was based on the NVIDA archetecture described in this NVIDA [blog post](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars).  While it was demonstrated that it is possible to successfully train the LeNet model to safely steer the car around the track, this write up will focus on the application of the NVIDA architecture.

The NVIDA architecture consited of sequence of four 2D convolutions with RELU activations followed by a sequence of four dense layers with linear activations.  The network parameters (convolution kernals, strides and dense layer sizes) are illustrated in Figure 1 which was obtained from the NVIDA post.

![Figure 1](./Figures/NVIDA_Network.png?raw=true)Figure 1: Illustration of the NVIDA network used in this study.

Each image fed into the network was preprocessed as follows: the upper and lower edges of the image is cropped then the image is normalized.  The upper 60 pixels of the image were cropped to minimize the horizon background clutter while simulatneously providing a full view of the road as it vanished into the image horizon.  The low 20 pixels of the image were cropped out because the lower section of the image is dominate by a static view of the car hood.  Because of this, it was assumed that the lower 20 pixels would not contribute to the training justifying their removal. Figure 2 illustrates the pre- and post-cropping of the camera image.
![Figure 2](./Figuress/Cropping_Example.png)Figure 2: Example of the pre and post cropped camera image.

The cropped image was then normalized as follows:
      normalized image = (cropped image)/255.0 - 0.5

Other forms of image manipulation such as histogram equalization or converstion to black and white could prove useful but were not explored.

### 2. Overfitting reduction
An early stopping callback was used to mitigate the possibility of overfitting the data. If the change in the validation loss was less than 0.001 between successive epochs, then the early stopping callback which terminated the fitting (training) of the model.  Figure 3 shows the training and validation loss as a function of epoch for the final trained case.
![Figure 3](./Figures/Loss_History.png)Figure 3: Loss as a function of epoch

Dropouts (with k=0.5) were placed between each of the convolution layers at the output of the last convolution layer in an effort to prevent overtraining.  However for each test where dropouts were used the car steering was worse.  Hence, no dropout layers were used in the final model.

### 3. Model parameter tuning
The model used an Adam optimizer.  In addition, the default Adam optimizer parmaters (lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) were used in this study.

### 4. Training data
Eight datasets of training simulations were generated.  Each dataset was stored in a serperate directory, which allowed for varying the quantity and quality of the training data to evaluate which combination of data yields a trained model that can successfully navigate the course in autonomous mode.  The eight datasets are:
1. One full lap around course one with the car traveling along the center of the road,
1. Two full laps around course one with the car traveling along the center of the road followed by two full laps with the car traveling in the opposit direction.
1. A partial lap around course two.
1. Two full laps around course one with the car traveling along the center of the road followed by two full laps with the car traveling in the opposit direction.
1. Two full laps around course one with the car zigzagging from left to right followed by two full laps with the car traveling in the opposit direction and zigzaggin.
1. Two full laps around course one with the car traveling along the center of the road followed by two full laps with the car traveling in the opposit direction.
1. Two passes by the dirt areas of course with the car traveling in the forward and opposite directions.
1. One full laps around course one with the car traveling along the center of the road followed by one full laps with the car traveling in the opposit direction.

Three camera images (center, left and right) were recorded as the car traveled the course.  In addition to the camera images, the steering angle was simultaneously recorded.  Note: the steering angles for the left and right cameras were corrected due to the fact that the cameras were angled away from the direction of travel of the car.  An analysis of the left, right and center camera images suggeted that the left and right cameras were angled by +0.1 and -0.1 radians from the center, respectivley.

The training data was further augmented by mirroring each camera around it's vertical (y) axis.  The steering angle for each mirrored image was taken to be the negitive of the steering angle of the non-flipped image.

This augmented data (three cameras plus mirroring) was used as the training input.

## Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
