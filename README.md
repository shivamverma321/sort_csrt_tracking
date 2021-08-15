# sort_csrt_tracking

## What is SORT
* The simple online and realtime tracking algorithm is meant for multi object tracking in a video.
* A CNN model first detects where objects are in a frame of a video
* The detection is made given no previous information
* Once an obstacle is detected however, it can be tracked
  * We can also set a parameter (MIN_HITS)  to decide how many consecutive detections of an object are necessary for it to be tracked
* Using the detections we can learn how to track the object even if the CNN model is unable to detect where the object is

## What is KF
* The original SORT algorithm uses a Kalman Filter where they build an internal velocity model of the obstacle using the detections and make predictions on where the object will be in the future based off of this model

## Why KF is not ideal
* The KF does not do a good job especially when the camera is shaky because an object could be still but if the camera is moving then the KF will unintentionally learn that the object has nonzero velocity

## Improvement with CSRT
* In the SORT CSRT algorithm the CSRT tracking algorithm will be used instead. The CSRT tracker learns a convolutional filter, essentially what the pixel distribution of the obstacle is like. Thus, when detections are unable to be made, the csrt tracker can make a prediction by finding where in the frame the learned pixel distribution matches in the future frame. This way even if the camera is moving we can make good predictions.
* The tracking will be done for MAX_AGE frames before dying off automatically unless if a new detection is fed before the tracker dies

## Results
Before using Kalman Filter
![caption](results/kf_sort.gif)
After using CSRT
![caption](results/csrt_sort.gif)
