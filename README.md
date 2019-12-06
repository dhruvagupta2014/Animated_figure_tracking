# Animated_figure_tracking

Watch the video here : https://www.youtube.com/watch?v=iQ1YJE9oDRs

This directory contains the following :

#1. Little Man Dataset 

Pictures : Scraped Pink Panther pictures from web and took screenshots from videos on Youtube.
Labels : XML labels containing the bounding box coordinates of the two classes : 'pink panther' , 'the little man'

#2. Custom Faster RCNN config file

#3. Label map

#4. video_prediction.py
Use this sript to generate a new video which tracks the two classes in the label map. Remember to change the paths inside the script. You need to give the path (I have given absolute paths) to the following :

a) source video
b) frozen graph
c) label map

#5. frozen_inference_graph.pb

The trained model can be used to detect pink panther and the little man in pictures.


A few notes and miscellaneous info about the model/predictions :

1. Model tried:
a) MobileNet SSD : low accuracy, faster inference times (~0.1 s/frame) . Discarded this model because of low accuracy. Needed to train this model for atleast 100,000 iterations and I didn't have resources neither the time to train this model further than 20k iterations. 
b) Faster RCNN model with Inception V3 as a base CNN : High Accuracy (more recall v/s precision). Higher prediction times (~1.5 s/frame). This is model we have used to predict on the video. The model was run for ~25000 iterations over google colab

2. The original video was converted into a different video with a different FPS. Since the original video has ~9000 frames, it'd taken a long time to predict over it, so I took 2 frames per second from the original video and made predictions on a total of ~900 frames and interleaved into the video file I've shared with you. I noticed slight re-ordering of the frames while making the prediction video. I didn't get time to debug the part of code which is messing up the order of the frames.

3. To make the video_prediction.py code work, the user's machine should have Tensorflow Object detection API setup and the user should change the paths inside the code. The code will work on any video (formats tried : .AVI, .MP4) 

4. Data used : I collected ~100 images of Pink Panther and the Little Man by taking screenshots and google image scraping and then labelling them by hand. 

TODO : 

1. The accuracy can be improved by training the model for more iterations, increasing the data size

2. Prediction speed can be improved by :
a)using a lighter model like SSD_mobilenet and training it for atleast ~100,000 iterations. 
b) Multithreading while predicting. (loading of video, Encoding-decoding , prediction all in different threads)

3. The model has a higher recall than precision. More training and a better (quality), bigger test dataset  would certainly balance it out. 

4. Data Augmentation without colour changing. 

5. Debug ordering of the frames.



