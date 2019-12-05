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



