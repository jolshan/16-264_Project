# 16-264_Project
Code for my 16-264 project--mostly used to research a classification task.

**Project Flow:**  
0\) Download files from VoxCeleb1 dataset (link below)  
1\) Run getImagesAndTags.m  
2\) Choose sample and run makefile.m (or alternative makefileCSV.m)

If makefile.m was run:  
&nbsp;3.1.1) Run pickler.py to extract features and save object to file  
&nbsp;3.1.2) Run either classifytest.py for CNN or treeclassify.py for RandomForestClassifier

If makefileCSV.m was run:  
&nbsp;3.2) Run audioCNN2.py

Many resources were used:  
Vox Celeb Data Set <http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html>  
Crossmodal Biometric Matching <http://www.robots.ox.ac.uk/~vgg/research/cross-modal-emotions/>  
How to apply machine learning and deep learning methods to audio analysis <https://towardsdatascience.com/how-to-apply-machine-learning-and-deep-learning-methods-to-audio-analysis-615e286fcbbc>  
Urban Sound Classification using Convolutional Neural Networks with Keras: Theory and Implementation <https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4>
