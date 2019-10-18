# Demo
A patch of color on the ball is selected to teach which color to detect.
Press "Enter" on webcam view to entering into teach mode.


### [Video](https://www.youtube.com/watch?v=qLh5igTTIkM) from youtube
![Color Segmentation](assets/color-balls.gif)


# How it works?
When a user selects a patch the region of interest(ROI) will be converted into HSV color space from RGB color space. On each H, S, and V channel, min and max values are determined. In the result view, any pixel intensity value falls outside from the min-max range will be hidden.


# Requirements
Python 3.6, Opencv 3.4.7 and other common packages listed in requirements.txt.