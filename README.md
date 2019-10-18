# Demo
A patch of color on the ball is selected to teach which color to detect.
Press "Enter" on webcam view to enter into teach mode.

![Color Segmentation](assets/color-balls.gif)

# How it works?
When user select a patch for teaching, the region of interest(ROI) will be converted into HSV color space from RGB color space.
On each H, S, and V layer, min and max values are determined.
In result view, any pixel intensity value fall within the min max range will be shown otherwise it will be hidden per layer
