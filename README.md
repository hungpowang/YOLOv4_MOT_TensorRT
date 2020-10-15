# Introduction
* deployed on NVIDIA Jetson 
* detecting people
* tracking detected people
* counting in/out of people

![](https://github.com/hungpowang/YOLOv4_MOT_TensorRT/blob/main/TRT_demo_480.gif)

### People detection
* using YOLOv4-tiny as object detection model
* model is trained in Darknet framework and converted to TensorRT inference engine with 608 x 608 input size

### Prepare to run
* a video with people
* modify the main.py: replace 'crowd.mp4' by your video
* modify the main.py: **point_1_y** and **point_2_y** to define a line to represent the boundary for in/out counting 
* download the engine file: 'yolov4_tiny_608.engine'

### Run python
* simply type: `python3 main.py`
