# Yolo v3 Object Detection in Tensorflow
Yolo v3 is an algorithm that uses deep convolutional neural networks to detect objects. <br> <br>

## Getting started

### Prerequisites
This project is written in Python 3.6.6 using Tensorflow (deep learning), NumPy (numerical computing), Pillow (image processing), OpenCV (computer vision) and seaborn (visualization) packages. I will be using anaconda to install all the library.

```
conda env create -f enviroment.yaml
```

### Downloading official pretrained weights
Let's download official weights pretrained on COCO dataset. 

```
!wget -P weights https://pjreddie.com/media/files/yolov3.weights
```

### Save the weights in Tensorflow format
Save the weights using `load_weights.py` script.

```
python load_weights.py
```

## Running the model
Now you can run the model using `detect.py` script. Don't forget to set the IoU (Intersection over Union) and confidence thresholds.
### Usage
```
python detect.py <images/video> <iou threshold> <confidence threshold> <filenames>
```
### Images example
Let's run an example using sample images.
```
python detect.py images 0.5 0.5 data/images/bali_street.jpg
```
Then you can find the detections in the `detections` folder.
<br>

```
python detect.py video 0.5 0.5 data/video/Street.mp4
```
The detections will be saved as `detections.mp4` file.

## To-Do List
* Model training in GPU

## Acknowledgments
* [Yolo v3 official paper](https://arxiv.org/abs/1804.02767)
* [What’s new in YOLO v3?](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)
* [A Tensorflow Slim implementation](https://github.com/mystic123/tensorflow-yolo-v3)
* [ResNet official implementation](https://github.com/tensorflow/models/tree/master/official/resnet)
* [DeviceHive video analysis repo](https://github.com/devicehive/devicehive-video-analysis)
* [Evening Walk ~ Thamrin Nine to Sudirman Plaza thru Setiabudi Astra MRT Station ~ Jakarta](https://www.youtube.com/watch?v=9P1r57NrPK8&t=15s)
