# SexySemantics
## About
The project builds a map from raw LIDAR point cloud and then transfer the predicted semantic labels from the camera image onto the LIDAR point cloud.
## Setup
This project uses [semseg](https://github.com/hszhao/semseg/) to predict semantic labels. semseg needs to be cloned into the same directory as the `Wrapper.py` file. The trained parameters can be downloaded from [here](https://drive.google.com/drive/folders/15wx9vOM0euyizq-M1uINgN0_wjVRf9J3).

## How to run
To run code, specify the `kitti360Path` on line 211 of `Wrapper.py`, and then execute: 
```
PYTHONPATH=./semseg/ python Wrapper.py 
```
