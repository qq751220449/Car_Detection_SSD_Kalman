# 车辆跟踪-基于SSD+Kalman


## 简介
基于google官方的models中的SSD_MobileNet网络实现车辆的检测，使用Kalman实现对目标的跟踪。

https://github.com/tensorflow/models

## 环境


  * Tensorflow-gpu = 1.10
  * Keras = 2.2.4
  * scikit-learn = 0.19.1
  * opencv-python = 3.4.2.16
  * GTX1080

## 跟踪效果
<img src="video/source.gif" alt="Drawing" style="width: 800px;"/>

```
图中蓝色的框为SSD目标检测网络检测的车辆位置，红色为进行卡尔曼跟踪后的修正的BBox.在进行跟踪目标与检测目标进行匹配时，使用的匈牙利算法(Munkres algorithm).对存在遮挡的情况处理的还不是很好，可以尝试其他的匹配算法-比如相似度检测等，具体可以参考TLD算法。
```

SSD_MobileNet识别速度很快，但是准确率不高，出现很多次有车无检测的情况，使用跟踪算法，可以减少这种情况的发生。只有在长时间检测不到跟踪对象才会去掉跟踪目标。