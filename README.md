# ros2_project_final功能包

## 一、节点

包含 **视觉识别节点** 和 **弹丸击打节点**

### 1、视觉识别节点

订阅摄像头仿真获取图像信息；
通过opencv对图像进行处理，获取目标点坐标；
使用深度学习方法判断装甲板数字；
发布要求的话题；

### 2、弹丸击打节点

订阅视觉识别节点发布的话题，对装甲板目标二维坐标进行运算得到三维坐标点，并进行缓存；
构建服务端，接收到打击请求后，根据时间戳找到缓存中对应的三维坐标；
计算设计目标点位的欧拉角，并返回；

## 二、依赖项

### 1、视觉识别节点

### 2、弹丸击打节点

ros2依赖：rclcpp，std_msgs，geometry_msgs，message_filters

opencv依赖：opencv2

自定义功能包依赖：referee_pkg

## 三、算法原理
https://www.yuque.com/yuqueyonghupik2ap/xofqch/tuk75vcfm55q59my?singleDoc# 《敲一辈子代码队技术报告》

## 四、启动方式

ros2 launch ros2_project_final_pkg recognition_shooter.launch.py
