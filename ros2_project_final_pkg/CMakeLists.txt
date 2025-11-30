cmake_minimum_required(VERSION 3.20)
project(ros2_project_final_pkg)

#设置C++标准
set(CMAKE_CXX_STANDARD 17)
#不支持就报错
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#编译器选项配置
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
        -Wall -Wextra -Wpedantic  # 开启基础警告
        -Wno-deprecated-declarations  # 忽略废弃声明警告
    )
endif()

# 设置LibTorch路径
set(LIBTORCH_PATH "/opt/libtorch")

find_package(Torch REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(referee_pkg REQUIRED)
find_package(message_filters REQUIRED)

#视觉识别节点
find_package(ament_index_cpp REQUIRED)

add_executable(RecognitionNode src/recognition.cpp src/recognition.hpp src/RecognitionNode.cpp)
ament_target_dependencies(RecognitionNode
  rclcpp
  std_msgs
  OpenCV 
  geometry_msgs
  cv_bridge
  referee_pkg
  sensor_msgs
  gazebo_msgs
  ament_index_cpp 
)
target_link_libraries(RecognitionNode ${OpenCV_LIBS})
target_link_libraries(RecognitionNode ${TORCH_LIBRARIES})

install(TARGETS RecognitionNode
  DESTINATION lib/${PROJECT_NAME}
)

#安装模型文件
install(
  DIRECTORY models/
  DESTINATION share/${PROJECT_NAME}/models
)


#弹丸击打节点
add_executable(ShooterNode src/shooter_node.cpp)
ament_target_dependencies(ShooterNode
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  gazebo_msgs
  OpenCV
  cv_bridge
  referee_pkg
  message_filters
  ament_index_cpp
)

#
#手动链接OpenCV库
target_link_libraries(ShooterNode ${OpenCV_LIBS})


# 安装全部节点
install(TARGETS ShooterNode
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
