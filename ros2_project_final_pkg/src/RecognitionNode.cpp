#include "recognition.hpp"

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/point.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/timer.hpp>
#include <referee_pkg/msg/multi_object.hpp>
#include <referee_pkg/msg/object.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <std_msgs/msg/header.hpp>
#include "sensor_msgs/msg/image.hpp"
using namespace rclcpp;

class PictureNode : public rclcpp::Node {
 public:
  PictureNode(string name) : Node(name) {
    RCLCPP_INFO(this->get_logger(), "启动！");

    Image_sub = this->create_subscription<sensor_msgs::msg::Image>(
        "/camera/image_raw", 10,
        bind(&PictureNode::callback_camera, this, std::placeholders::_1));
    Target_pub = this->create_publisher<referee_pkg::msg::MultiObject>(
        "/vision/target", 10);

    cv::namedWindow("Detection Result", cv::WINDOW_AUTOSIZE);

    RCLCPP_INFO(this->get_logger(), "成功启动");
  }

  ~PictureNode() { cv::destroyWindow("Detection Result"); }

 private:
  void callback_camera(sensor_msgs::msg::Image::SharedPtr msg);
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr Image_sub;
  rclcpp::Publisher<referee_pkg::msg::MultiObject>::SharedPtr Target_pub;
};


Recognition start;

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PictureNode>("Picturenode");
  RCLCPP_INFO(node->get_logger(), "Starting PictureNode");
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}



void PictureNode::callback_camera(sensor_msgs::msg::Image::SharedPtr msg) {
  try {
    // 图像转换
    cv_bridge::CvImagePtr cv_ptr;

    if (msg->encoding == "rgb8" || msg->encoding == "R8G8B8") {
      cv::Mat image(msg->height, msg->width, CV_8UC3,
                    const_cast<unsigned char *>(msg->data.data()));
      cv::Mat bgr_image;
      cv::cvtColor(image, bgr_image, cv::COLOR_RGB2BGR);
      cv_ptr = std::make_shared<cv_bridge::CvImage>();
      cv_ptr->header = msg->header;
      cv_ptr->encoding = "bgr8";
      cv_ptr->image = bgr_image;
    } else {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }

    cv::Mat image = cv_ptr->image;

    if (image.empty()) {
      RCLCPP_WARN(this->get_logger(), "Received empty image");
      return;
    }
    
    
    cv::Mat result_image = image.clone();
    cv::imshow("Detection Result", result_image);
    start.Integration(result_image);
    cv::waitKey(1);

    referee_pkg::msg::MultiObject msg_object;
    msg_object.header = msg->header;
    msg_object.num_objects = start.allObjects.size();


    for (size_t k = 0; k < msg_object.num_objects; k++) {
      referee_pkg::msg::Object obj;
      obj.target_type=start.allObjects[k].shape+"_"+start.allObjects[k].color;
      for (size_t index=0 ;index<start.allObjects[k].points.size();index++) {
          geometry_msgs::msg::Point corner;
          corner.x = start.allObjects[k].points[index].x;
          corner.y = start.allObjects[k].points[index].y;
          corner.z = 0.0;
          obj.corners.push_back(corner);
        }
      msg_object.objects.push_back(obj);
    }

    Target_pub->publish(msg_object);
    RCLCPP_INFO(this->get_logger(), "Published %d targets",
                msg_object.num_objects);
  
    }
  catch (const cv_bridge::Exception &e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  } 
  catch (const std::exception &e) {
    RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
  }
}

