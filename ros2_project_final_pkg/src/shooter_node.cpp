#include "rclcpp/rclcpp.hpp"
#include "referee_pkg/msg/multi_object.hpp"
#include "referee_pkg/srv/hit_armor.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "std_msgs/msg/header.hpp"
#include <message_filters/cache.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

// 缓存数据结构：仅保留“时间戳+世界系坐标”
struct CachedArmorData {
    rclcpp::Time timestamp;
    geometry_msgs::msg::Point world_point;
};

class ShooterNode : public rclcpp::Node
{
private:
    // 装甲板参数
    const double armor_h = 0.705;
    const double armor_w = 0.230;

    // 相机参数
    cv::Mat camera_matrix_K = (cv::Mat_<double>(3, 3) <<
            554.383, 0.0,    320.0,
            0.0,    554.383, 320.0,
            0.0,    0.0,    1.0
        );
    cv::Mat extrinsic_matrix_l = (cv::Mat_<double>(3, 4) <<
            0.0, 1.0, 0.0, 0.0,  
            -1.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, -0.5 
        );
    cv::Mat dist_coeffs_5 = (cv::Mat_<float>(5, 1) << 0.0, 0.0, 0.0, 0.0, 0.0);

    // 核心组件
    rclcpp::Subscription<referee_pkg::msg::MultiObject>::SharedPtr multi_obj_sub_;
    std::shared_ptr<message_filters::Cache<CachedArmorData>> armor_cache_;
    rclcpp::Service<referee_pkg::srv::HitArmor>::SharedPtr hit_armor_server_;

    // 配置：缓存300条 
    const size_t CACHE_SIZE = 300;
    const double PROJECTILE_SPEED = 1.5;

public:
    ShooterNode() : Node("shooter_node")
    {
        // 初始化缓存
        armor_cache_ = std::make_shared<message_filters::Cache<CachedArmorData>>(CACHE_SIZE);

        // 订阅目标消息
        multi_obj_sub_ = this->create_subscription<referee_pkg::msg::MultiObject>(
            "/vision/target",
            10,
            std::bind(&ShooterNode::multi_obj_callback, this, std::placeholders::_1));

        // 创建服务端
        hit_armor_server_ = this->create_service<referee_pkg::srv::HitArmor>(
            "/referee/hit_arror",
            std::bind(&ShooterNode::hit_armor_service_callback, this, 
                      std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(this->get_logger(), "节点启动：缓存300条，弹丸速度固定为%.1f", PROJECTILE_SPEED);
    }

private:
    // 接收目标消息，缓存“时间戳+世界系坐标”
    void multi_obj_callback(const referee_pkg::msg::MultiObject::SharedPtr msg)
    {
        for (const auto& obj : msg->objects) {
            if (obj.target_type.find("armor") != std::string::npos) {
                // 计算相机系坐标
                geometry_msgs::msg::Point camera_point = calculate_armor_center(
                    obj.corners, camera_matrix_K, dist_coeffs_5, armor_h, armor_w);
                
                // 转换为世界系坐标
                geometry_msgs::msg::Point world_point = camera_to_world_point(
                    camera_point, extrinsic_matrix_l);
                
                // 缓存数据（仅时间戳+世界坐标）
                CachedArmorData cached_data;
                cached_data.timestamp = msg->header.stamp;
                cached_data.world_point = world_point;
                armor_cache_->add(cached_data);
            }
        }
    }

    // 服务回调：仅用请求的“时间戳”和“g”
    void hit_armor_service_callback(
        const std::shared_ptr<referee_pkg::srv::HitArmor::Request> request,
        std::shared_ptr<referee_pkg::srv::HitArmor::Response> response)
    {
        // 1. 提取请求的时间戳和g
        rclcpp::Time request_stamp = request->header.stamp;
        double g = request->g;
        RCLCPP_INFO(this->get_logger(), "收到请求：时间戳=%.3f，g=%.2f",
                    request_stamp.seconds(), g);

        // 2. 查找缓存中时间戳最接近的数据
        auto cached_list = armor_cache_->getAll();
        if (cached_list.empty()) {
            RCLCPP_WARN(this->get_logger(), "缓存为空，无数据");
            return;
        }

        auto closest_data = find_closest_cached_data(cached_list, request_stamp);
        if (closest_data.timestamp.seconds() < 0) {
            RCLCPP_WARN(this->get_logger(), "无匹配数据");
            return;
        }

        // 3. 计算欧拉角
        geometry_msgs::msg::Vector3 euler = calculate_euler_angle(
            closest_data.world_point, g, PROJECTILE_SPEED);

        // 4. 返回响应（匹配服务定义）
        response->yaw = euler.z;
        response->pitch = euler.y;
        response->roll = euler.x;
    }

    // 查找时间戳最接近的缓存数据（±100ms）
    CachedArmorData find_closest_cached_data(
        const std::vector<CachedArmorData>& list,
        const rclcpp::Time& target_stamp)
    {
        CachedArmorData result;
        result.timestamp = rclcpp::Time(-1);
        double min_diff = 0.1;  // 最大允许100ms误差

        for (const auto& data : list) {
            double diff = std::abs((data.timestamp - target_stamp).seconds());
            if (diff < min_diff) {
                min_diff = diff;
                result = data;
            }
        }
        return result;
    }

    // 装甲板中心点计算（相机系）
    geometry_msgs::msg::Point calculate_armor_center(
        const std::vector<geometry_msgs::msg::Point>& corners,
        const cv::Mat& camera_matrix,
        const cv::Mat& dist_coeffs,
        const double armor_h,
        const double armor_w)
    {
        std::vector<cv::Point2f> image_points;
        for (const auto& corner : corners) {
            image_points.emplace_back(static_cast<float>(corner.x), static_cast<float>(corner.y));
        }

        std::vector<cv::Point3f> object_points;
        double half_h = armor_h / 2.0;
        double half_w = armor_w / 2.0;
        object_points.emplace_back(-half_h,  half_w, 0.0);
        object_points.emplace_back( half_h,  half_w, 0.0);
        object_points.emplace_back( half_h, -half_w, 0.0);
        object_points.emplace_back(-half_h, -half_w, 0.0);

        cv::Mat rvec, tvec;
        cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

        geometry_msgs::msg::Point center;
        center.x = static_cast<double>(tvec.at<float>(0, 0));
        center.y = static_cast<double>(tvec.at<float>(1, 0));
        center.z = static_cast<double>(tvec.at<float>(2, 0));
        return center;
    }

    // 相机系→世界系转换
    geometry_msgs::msg::Point camera_to_world_point(
        const geometry_msgs::msg::Point& camera_point,
        const cv::Mat& extrinsic_w2c)
    {
        cv::Mat R = extrinsic_w2c.colRange(0, 3);
        cv::Mat t = extrinsic_w2c.col(3);

        cv::Mat P_cam = (cv::Mat_<double>(3, 1) << camera_point.x, camera_point.y, camera_point.z);
        cv::Mat P_world = R.t() * (P_cam - t);

        geometry_msgs::msg::Point world_point;
        world_point.x = P_world.at<double>(0, 0);
        world_point.y = P_world.at<double>(1, 0);
        world_point.z = P_world.at<double>(2, 0);
        return world_point;
    }

    // 欧拉角计算
    geometry_msgs::msg::Vector3 calculate_euler_angle(
        const geometry_msgs::msg::Point& world_point,
        double g,
        double projectile_speed)
    {
        geometry_msgs::msg::Vector3 euler;
        euler.x = 0.0;

        double x = world_point.x;
        double y = world_point.y;
        double z = world_point.z;
        double d_xy = std::sqrt(x*x + y*y);

        euler.z = std::atan2(y, x);

        if (d_xy < 1e-6) {
            euler.y = (z >= 0) ? M_PI_2 : -M_PI_2;
            return euler;
        }

        double A = (g * g) / 4.0;
        double B = - (g * z / 2.0 + projectile_speed * projectile_speed);
        double C = d_xy * d_xy;
        double discriminant = B*B - 4*A*C;

        if (discriminant < 1e-6) {
            RCLCPP_WARN(this->get_logger(), "目标不可达，判别式=%.2f", discriminant);
            euler.y = 0.0;
            return euler;
        }

        double sqrt_disc = std::sqrt(discriminant);
        double u = (-B - sqrt_disc) / (2*A);
        u = (u > 1e-6) ? u : (-B + sqrt_disc) / (2*A);
        double t = std::sqrt(std::max(u, 1e-6));
        double v_xy = d_xy / t;
        double v_z = (z + 0.5 * g * t * t) / t;

        euler.y = std::atan2(v_z, v_xy);
        return euler;
    }
};

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ShooterNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
