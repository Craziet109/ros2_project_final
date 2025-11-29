#include "rclcpp/rclcpp.hpp"
#include "referee_pkg/msg/multi_object.hpp"
#include "referee_pkg/msg/object.hpp"
#include "referee_pkg/srv/hit_armor.hpp"
#include "std_msgs/msg/header.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/core/mat.hpp"
#include <deque>
#include <mutex>
#include <string>
#include <vector>
#include <limits>
#include <cmath>
#include <functional>
#include <algorithm>
#include <memory>


class ShooterNode : public relcpp::Node
{
    private:

        struct CachedArmorData {                   //缓存条目结构
            rclcpp::Time timestamp;                // 数据时间戳
            geometry_msgs::msg::Point world_point; // 世界坐标系下的装甲板位置
        };

        std::deque<CachedArmorData> armor_cache_;  // 缓存容器
        std::mutex cache_mutex_;                   // 线程安全保护
        
        rclcpp::Subscription<referee_pkg::msg::MultiObject>::SharedPtr multi_obj_sub_;//话题
        
        rclcpp::Service<referee_pkg::srv::HitArmor>::SharedPtr hit_armor_server_;//服务

        const size_t CACHE_SIZE = 1000;
        const double PROJECTILE_SPEED = 1.5;
        const double MAX_TIME_DIFF = 0.1;  // 最大允许100ms时间差

        const double armor_h = 0.705;
        const double armor_w = 0.230;

        const double speed = 25.00;

        // 相机参数
        cv::Mat camera_matrix_K = (cv::Mat_<double>(3, 3) <<
            1108.383, 0.0,    640.0,
            0.0,    1108.383, 640.0,
            0.0,    0.0,    1.0
        );
        cv::Mat extrinsic_matrix_l = (cv::Mat_<double>(3, 4) <<
            0.0, 1.0, 0.0, 0.0,  
            -1.0, 0.0, 0.0, 0.0, 
            0.0, 0.0, 1.0, -0.5 
        );
        cv::Mat dist_coeffs_5 = (cv::Mat_<float>(5, 1) << 0.0, 0.0, 0.0, 0.0, 0.0);

    public:

        ShooterNode() : Node("shooter_node")
    {
        // 订阅目标消息
        multi_obj_sub_ = this->create_subscription<referee_pkg::msg::MultiObject>(
            "/vision/target",
            10,
            std::bind(&ShooterNode::multi_obj_callback, this, std::placeholders::_1));

        // 创建服务端 
        hit_armor_server_ = this->create_service<referee_pkg::srv::HitArmor>(
            "/referee/hit_arror",  // 保持错误拼写哦
            std::bind(&ShooterNode::hit_armor_service_callback, this, 
            std::placeholders::_1, std::placeholders::_2));
    }

    private:
        // 接收目标消息，缓存“时间戳+世界系坐标”
        void multi_obj_callback(const referee_pkg::msg::MultiObject::SharedPtr msg)
        {
            RCLCPP_INFO(this->get_loger(),"成功进入回调函数");
            //写入的锁保护
            std::lock_guard<std::mutex> lock(cache_mutex_);
            
            // 遍历消息中的所有目标对象）
            for (const auto& obj : msg->objects) {
            // 筛选“装甲板”目标
                if (obj.target_type.find("armor") != std::string::npos) {
                
                    // 根据装甲板角点和相机内参，计算相机坐标系下的装甲板中心点
                    geometry_msgs::msg::Point camera_point = calculate_armor_center(
                        obj.corners,          // 装甲板角点
                        camera_matrix_K,      // 相机内参矩阵
                        dist_coeffs_5,        // 相机畸变系数
                        armor_h, armor_w      // 装甲板实际尺寸
                    );
            
                    // 将相机系中心点转换为世界坐标系下的坐标
                    geometry_msgs::msg::Point world_point = camera_to_world_point(
                        camera_point,          // 相机系下的装甲板中心点
                        extrinsic_matrix_l     // 相机外参矩阵
                    );
            
                    CachedArmorData cached_data;
                    //数据写入缓存
                    cached_data.timestamp = msg->header.stamp;
                    cached_data.world_point = world_point;

                    armor_cache_.push_back(cached_data);写入容器
                
                    if (armor_cache_.size() > CACHE_SIZE) {
                        armor_cache_.pop_front();  // 容器控制：移除最旧数据
                    }
                }
            }
        }

        geometry_msgs::msg::Point calculate_armor_center(
            const std::vector<geometry_msgs::msg::Point>& corners,
            const cv::Mat& camera_matrix,
            const cv::Mat& dist_coeffs,
            const double armor_h,
            const double armor_w)
        {
            RCLCPP_INFO(this->get_loger(),"开始计算装甲板相机参考系坐标");
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
            center.x = static_cast<double>(tvec.at<float>(2, 0));
            center.y = -1*static_cast<double>(tvec.at<float>(0, 0));
            center.z = -1*static_cast<double>(tvec.at<float>(1, 0));
            return center;
        }

        geometry_msgs::msg::Point camera_to_world_point(
            const geometry_msgs::msg::Point& camera_point,
            const cv::Mat& extrinsic_w2c)
        {
            RCLCPP_INFO(this->get_loger(),"开始进行坐标系转换");
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

        //欧拉角计算
        geometry_msgs::msg::Vector3 calculate_euler_angle(
            const geometry_msgs::msg::Point& spatial_point,
            double g, 
            double projectile_speed)
        {
            RCLCPP_INFO(this->get_loger(),"开始计算欧拉角");
            geometry_msgs::msg::Vector3 euler;
            euler.x = 0.0; // 滚转角默认设为0

            const double x = spatial_point.x;
            const double y = spatial_point.y;
            const double z = spatial_point.z;
            const double d_xy = std::sqrt(x*x + y*y);

            // 计算偏航角（范围[-π, π]）
            euler.z = std::atan2(y, x);

            // 目标点在Z轴上（d_xy趋近于0），垂直发射（不过大概
            if (d_xy < 1e-6){
                euler.y = (z >= 0) ? M_PI_2 : -M_PI_2;
                return euler;
            }

            // 构建一元二次方程 A*u² + B*u + C = 0（u = t²）
            const double A = (g * g) / 4.0;
            const double B = - ( (g * z) / 2.0 + projectile_speed * projectile_speed );
            const double C = d_xy * d_xy;

            //r如果飞不到直接输出默认值
            const double discriminant = B*B - 4*A*C;
            if (discriminant < 1e-6 || projectile_speed <= 0) {
                RCLCPP_WARN(this->get_logger(), "Target unreachable (speed: %.2f, discriminant: %.2f)", projectile_speed, discriminant);
                euler.y = 0.0;  // 设默认值，避免崩溃
                return euler;
            }
        
            const double sqrt_disc = std::sqrt(discriminant);
            double u = (-B - sqrt_disc) / (2*A);
            u = (u > 1e-6) ? u : (-B + sqrt_disc) / (2*A);

            // 计算飞行时间、速度分量与俯仰角
            const double t = std::sqrt(std::max(u, 1e-6));
            const double v_xy = d_xy / t;
            const double v_z = (z + 0.5 * g * t * t) / t;
            euler.y = std::atan2(v_z, v_xy);

            return euler;
        }

        bool find_closest_cached_data(
            const rclcpp::Time& target_stamp, 
            CachedArmorData& result)
        {
            // 读取的锁保护
            std::lock_guard<std::mutex> lock(cache_mutex_);
    
            if (armor_cache_.empty()) {
                return false;
            }

            double min_diff = std::numeric_limits<double>::max();  // 初始化为最大值
            bool found = false;

            // 遍历所有缓存数据
            for (const auto& data : armor_cache_) {
                double diff = std::abs((data.timestamp - target_stamp).seconds());
        
                // 双重条件：时间差最小且在允许范围内
                if (diff < min_diff && diff <= MAX_TIME_DIFF) {
                        min_diff = diff;
                        result = data;
                        found = true;
                }
            }

            return found;
        }

        void hit_armor_service_callback(
            const std::shared_ptr<referee_pkg::srv::HitArmor::Request> request,
            std::shared_ptr<referee_pkg::srv::HitArmor::Response> response)
        {
            //提取请求的时间戳和g
            rclcpp::Time request_stamp = request->header.stamp;
            double g = request->g;
            RCLCPP_INFO(this->get_logger(), "收到请求：时间戳=%.3f，g=%.2f", 
                request_stamp.seconds(), g);

            // 查找缓存中时间戳最接近的数据
            CachedArmorData closest_data;
            if (!find_closest_cached_data(request_stamp, closest_data)) {
                RCLCPP_WARN(this->get_logger(), "未找到匹配的缓存数据");
                return;
            }

            // 计算欧拉角
            geometry_msgs::msg::Vector3 euler = calculate_euler_angle(spatial_point_world, g, speed)

            // 返回响应
            response->yaw = euler.z;
            response->pitch = euler.y;
            response->roll = euler.x;
    
            RCLCPP_INFO(this->get_logger(), "计算完成：yaw=%.3f, pitch=%.3f", 
                response->yaw, response->pitch);
        }
}

int main(int argc, char* argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ShooterNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
