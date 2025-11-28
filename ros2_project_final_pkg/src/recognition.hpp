#ifndef RECOGNITION_HPP
#define RECOGNITION_HPP


#include <algorithm>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include <filesystem>

using namespace std;
using namespace cv;

class Recognition{
public:
    void Integration(Mat &original_img); 
    void ColorFiltering(Mat &original_img,Mat &processed_img,string colortype);
    void GetCountoursAndJudge(Mat &original_img,Mat &AfterColorFiltering,string colortype);
    bool JudgeSphere(vector<Point>& contour,float peri,int area);
    vector<Point2f> sphere_locate(vector<Point> contour);
    
    void Level1_arrangePoints(vector<Point2f>& points,vector<Point>& conPoly);
    void Armour_arrangePoints(vector<Point2f>& Points);

    void findArmourIntegration(Mat &original_img);
    void JudgeArmour(Mat &original_img,Mat &AfterColorFiltering);
    void CutArmour(Mat &original_img,Mat &warp_img,Rect boundingRect);
    void locate_Armour(Mat warp_img,Rect boundRect);
    void transformLocation(vector<Point2f> Points,Rect boundRect,string &color,int number);
    int number_recognition(Mat &afterWarp);

    // 追踪相关函数
    void initTracker(Mat &frame, Rect2d bbox);
    bool updateTracker(Mat &frame);
    void TrackVisualization(Mat &frame);

    void Visualization(Mat &processed_img);
    void ShowAllObjects();
    
    Recognition() 
        : red_upper1(10, 255, 255), red_lower1(0, 120, 70),
          red_upper2(180, 255, 255), red_lower2(170, 120, 70),
          green_upper(80, 255, 255), green_lower(40, 40, 40),
          blue_upper(145, 255, 255), blue_lower(75, 40, 40),
          black_upper1(180,255,50), black_upper2(180,50,80), black_lower(0,0,0),
          is_tracking(false)  // 初始化追踪状态为false
    {};
    
    
    struct ObjectInfo {
        string shape;
        string color;
        vector<Point2f> points;
        Rect AmourRect;
    };
    vector<ObjectInfo> allObjects;

private: 

    
    string ColorType[4]={"red","green","blue","black"};
    Scalar red_upper1,red_lower1,red_upper2,red_lower2;
    Scalar green_upper,green_lower;
    Scalar blue_upper,blue_lower;
    Scalar black_upper1,black_upper2,black_lower;

    Mat kernel_open=getStructuringElement(MORPH_ELLIPSE,Size(3,3));
    Mat kernel__close=getStructuringElement(MORPH_ELLIPSE,Size(7,7));

    const int Min_area=1500;
    const int Min_area_armour=1500;
    const int Min_area_light=50;
    const int Max_area_light=8000;

    const float Min_ratio=0.8;
    const float Max_ratio=1.1;

    bool is_tracking;
    Rect2d tracking_bbox;
    Mat tracking_template;
    int tracking_fail_count;
    const int max_track_fail=5;  // 最大连续追踪失败次数
};



class numberRecognition {
private:
    torch::jit::script::Module model;
public:
    numberRecognition(const std::string& model_path) {
            model = torch::jit::load(model_path);
            model.eval();
    }
    std::pair<int, float> predict(const cv::Mat& image) {
            // 预处理图像
            torch::Tensor tensor = preprocess_image(image);
            
            auto output = model.forward({tensor}).toTensor();
            auto probs = torch::softmax(output, 1);
            auto predicted = torch::argmax(output, 1);
            
            float confidence = probs[0][predicted[0]].item<float>();
            int digit = predicted[0].item<int>() + 1;
            
            return {digit, confidence};
    }
private:
    torch::Tensor preprocess_image(const cv::Mat& image) {
        cv::Mat processed;
        // 转换为灰度图
        cv::cvtColor(image,processed, cv::COLOR_BGR2GRAY);
        // 调整大小
        cv::resize(processed,processed, cv::Size(64, 64));
        processed.convertTo(processed, CV_32F, 1.0 / 255.0);
        processed = (processed - 0.5) / 0.5;
        torch::Tensor tensor = torch::from_blob(
            processed.data, 
            {1, 1, 64, 64}, 
            torch::kFloat32).clone();
        return tensor;
    }
};

#endif
