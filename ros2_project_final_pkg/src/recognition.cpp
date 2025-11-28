#include "recognition.hpp"


void Recognition::Integration(Mat &original_img){
    if (original_img.empty())
    {cout<<"error in imread original_img"<<endl;return;}

    allObjects.clear();

    Mat process_img,hsv_img;
    //imshow("orginal_img",original_img);waitKey(0);
    string color;
    //转为hsv图像后检测颜色
    cvtColor(original_img,hsv_img, cv::COLOR_BGR2HSV);
    //用不同颜色进行筛选，并进行进一步操作
    for (int i=0;i<4;i++){
        color=ColorType[i];
        ColorFiltering(hsv_img,process_img,color);
        GetCountoursAndJudge(hsv_img,process_img,color);
    }
    //可视化并在终端打印出allObjects的内容
    Visualization(original_img);
    ShowAllObjects();
}

void Recognition::findArmourIntegration(Mat &original_img){
    allObjects.clear();

    //正在追踪的画先更新
    if (is_tracking) {
        bool tracking_success = updateTracker(original_img);
        if (tracking_success) {
            TrackVisualization(original_img);
            Visualization(original_img);
            //return;  
            } 
            else {
            is_tracking = false;
            tracking_fail_count++;
            cout << "Tracking failed, count:" << tracking_fail_count << endl;
            if (tracking_fail_count>=max_track_fail) {
                cout << "Too many tracking failures" << endl;
                tracking_fail_count = 0;
                is_tracking=false;}
        }
    }
    
    Mat processed_img, hsv_img;
    cvtColor(original_img, hsv_img, cv::COLOR_BGR2HSV);
    ColorFiltering(hsv_img, processed_img, "black");
    JudgeArmour(original_img, processed_img);
    
    // 如果检测到装甲板且不在追踪状态，初始化追踪器
    if (!allObjects.empty() && !is_tracking) {
        // 这里只实现了一个的跟踪，来不及写其他了TT
        const ObjectInfo& obj = allObjects[0];
        if (obj.points.size() == 4) {
            Rect2d bbox=obj.AmourRect;
            bbox.x = max(0.0,bbox.x);
            bbox.y = max(0.0,bbox.y);
            bbox.width = min(bbox.width, original_img.cols - bbox.x);
            bbox.height = min(bbox.height, original_img.rows - bbox.y);
            
            //大小要合理
            if (bbox.width > 20 && bbox.height > 20) {
                initTracker(original_img,bbox);
                tracking_fail_count = 0;
                cout << "New tracker initialized with expanded bbox: "<<bbox<<endl;
            }
        }
    }
    ShowAllObjects();
    Visualization(original_img);
}


void Recognition::initTracker(Mat &frame, Rect2d bbox) {
    // 确保bbox在图像范围内
    bbox.x = max(0.0, bbox.x);
    bbox.y = max(0.0, bbox.y);
    bbox.width = min(bbox.width, frame.cols - bbox.x);
    bbox.height = min(bbox.height, frame.rows - bbox.y);
    
    if (bbox.width > 20 && bbox.height > 20) { 
        tracking_bbox = bbox;
        tracking_template = frame(bbox).clone();
        is_tracking = true;
        cout << "Tracker initialized"<<endl;}
}

bool Recognition::updateTracker(Mat &frame) {
    if (tracking_template.empty() || !is_tracking) {
        return false;}

    // 扩大搜索区域,同时防止越界
    int search=300;
    Rect search_roi;
    search_roi.x = max(0, static_cast<int>(tracking_bbox.x)-search);
    search_roi.y = max(0, static_cast<int>(tracking_bbox.y)-search);
    search_roi.width = min(frame.cols-search_roi.x, static_cast<int>(tracking_bbox.width)+2*search);
    search_roi.height = min(frame.rows-search_roi.y, static_cast<int>(tracking_bbox.height)+2*search);
    
    Mat search_area=frame(search_roi);
    // 模板匹配
    Mat result;
    matchTemplate(search_area, tracking_template, result, TM_CCOEFF_NORMED);
    double min_val, max_val;
    Point min_loc, max_loc;
    minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);
    
    cout << "Template matching max value: " << max_val << endl;
    
    // 降低匹配阈值以提高追踪稳定性
    double threshold = 0.25;
    if (max_val > threshold) {
        // 更新追踪框位置
        tracking_bbox.x = search_roi.x + max_loc.x;
        tracking_bbox.y = search_roi.y + max_loc.y;
        return true;}
    cout << "Template matching failed, value below threshold: " << max_val << endl;
    return false;
}

void Recognition::TrackVisualization(Mat &frame) {
    rectangle(frame, tracking_bbox, Scalar(0, 255, 0), 3);
    Point center(static_cast<int>(tracking_bbox.x + tracking_bbox.width/2),
                static_cast<int>(tracking_bbox.y + tracking_bbox.height/2));
    circle(frame, center, 5, Scalar(0, 255, 255), -1);
    // 绘制坐标信息
    string coord_text = "X: " + to_string(static_cast<int>(tracking_bbox.x)) + 
                       " Y: " + to_string(static_cast<int>(tracking_bbox.y));
    putText(frame, coord_text, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
    string size_text = "W: " + to_string(static_cast<int>(tracking_bbox.width)) + 
                      " H: " + to_string(static_cast<int>(tracking_bbox.height));
    putText(frame, size_text, Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
}


void Recognition::ColorFiltering(Mat &hsv_img,Mat &processed_img,string colortype){
    Mat mask;
    if (colortype==ColorType[0]){
        Mat mask1, mask2;
        inRange(hsv_img,red_lower1,red_upper1,mask1);
        inRange(hsv_img,red_lower2,red_upper2,mask2);
        mask=mask1 | mask2;
    }
    else if(colortype==ColorType[3]){
        Mat mask1, mask2;
        inRange(hsv_img,black_lower,black_upper1,mask1);
        inRange(hsv_img,black_lower,black_upper2,mask2);
        mask=mask1 | mask2;
    }
    else if (colortype==ColorType[1]){
        inRange(hsv_img,green_lower,green_upper,mask);}

    else if (colortype==ColorType[2]){
        inRange(hsv_img,blue_lower,blue_upper,mask);
    }
    else {cout<<"error in input color"<<endl;return;}
    //进行基本的图像转换
    //imshow("mask",mask);waitKey(0);
    //改了一下形态学操作
    morphologyEx(mask,processed_img,MORPH_OPEN,kernel_open);
    morphologyEx(processed_img,processed_img,MORPH_CLOSE,kernel__close);
    //imshow("after ColorFilter",processed_img);waitKey(0);
 }


void Recognition::GetCountoursAndJudge(Mat &original_img,Mat &AfterColorFiltering,string colortype){
    (void)original_img; 
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(AfterColorFiltering,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    int area;
    vector<vector<Point>> conPoly(contours.size());
    string objectType;
    //因为size（）返回的是无符号的，所以用size_t
    for (size_t  i=0;i<contours.size();i++){
        area=contourArea(contours[i]);
        //cout<<area<<endl;
        //去除噪声，参数还要调一下
        if (area>Min_area){
            float peri=arcLength(contours[i],true);
            approxPolyDP(contours[i],conPoly[i],0.01*peri,true);

            //drawContours(original_img,conPoly,i,Scalar(0,0,0),2);
            vector<Point2f> Points;
            int objCor=(int)conPoly[i].size();
            //判断形状
            if (objCor==4){
                objectType="rect";
                Level1_arrangePoints(Points,conPoly[i]);
                }
            else if (objCor>6 && JudgeSphere(contours[i],peri,area)){
                objectType="circle";
                Points=sphere_locate(contours[i]);
            }
            else {cout<<"no such a structure in "<<colortype<<endl;continue;}

            //创建对应的obj并加到allObjects里
            if (Points.size()==4){
            ObjectInfo obj;
            obj.shape = objectType;    
            obj.color = colortype;    
            obj.points = Points;       
            allObjects.push_back(obj); 
            Points.clear();}
        }}
   //imshow("approxPoly",original_img);waitKey(0);
}
            


void Recognition::JudgeArmour(Mat &original_img,Mat &AfterColorFiltering){
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(AfterColorFiltering,contours,hierarchy,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
    int area;
    vector<vector<Point>> conPoly(contours.size());
    string objectType;
    for (size_t  i=0;i<contours.size();i++){
        area=contourArea(contours[i]);
        //分别用面积和长宽比筛选出装甲板
        if (area>Min_area_armour){
            float peri=arcLength(contours[i],true);
            approxPolyDP(contours[i],conPoly[i],0.01*peri,true);

            //drawContours(original_img,conPoly,i,Scalar(255,0,0),2);
            //imshow("conPoly",original_img);
            //找外接矩形
            Rect boundRect=boundingRect(conPoly[i]);
            float ratio=boundRect.width/boundRect.height;
            //cout<<“the ratio is ”<<ratio;
            if (ratio< Min_ratio || ratio> Max_ratio){cout<<"no such a structure"<<endl;continue;}
            Mat warp_img;
            //找四个点
            CutArmour(original_img,warp_img,boundRect);
            locate_Armour(warp_img,boundRect);
        }
    } 
}

void Recognition::Level1_arrangePoints(vector<Point2f>& points,vector<Point>& conPoly){
  //排除异常数据的干扰
    if (points.size() != 4) return;
    int left=0;
    int low1=0;
    int low2=1;
    if (conPoly[low2].y>conPoly[low1].y){
    swap(low1,low2);}
    for (int i=2; i<4;i++) {
    if (conPoly[i].y>conPoly[low1].y){
        low2=low1;  
        low1=i;}  
    else if (conPoly[i].y>conPoly[low2].y){
        low2=i;}}

    if (conPoly[low1].x<conPoly[low2].x){left=low1;}
    else{left=low2;}
    points.clear();
    points.push_back(Point2f(conPoly[left])); // 左下
    points.push_back(Point2f(conPoly[(left+1)%4])); 
    points.push_back(Point2f(conPoly[(left+2)%4])); 
    points.push_back(Point2f(conPoly[(left+3)%4])); 
}


void Recognition::Armour_arrangePoints(vector<Point2f>& Points){
    vector<Point2f> points_arrange;
    vector<int> arrange;
    if (Points.size() != 4) return;
    if (Points[0].x>Points[2].x){
        arrange={3,1,0,2};
    }
    else{
        arrange={1,3,2,0};
    }
    for (int i=0;i<4;i++){
    points_arrange.push_back(Points[arrange[i]]);}
    Points=points_arrange;

}

void Recognition::locate_Armour(Mat warp_img,Rect boundRect){
    int number=number_recognition(warp_img);
    if (number==0){cout<<"error in number recognition";return;}
    string color;
    Mat hsv_img,process_img;
    cvtColor(warp_img,hsv_img, cv::COLOR_BGR2HSV);
    for (int i=0;i<3;i++){
        color=ColorType[i];
        ColorFiltering(hsv_img,process_img,color);
        //imshow("after color filtering",process_img);

        vector<vector<Point>> contours;
        findContours(process_img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        if (contours.empty()) {
        cout<<"no "<<color<<endl;continue;}//颜色不符合
        float area;
        vector<Point2f> Points;
        for (size_t j=0;j<contours.size();j++){
        area=contourArea(contours[j]);

        cout<<"the area is "<<area<<endl;

        //根据面积筛选出灯条
        if (area<Min_area_light || area>Max_area_light){cout<<"too small or big"<<endl;continue;}
        
        // 找到最高点和最低点，作为目标点
        Point2f highestPoint=Point2f(warp_img.cols, warp_img.rows);
        Point2f lowerstPoint=Point2f(0.0f,0.0f);
        for (const auto& point : contours[j]) {
            if (point.y<highestPoint.y) {
                highestPoint=point;}
            if (point.y>lowerstPoint.y){
                lowerstPoint=point;}
            }
        
        Points.push_back(highestPoint);
        Points.push_back(lowerstPoint);
        
    }
    
    //两边各一个灯条对应四个点，要按照要求排序
    Armour_arrangePoints(Points);
    transformLocation(Points,boundRect,color,number) ;
    //因为灯条颜色只有一种可能，找到对应的直接退出循环
    return;
}
}

void Recognition::transformLocation(vector<Point2f> Points,Rect boundRect,string &color,int number){
    vector<Point2f> transformPoints;
    float w=boundRect.x;
    float h=boundRect.y;
    for (size_t i=0;i<Points.size();i++){
        Point2f newPoint; // 存储转换后坐标
        newPoint.x = Points[i].x + w;
        newPoint.y = Points[i].y + h;
        transformPoints.push_back(newPoint);
        }
        //转换后将所有的信息进行存储
            ObjectInfo obj;
            //注意此处字符串的拼接
            obj.shape = string("Armor")+"_"+to_string(number);    
            obj.color = color;    
            obj.points = transformPoints;  
            obj.AmourRect=boundRect;     
            allObjects.push_back(obj); 
            transformPoints.clear();
}


bool Recognition::JudgeSphere(vector<Point>& contour,float peri,int area){
    float circularity=(4*CV_PI*area)/(peri*peri);
    if (circularity<0.8){return false;}
    Point2f center;
    float radius;
    minEnclosingCircle(contour,center,radius);
    float circlerArea =CV_PI*radius*radius;
    float areaRadio=area/circlerArea;
    return (areaRadio>0.8);
}


vector<Point2f> Recognition::sphere_locate(vector<Point> contour){
    Point2f center;
    float radius;
    vector<Point2f> sphere_points;
    minEnclosingCircle(contour,center,radius);
    sphere_points.push_back(Point2f(center.x-radius,center.y));
    sphere_points.push_back(Point2f(center.x,center.y+radius));
    sphere_points.push_back(Point2f(center.x+radius,center.y));
    sphere_points.push_back(Point2f(center.x,center.y-radius));
    return sphere_points;
}


void Recognition::CutArmour(Mat &original_img,Mat &warp_img,Rect boundingRect){
    float h=boundingRect.height;
    float w=boundingRect.width;
    float y = static_cast<float>(boundingRect.y);
    float x = static_cast<float>(boundingRect.x);

    Point2f dst[4]={{0.0f,0.0f},{0.0f,h},{w,h},{w,0.0f}};
    vector<Point2f> Points={{x,y},{x,y+h},{x+w,y+h},{x+w,y}};
    //映射要用指针，所以要用data()
    Mat matrix=getPerspectiveTransform(Points.data(),dst);
    warpPerspective(original_img,warp_img,matrix,Point(w,h));
    //imshow("after warp",warp_img);
}



void Recognition::Visualization(Mat &processed_img){
    vector<string> point_numbers={"1","2","3","4"};
    vector<Scalar> point_colors={
    Scalar(255,0,0),Scalar(0,255,0),Scalar(0,255,255),Scalar(255,0,255)};
    for (size_t j =0;j<allObjects.size();j++){
        const ObjectInfo& obj = allObjects[j];
        const vector<Point2f>& Points = obj.points;
        if (Points.size() != 4){continue;}
        //标出物体的形状和颜色
        putText(processed_img,obj.color+"_"+obj.shape,Point(Points[3].x-15,Points[3].y-30),FONT_HERSHEY_COMPLEX,0.5,Scalar(0,0,0),1.5);
        for (int i=0;i<4;i++){
            //标出坐标位置并标号
            circle(processed_img,Points[i],6,point_colors[i],-1);
            putText(processed_img,point_numbers[i],Point(Points[i].x-15,Points[i].y-10),FONT_HERSHEY_COMPLEX,0.7,point_colors[i],1);
    }
}
    imshow("Visualization",processed_img);
    //waitKey(0);
}


void Recognition::ShowAllObjects(){
    for (size_t i=0;i<allObjects.size();i++){
        cout<<i+1<<" : ";
        if (allObjects[i].points.size()!=4){cout<<"no 4 points in object"<<i;continue;}
        for (size_t j=0;j<4;j++){
            cout<<allObjects[i].points[j]<<" ";
        }
        cout<<allObjects[i].shape<<" "<<allObjects[i].color<<endl;
    }
}


int Recognition::number_recognition(Mat &afterWarp){
    string package_share_dir = ament_index_cpp::get_package_share_directory("project1");
    string model_path = package_share_dir + "/models/digit_model_improved.pth";
    numberRecognition classifier(model_path);
        // 预测
        auto result=classifier.predict(afterWarp);
        cout << "预测数字: "<< result.first<<std::endl;
        cout << "置信度: "<<result.second<<std::endl;
        return result.first;
}
