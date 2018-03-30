///////////////////////////////////////////////////////////////////////////////////////////////////////
/// DemoTracker.cpp
/// 
/// Description:
/// This program shows you how to use FaceAlignment class in tracking facial landmarks in a video or realtime. 
/// There are two modes: VIDEO, REALTIME.
/// In the VIDEO mode, the program reads input from a video and perform tracking.
/// In the REALTIME mode, the program reads input from the first camera it finds and perform tracking.
/// Note that tracking is performed on the largest face found. The face is detected through OpenCV.
/// In this version,  we add head pose estimation. 
///
/// Dependencies: None. OpenCV DLLs and include folders are copied.
///
/// Author: Xuehan Xiong, xiong828@gmail.com
///
/// Creation Date: 1/24/2014
///
/// Version: 1.2
///
/// Citation: 
/// Xuehan Xiong, Fernando de la Torre, Supervised Descent Method and Its Application to Face Alignment. CVPR, 2013
///////////////////////////////////////////////////////////////////////////////////////////////////////

#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <time.h>
#include <Windows.h>
#include <intraface/FaceAlignment.h>
#include <intraface/XXDescriptor.h>

using namespace std;
using namespace cv;

bool compareRect(cv::Rect r1, cv::Rect r2) { return r1.height < r2.height; }

// 2 modes: REALTIME,VIDEO
#define REALTIME

void drawPose(cv::Mat& img, const cv::Mat& rot, float lineL)
{
	int loc[2] = {70, 70};
	int thickness = 2;
	int lineType  = 8;

	cv::Mat P = (cv::Mat_<float>(3,4) << 
		0, lineL, 0,  0,
		0, 0, -lineL, 0,
		0, 0, 0, -lineL);
	P = rot.rowRange(0,2)*P;
	P.row(0) += loc[0];
	P.row(1) += loc[1];
	cv::Point p0(P.at<float>(0,0),P.at<float>(1,0));

	line(img, p0, cv::Point(P.at<float>(0,1),P.at<float>(1,1)), cv::Scalar( 255, 0, 0 ), thickness, lineType);
	line(img, p0, cv::Point(P.at<float>(0,2),P.at<float>(1,2)), cv::Scalar( 0, 255, 0 ), thickness, lineType);
	line(img, p0, cv::Point(P.at<float>(0,3),P.at<float>(1,3)), cv::Scalar( 0, 0, 255 ), thickness, lineType);

	//printf("%f %f %f\n", rot.at<float>(0, 0), rot.at<float>(0, 1), rot.at<float>(0, 2));
	//printf("%f %f %f\n", rot.at<float>(1, 0), rot.at<float>(1, 1), rot.at<float>(1, 2));
	//printf("%f %f %f\n", rot.at<float>(2, 0), rot.at<float>(2, 1), rot.at<float>(2, 2));

	Vec3d eav;
	Mat tmp,tmp1,tmp2,tmp3,tmp4,tmp5;
	double _pm[12] = {rot.at<float>(0, 0), rot.at<float>(0, 1),rot.at<float>(0, 2), 0,
						rot.at<float>(1, 0), rot.at<float>(1, 1),rot.at<float>(1, 2),0,
						rot.at<float>(2, 0),rot.at<float>(2, 1),rot.at<float>(2, 2),0};
	decomposeProjectionMatrix(Mat(3,4,CV_64FC1,_pm),tmp,tmp1,tmp2,tmp3,tmp4,tmp5,eav);
	stringstream ss;
	ss << eav[0];
	string txt = "Pitch: " + ss.str();
	putText(img, txt,  Point(60, 20), 0.5,0.5,Scalar(255,255,255));
	stringstream ss1;
	ss1 << eav[1];
	string txt1 = "Yaw: " + ss1.str();
	putText(img, txt1,  Point(60, 40), 0.5,0.5,Scalar(255,255,255));
	stringstream ss2;
	ss2 << eav[2];
	string txt2 = "Roll: " + ss2.str();
	putText(img, txt2,  Point(60, 60), 0.5,0.5,Scalar(255,255,255));

}



cv::Point getPoint(const cv::Mat &pts,int i )
{
	int length = pts.cols;
	return cv::Point((int)pts.at<float>(0, i), (int)pts.at<float>(1, i));
}


inline float computeDist(cv::Point p1, cv::Point p2)
{
	int diff_x = p1.x - p2.x;
	int diff_y = p1.y - p2.y;
	float dist = sqrt(static_cast<float>( diff_x*diff_x + diff_y*diff_y));
	return dist;
}


int detectEyeBinkLeft( cv::Mat &demo,const cv::Mat &pts ,float thres_ = 0.24)
{
	// eye norm left p1 19 p2 22
	// eye norm left p1 25 p2 28

	// eyelid left 20 24
	// eyelid right 27 29

	cv::Point l_con1 = getPoint(pts, 19);
	cv::Point l_con2 = getPoint(pts, 22 );

	cv::Point l_eyelid1 = getPoint(pts, 20);
	cv::Point l_eyelid2 = getPoint(pts, 24);
	
	/*cv::circle(demo, l_con1, 2, Scalar(255, 255, 255), 3);
	cv::circle(demo, l_con2, 2, Scalar(255, 255, 255), 3);
	cv::circle(demo, l_eyelid1, 2, Scalar(255, 255, 255), 3);
	cv::circle(demo, l_eyelid2, 2, Scalar(255, 255, 255), 3);*/

	
	float l_dist = computeDist(l_con1, l_con2);
	float l_eyelid_dist = computeDist(l_eyelid1, l_eyelid2);
	float ratio_eye = l_eyelid_dist / l_dist;

	if (ratio_eye < thres_)
	{
		//
		std::cout << "left eye close" << std::endl;

		return 1;
	}
	std::cout << "left eye open" << std::endl;

	return 0;

}

int detectEyeBinkRight(cv::Mat &demo, const cv::Mat &pts, float thres_ = 0.24)
{
	// eye norm left p1 19 p2 22
	// eye norm left p1 25 p2 28

	// eyelid left 20 24
	// eyelid right 27 29

	cv::Point l_con1 = getPoint(pts, 25);
	cv::Point l_con2 = getPoint(pts, 28);

	cv::Point l_eyelid1 = getPoint(pts, 27);
	cv::Point l_eyelid2 = getPoint(pts, 29);

	/*cv::circle(demo, l_con1, 2, Scalar(255, 255, 255), 3);
	cv::circle(demo, l_con2, 2, Scalar(255, 255, 255), 3);
	cv::circle(demo, l_eyelid1, 2, Scalar(255, 255, 255), 3);
	cv::circle(demo, l_eyelid2, 2, Scalar(255, 255, 255), 3);*/


	float l_dist = computeDist(l_con1, l_con2);
	float l_eyelid_dist = computeDist(l_eyelid1, l_eyelid2);
	float ratio_eye = l_eyelid_dist / l_dist;

	if (ratio_eye < thres_)
	{
		//
		std::cout << "right eye close" << std::endl;

		return 1;
	}
	std::cout << "right eye open" << std::endl;

	return 0;

}
#define CYCLE_ACTIVE 24
class ActiveDetector_Shake{
public:
	float frames[CYCLE_ACTIVE];
	int idx;
	ActiveDetector_Shake(){
		idx = 0;

	}
	void moveForward(float *frames)
	{
		for (int i = 1; i <CYCLE_ACTIVE; i++)
		{
			frames[i - 1] = frames[i];
		}

	}
	void addFrame(float frame)
	{
		if (idx >= 24) {
			moveForward(frames);
			frames[CYCLE_ACTIVE - 1] = frame;
		}
		else {
			frames[idx] = frame;
		}

		idx += 1;
	}
	bool getState() {
		if (idx < 24)
			return false;
		int sum = 0;
		bool flag = 0;

		for (int i = 0; i < CYCLE_ACTIVE; i++) {
			if (frames[i] > 10 || frames[i] < -10) {
				flag = 1;
			}
			if (frames[i] > 8)
				sum++;
			else if (frames[i] < -8)
				sum--;

		}
		if (abs(sum - 0) < 8 && flag == 1)
			return true;
		else
			return false;
	}
};

class ActiveDetector_updown{
public:
	float frames[CYCLE_ACTIVE];
	int idx;
	ActiveDetector_updown(){
		idx = 0;

	}
	void moveForward(float *frames)
	{
		for (int i = 1; i <CYCLE_ACTIVE; i++)
		{
			frames[i - 1] = frames[i];
		}

	}
	void addFrame(float frame)
	{
		if (idx >= 24) {
			moveForward(frames);
			frames[CYCLE_ACTIVE - 1] = frame;
		}
		else {
			frames[idx] = frame;
		}

		idx += 1;
	}
	bool getState(bool up) {
		if (idx < 24)
			return false;
		int sum = 0;
		bool flag = 0;

		for (int i = 0; i < CYCLE_ACTIVE; i++) {
			if (frames[i] > 15 || frames[i] < -15) {
				flag = 1;
			}
			if (up) {
				if (frames[i] > 15)
					sum++;
			}
			else if (frames[i]<-15)
				sum++;
		}

		if (sum>8 && flag == 1)
			return true;
		else
			return false;
	}

};



class TextShowQ{
	
public:

	double tick_time; 

	int flag = 0;
	 string m_message;

	string m_action1;

	string m_action2;
	string text_alive; 
	int stage = 0;


	int res = -1;


	TextShowQ( std::string message, std::string action1, std::string action2)
	{

		m_message = message;
		m_action1 = action1;
		m_action2 = action2;
		text_alive = m_message;

	}

	void show(cv::Mat &frame)
	{
		if (flag == 0)
		{
			flag = 1;	
			tick_time = (double)getTickCount();

		}
		double diff = ((double)getTickCount() - tick_time) / getTickFrequency();

		if (diff > 2.0 && stage !=-2)
		{
			flag = 0;
			tick_time = (double)getTickCount();
			stage += 1;


			if (stage == 1)
			{
				text_alive = m_action1;

			}
	

			if (stage == 2 )
			{
				if (res == 1)
				text_alive = m_action2;
				else{
					text_alive = "Failed";
					stage == -2;

				}
			}
	
			if (stage == 3 && res == 1)
			{
				if (res == 1)
					text_alive = "Success";
				else{
					text_alive = "Failed";
					stage == -2;

				}
			}
		}
		cv::putText(frame, text_alive, cv::Point(15, 40), cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(255, 0, 0), 1);
	}

};
int main()
{
	char detectionModel[] = "../models/DetectionModel-v1.5.bin";
	char trackingModel[] = "../models/DetectionModel-v1.5.bin";
	string faceDetectionModel("../models/haarcascade_frontalface_alt2.xml");
	ActiveDetector_Shake shaker_detector;

	ActiveDetector_Shake Pitch_detector;
	TextShowQ tss("ready", "shake", "blink");


	// initialize a XXDescriptor object
	INTRAFACE::XXDescriptor xxd(4);
	// initialize a FaceAlignment object
	INTRAFACE::FaceAlignment fa(detectionModel, detectionModel, &xxd);
	if (!fa.Initialized()) {
		cerr << "FaceAlignment cannot be initialized." << endl;
		return -1;
	}
	// load OpenCV face detector model
	cv::CascadeClassifier face_cascade;
	if( !face_cascade.load( faceDetectionModel ) )
	{ 
		cerr << "Error loading face detection model." << endl;
		return -1; 
	}
	

#ifdef REALTIME
	// use the first camera it finds
	cv::VideoCapture cap(0); 
#endif 

#ifdef VIDEO
	string filename("../data/vid.wmv");
	cv::VideoCapture cap(filename); 
#endif

	if(!cap.isOpened())  
		return -1;

	int key = 0;
	bool isDetect = true;
	bool eof = false;
	float score, notFace = 0.3;
	cv::Mat X,X0;
	string winname("Demo Living Detection");
	cv::namedWindow(winname);

	int exceptcode[3] = { -1, 1, 2 };


	while (key!=27) // Press Esc to quit
	{

		cv::Mat frame;
		cap >> frame; // get a new frame from camera
		//frame = cv::imread("xjp4.jpg");
		
		tss.show(frame);

		if (frame.rows == 0 || frame.cols == 0)
			break;

		if (isDetect)
		{
			// face detection
			vector<cv::Rect> faces;
			face_cascade.detectMultiScale(frame, faces, 1.2, 2, 0, cv::Size(50, 50));
			// if no face found, do nothing
			if (faces.empty()) {
				cv::imshow(winname,frame);
				key = cv::waitKey(5);
				continue ;
			}
			clock_t start_time1 = clock();
			// facial feature detection on largest face found
			if (fa.Detect(frame,*max_element(faces.begin(),faces.end(),compareRect),X0,score) != INTRAFACE::IF_OK)
				break;
			clock_t finish_time1 = clock();
			double total_time = (double)(finish_time1-start_time1)/CLOCKS_PER_SEC;
			std::cout<< "detect time: " << total_time*1000 << endl;
			isDetect = false;
		}
		else
		{
			clock_t start_time1 = clock();
			// facial feature tracking
			if (fa.Track(frame,X0,X,score) != INTRAFACE::IF_OK)
				break;
			clock_t finish_time1 = clock();
			double total_time = (double)(finish_time1-start_time1)/CLOCKS_PER_SEC;
			std::cout<< "track time: " << total_time*1000 << endl;
			X0 = X;
		}
		if (score < notFace) // detected face is not reliable
			isDetect = true;
		else
		{

			bool l = detectEyeBinkLeft(frame,X0);
			bool r = detectEyeBinkRight(frame, X0);
			if (r&&l&&tss.stage==2)
			{
				tss.res = 1;

			}
			

			// plot facial landmarks
			for (int i = 0; i < X0.cols; i++){
				cv::circle(frame, cv::Point((int)X0.at<float>(0, i), (int)X0.at<float>(1, i)), 1, cv::Scalar(0, 255, 0), -1);
				char a[3];
				itoa(i, a, 10);
//cv::putText(frame, a, cv::Point((int)X0.at<float>(0, i), (int)X0.at<float>(1, i)), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0),0.1);

			}
			// head pose estimation
			INTRAFACE::HeadPose hp;
			fa.EstimateHeadPose(X0,hp);
			Pitch_detector.addFrame(hp.angles[2]);

			shaker_detector.addFrame(hp.angles[1]);
			if (shaker_detector.getState() && tss.stage == 1)
			{
				tss.res = 1;
			}

			/*if (shaker_detector.getState()){
				cv::putText(frame,"shake",cv::Point(100,100), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 0.1);
			}
		

			else if (Pitch_detector.getState()){
				cv::putText(frame, "nod", cv::Point(100, 100), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0), 0.1);
			}*/
		

	

			
			// plot head pose
			//drawPose(frame, hp.rot, 50);
		}

		printf("score %f\n", score);
		cv::imshow(winname,frame);	
		key = cv::waitKey(5);
	}

	return 0;

}






