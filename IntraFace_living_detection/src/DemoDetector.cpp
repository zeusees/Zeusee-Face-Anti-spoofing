///////////////////////////////////////////////////////////////////////////////////////////////////////
/// DemoDetector.cpp
/// 
/// Description:
/// This program shows you how to use FaceAlignment class in detecting facial landmarks on one image. 
/// In this version,  we add head pose estimation. See DemoTracker for a demo.
///
/// There are two modes: INTERACTIVE, AUTO.
///
/// In the INTERACTIVE mode, the user is asked to create a draggable rectangle to locate one's face. 
/// To obtain good performance, the upper and lower boundaries need to exceed one's eyebrow and lip. 
/// For examples of good input rectangles, please refer to "../data/good_input_rect.jpg".
///
/// In the AUTO mode, the faces are found through OpenCV face detector.
///
/// Note that the initialization is optimized for OpenCV face detector. However, the algorithm is not
/// very sensitive to initialization. It is possible to replace OpenCV's with your own face detector. 
/// If the output of your face detector largely differs from the OpenCV's, you can add a constant offset
/// to the output of your detector using an optional parameter in the constructor of FaceAlignment.
/// See more details in "FaceAlignment.h".
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
#include <iostream>
#include <string>
#include <vector>
#include <time.h>
#include <intraface/FaceAlignment.h>
#include <intraface/XXDescriptor.h>

using namespace std;


// 2 modes: AUTO, INTERACTIVE
#define AUTO

bool drawing_box = false;
cv::Mat X;
cv::Rect box;
INTRAFACE::FaceAlignment *fa;

void draw_box(cv::Mat* img, cv::Rect rect ){
	cv::rectangle( *img, rect, cv::Scalar(0,255,0));
}

// Implement mouse callback
void my_mouse_callback( int event, int x, int y, int flags, void* param ){
	cv::Mat* image = (cv::Mat*) param;

	switch( event ){
		case CV_EVENT_MOUSEMOVE: 
			if( drawing_box ){
				box.width = x-box.x;
				box.height = y-box.y;
			}
			break;

		case CV_EVENT_LBUTTONDOWN:
			drawing_box = true;
			box = cv::Rect( x, y, 0, 0 );
			break;
		// when right button is release, face alignment is performed on the created rectangle.
		case CV_EVENT_LBUTTONUP:
			drawing_box = false;
			if( box.width < 0 ){
				box.x += box.width;
				box.width *= -1;
			}
			if( box.height < 0 ){
				box.y += box.height;
				box.height *= -1;
			}
			draw_box( image, box );
			float score;
			// detect facial landmarks
			fa->Detect(*image,box,X,score);
			// draw prediction
			for (int i = 0 ; i < X.cols ; i++)
				cv::circle(*image,cv::Point((int)X.at<float>(0,i), (int)X.at<float>(1,i)), 1, cv::Scalar(0,255,0), -1);

			break;
	}
}


int main()
{
	char detectionModel[] = "../models/DetectionModel-v1.5.bin";
	char trackingModel[] = "../models/TrackingModel-v1.10.bin";
	INTRAFACE::XXDescriptor xxd(4);

	fa = new INTRAFACE::FaceAlignment(detectionModel, trackingModel, &xxd);

	if (!fa->Initialized()) {
		cerr << "FaceAlignment cannot be initialized." << endl;
		return -1;
	}
	// read image
	string filename("../data/pic.jpg");
	cv::Mat frame  = cv::imread(filename);
	// create a window
	string winname("Demo IntraFace Detector");
	cv::namedWindow(winname);
	

#ifdef INTERACTIVE
	
	// Set up the callback
	cvSetMouseCallback( winname.c_str(), my_mouse_callback, (void*) &frame);
	
	while( true ) {
		cv::Mat image = frame.clone();
		if( drawing_box ) 
			draw_box(&image, box );
		cv::imshow(winname,image);
		// press Esc to quit
		if( cv::waitKey( 30 )==27 ) 
			break;
	}

#endif

#ifdef AUTO
	// load OpenCV face detector model
	string faceDetectionModel("../models/haarcascade_frontalface_alt2.xml");
	cv::CascadeClassifier face_cascade;
	if( !face_cascade.load( faceDetectionModel ) )
	{ 
		cerr << "Error loading face detection model." << endl;
		return -1; 
	}
	vector<cv::Rect> faces;
	float score, notFace = 0.5;
	// face detection
	face_cascade.detectMultiScale(frame, faces, 1.2, 2, 0, cv::Size(50, 50));
	
	INTRAFACE::HeadPose pose;

	for (int i = 0 ;i < faces.size(); i++) {
		// face alignment
		if (fa->Detect(frame,faces[i],X,score) == INTRAFACE::IF_OK)
		{
			// only draw valid faces
			if (score >= notFace) {
				cout << faces[i] << endl;
				for (int i = 0 ; i < X.cols ; i++)
					cv::circle(frame,cv::Point((int)X.at<float>(0,i), (int)X.at<float>(1,i)), 1, cv::Scalar(0,255,0), -1);
			}
		}
	}
	cv::imshow(winname,frame);
	cv::waitKey(0); // press any key to quit
#endif

	delete fa;

	return 0;

}

