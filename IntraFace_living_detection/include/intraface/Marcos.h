#ifndef __MARCROS_H__
#define __MARCROS_H__

//#define DLLDIR_EX

#ifdef DLLDIR_EX
   #define DLLDIR  __declspec(dllexport)   // export DLL information
#else
   #define DLLDIR  __declspec(dllimport)   // import DLL information
#endif 

namespace INTRAFACE {

	typedef struct {
		cv::Mat R;
		cv::Mat T;
		cv::Mat S;
	} RigidMotion;

	typedef struct {
		cv::Mat rot;
		// angles[0] - roll,
		// angles[1] - yaw, 
		// angles[2] - pitch
		float angles[3]; 
	} HeadPose;

	enum IFRESULT 
	{
		IF_OK = 0,
		IF_INVALID_INPUT = -1,
		IF_INVALID_IMAGE_FORMAT = -2,
		IF_INVALID_IMAGE = -3,
		IF_MODEL_NOT_INITIALIZED = -4
	};

	
	#define eps 0.000001f
	#define PI 3.1415926f
	#define PIBY2 1.5707963f
	#define RAD_TO_DEG_FACTOR 57.2957795
	#define DEG_TO_RAD_FACTOR 0.01745329


	#define max(a,b) (((a)>(b))?(a):(b))
	#define fast_floor(x) (int)( x - ((x>=0)?0:1) )

}


#endif