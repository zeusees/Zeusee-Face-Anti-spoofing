#ifndef __FACE_ALIGNMENT_H__
#define __FACE_ALIGNMENT_H__


#include <iostream>
#include <string>
#include <vector>
#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <intraface/Marcos.h>
#include <intraface/XXDescriptor.h>
#include <intraface/binary_model_file.h>

using namespace std;

namespace INTRAFACE {
	class DLLDIR FaceAlignment {

	public:
		/// <summary>
		/// Initializes a new instance of the FaceAlignment class.
		/// </summary>
		/// <param name="detectionModel">The detection model.</param>
		/// <param name="trackingModel">The tracking model.</param>
		/// <param name="pXXD">The pointer to XXD object.</param>
		/// <param name="offset">
		/// The optional parameter, the offset between your face detector output and OpenCV's. 
		/// Say the output of your face detector is (x,y,w,h). After applying the offset,
		/// it becomes (x+offset.x, y+offset.y, w*offset.width, h*offset.height).
		/// </param>
		FaceAlignment(const char* detectionModel, const char* trackingModel, 
			const XXDescriptor* pXXD, const cv::Rect_<double>& offset=cv::Rect_<double>(0,0,1,1));
		
		inline bool Initialized() const {
			return m_init;
		}

		/// <summary>
		/// Tracks facial landmarks in the input image.
		/// </summary>
		/// <param name="image">The image (grayscale or rgb).</param>
		/// <param name="prev">The previous landmarks (2xn).</param>
		/// <param name="landmarks">The predicted landmarks (2xn).</param>
		/// <param name="score">The confidence score.</param>
		/// <returns>status</returns>
		IFRESULT Track(const cv::Mat& image, const cv::Mat& prev, cv::Mat& landmarks, float& score);

		/// <summary>
		/// Detects facial landmarks in the input image.
		/// </summary>
		/// <param name="image">The image (grayscale or rgb).</param>
		/// <param name="face">
		/// The face square (x,y,w,h). (x,y) is the upper left corner of face region. 
		/// (w,h) are the width and height of the face region.
		/// </param>
		/// <param name="landmarks">The predicted landmarks (2xn).</param>
		/// <param name="score">The confidence score.</param>
		/// <returns>status</returns>
		IFRESULT Detect(const cv::Mat& image, const cv::Rect& face, cv::Mat& landmarks, float& score);
		
		/// <summary>
		/// Estimate head pose from predicted landmarks.
		/// </summary>
		/// <param name="p2D">Landmark prediction (2xn).</param>
		/// <param name="pose">Returned head pose</param>
		/// <returns>status</returns>
		IFRESULT EstimateHeadPose(const cv::Mat& p2D, HeadPose& pose);

	protected:
		// protected member variables
		bool   m_init;
		bool   m_isDetection;
		int    m_iter;
		int    m_points;
		double  m_wb;
		double m_ratio;

		vector<cv::Mat> m_DR,m_Db,m_TR,m_Tb;
		vector<int>  m_winsize;
		vector<bool> m_flags;
		cv::Mat m_w;
		cv::Mat m_meanShape2D;
		const XXDescriptor *m_pXXD;
		cv::Rect m_cropROI;
		cv::Rect_<double> m_offset;
		RigidMotion m_rm;
		
		// protected member functions
		float Align(const cv::Mat& image, cv::Mat& X);

		void Extract(const cv::Mat& image, const cv::Mat& X, cv::Mat& Phi, int winsize, bool flag);

		void Imcrop(const cv::Mat& image, cv::Mat& output, const cv::Rect& roi, cv::Rect& outputROI);

		void NormalizeDetect(const cv::Mat& image, cv::Mat& normalizedImage, cv::Rect_<double>& face);

		void InitializeDetect(cv::Rect_<double>& face, cv::Mat& landmarks);

		void NormalizeTrack(const cv::Mat& image, cv::Mat& normalized, cv::Mat& landmarks);

		void InitializeTrack(const cv::Mat& prev, cv::Mat& landmarks);

	};
}

#endif