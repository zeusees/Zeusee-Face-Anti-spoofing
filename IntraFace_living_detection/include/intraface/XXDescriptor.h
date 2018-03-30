#ifndef __XX_DESCRIPTOR_H__
#define __XX_DESCRIPTOR_H__

#include <opencv2/core/core_c.h>
#include <opencv2/core/core.hpp>
#include <math.h>
#include "Marcos.h"

namespace INTRAFACE {


class DLLDIR XXDescriptor {

public: 
	/// <summary>
	/// Initializes a new instance of XXDescriptor class.
	/// </summary>
	/// <param name="nsb">The number of spatial bins.</param>
	XXDescriptor(int nsb) : 
	  m_nsb(nsb)
	  {
		  m_nob = 8; // number of orientation bins fixed
		  m_dimPerPoint = nsb*nsb*8;
	  }

	/// <summary>
	/// Computes image descriptors for the input image centered at each landmark location.
	/// </summary>
	/// <param name="image">The image (double grayscale).</param>
	/// <param name="landmark">The landmark(2xn).</param>
	/// <param name="output">The output (float).</param>
	/// <param name="winsize">The patch size.</param>
	void Compute(const cv::Mat& image, const cv::Mat& landmark, cv::Mat& output, int winsize=32) const;



private:

	int m_nob;
	int m_nsb;
	int m_dimPerPoint;

	void Process(const double *im, float *output, int pt_ind, int dim, int winsize) const;

};




}


#endif