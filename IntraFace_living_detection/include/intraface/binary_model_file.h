#ifndef __BINARY_IO__
#define __BINARY_IO__

#include <vector>
#include <opencv2/core/core.hpp>

bool load_binary_model_file(const char * fn, int & iteration, int & points, cv::Mat & mean_shape, 
	cv::Mat & w, double & wb, std::vector<cv::Mat> & R, std::vector<cv::Mat> & b);

bool save_binary_model_file(const char * fn, int iteration, int points, const cv::Mat & mean_shape, 
	const cv::Mat & w, double wb, const std::vector<cv::Mat> & R, const std::vector<cv::Mat> & b);


#endif