// Author: Philipp Werner <philipp.werner@ovgu.de>
// This code is part of IntraFace.

#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <assert.h>


// Define fixed width types
#if _MSC_VER < 1600
typedef unsigned __int8 uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
typedef __int8 int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
#else
#include <stdint.h>
#endif


// Check whether we have a big endian architecture
inline bool is_bigendian()
{
	const int i = 1;
	return (*(char*)&i) == 0;
}

// TODO: To speed up loading in big endian architectures, we may overload the function to use optimized platform specific conversion functions
template <typename T>
T swap_endian(T u)
{
    union
    {
        T u;
        unsigned char u8[sizeof(T)];
    } source, dest;

    source.u = u;

    for (size_t k = 0; k < sizeof(T); k++)
        dest.u8[k] = source.u8[sizeof(T) - k - 1];

    return dest.u;
}

// Write array of n elements to file
template <typename T>
bool write_n(FILE * file, const T * data, size_t n)
{
	if (is_bigendian()) {
		// This is not optimized for speed, however it's only big endian writing....
		bool okay = true;
		for (size_t i = 0; i < n; ++i) {
			T swapped = swap_endian(data[i]);
			okay &= fwrite(&swapped, sizeof(swapped), 1, file) == 1;
		}
		return okay;
	} else {
		return fwrite(data, sizeof(*data), n, file) == n;
	}
}

// Read array of n elements from file
template <typename T>
bool read_n(FILE * file, T * data, size_t n)
{
	if (fread(data, sizeof(*data), n, file) != n)
		return false;
	if (is_bigendian()) {
		for (size_t i = 0; i < n; ++i)
			data[i] = swap_endian(data[i]);
	}
	return true;
}

// Write one element to file
template <typename T>
bool write_one(FILE * file, T data)
{
	return write_n(file, &data, 1);
}

// Read one element from file
template <typename T>
bool read_one(FILE * file, T & data)
{
	return read_n(file, &data, 1);
}

// Write one cv::Mat to file
bool write_one(FILE * file, const cv::Mat & data)
{
	bool okay = true;
	okay &= write_one(file, int32_t(data.rows));
	okay &= write_one(file, int32_t(data.cols));
	okay &= write_one(file, uint32_t(data.type()));

	// If matrix memory is continuous, we can reshape the matrix
	int rows = data.rows, cols = data.cols;
	if (data.isContinuous()) {
		cols = rows*cols;
		rows = 1;
	}

	// Currently only supports float/double matrices!
	assert(data.depth() == CV_32F || data.depth() == CV_64F);
	if (data.depth() == CV_32F)
		for (int r = 0; r < rows; ++r)
			okay &= write_n(file, data.ptr<float>(r), cols);
	else if (data.depth() == CV_64F)
		for (int r = 0; r < rows; ++r)
			okay &= write_n(file, data.ptr<double>(r), cols);
	else
		return false;

	return okay;
}

// Read one cv::Mat from file
bool read_one(FILE * file, cv::Mat & data)
{
	bool okay = true;
	int32_t rows, cols; uint32_t type;
	okay &= read_one(file, rows);
	okay &= read_one(file, cols);
	okay &= read_one(file, type);
	if (rows <= 0 || cols <= 0 || (type & ~CV_MAT_TYPE_MASK) != 0)
		return false;
	data.create(rows, cols, type);

	// If matrix memory is continuous, we can reshape the matrix
	if (data.isContinuous()) {
		cols = rows*cols;
		rows = 1;
	}

	// Currently only supports float/double matrices!
	if (data.depth() == CV_32F)
		for (int r = 0; r < rows; ++r)
			okay &= read_n(file, data.ptr<float>(r), cols);
	else if (data.depth() == CV_64F)
		for (int r = 0; r < rows; ++r)
			okay &= read_n(file, data.ptr<double>(r), cols);
	else
		return false;

	return okay;
}


// Load binary IntraFace model file
bool load_binary_model_file(
		const char * fn,
		int & iteration, int & points,
		cv::Mat & mean_shape, cv::Mat & w, double & wb,
		std::vector<cv::Mat> & R, std::vector<cv::Mat> & b)
{
	FILE * file = fopen(fn, "rb");
	if (!file) {
		fprintf(stderr, "Error opening model file \"%s\"!\n", fn);
		return false;
	}

	bool okay = true;

	uint32_t n_iter, n_points;
	okay &= read_one(file, n_iter);
	okay &= read_one(file, n_points);
	iteration = n_iter; points = n_points;

	okay &= read_one(file, mean_shape);
	okay &= read_one(file, w);
	okay &= read_one(file, wb);

	R.resize(n_iter);
	b.resize(n_iter);
	for (uint32_t i = 0; i < n_iter; ++i) {
		okay &= read_one(file, R[i]);
		okay &= read_one(file, b[i]);
	}

	fclose(file);

	if (!okay)
		fprintf(stderr, "Error reading model file \"%s\"! It seems to be corrupted!\n", fn);

	return okay;
}

// Save binary IntraFace model file
bool save_binary_model_file(
		const char * fn,
		int iteration, int points,
		const cv::Mat & mean_shape, const cv::Mat & w, double wb,
		const std::vector<cv::Mat> & R, const std::vector<cv::Mat> & b)
{
	FILE * file = fopen(fn, "wb");
	if (!file) {
		fprintf(stderr, "Cannot open model file \"%s\" for writing!\n", fn);
		return false;
	}

	bool okay = true;

	okay &= write_one(file, uint32_t(iteration));
	okay &= write_one(file, uint32_t(points));

	okay &= write_one(file, mean_shape);
	okay &= write_one(file, w);
	okay &= write_one(file, wb);

	assert(R.size() == iteration);
	assert(b.size() == iteration);
	for (int i = 0; i < iteration; ++i) {
		okay &= write_one(file, R[i]);
		okay &= write_one(file, b[i]);
	}

	fclose(file);

	if (!okay)
		fprintf(stderr, "Error writing model file \"%s\"!\n", fn);

	return okay;
}

