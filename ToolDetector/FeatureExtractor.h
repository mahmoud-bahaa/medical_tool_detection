#pragma once
#include "AllIncludes.h"
using namespace std;
using namespace cv;

class FeatureExtractor
{
public:
	FeatureExtractor();
	Vector<Mat> orientationMap(const cv::Mat& mag, const cv::Mat& ori, int nbins, double thresh = 1.0);
	Vector<Mat> HoG_integral_images(Mat img, int bins);
	virtual Vector<int> extract(Vector<Mat> img, int r, int c, int wind_width, int wind_height, int block_size, int stride, int cell_size, int bin_numbers)=0;
	~FeatureExtractor();
};

