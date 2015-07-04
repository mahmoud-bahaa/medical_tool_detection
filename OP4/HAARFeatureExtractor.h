#pragma once
#include "FeatureExtractor.h"
class HAARFeatureExtractor :
	public FeatureExtractor
{
public:
	HAARFeatureExtractor();
	Vector<int> extract(Vector<Mat> img, int r, int c, int wind_width, int wind_height, int block_size, int stride, int cell_size, int bin_numbers);
	~HAARFeatureExtractor();
};

