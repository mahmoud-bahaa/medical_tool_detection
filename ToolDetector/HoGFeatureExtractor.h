#pragma once
#include "FeatureExtractor.h"
class HoGFeatureExtractor :
	public FeatureExtractor
{
public:
	HoGFeatureExtractor();
	/*
	* Description: Computes the a set of values representing the features for each block in the window
	* @param img: the orientation map of the image
	* @param r: row offset in the image
	* @param c: column offset in the image
	* @param wind_width,wind_height: the window dimensions (the image size)
	* @param block_size: the block dimensions
	* @param stride: the shift distance in the window for the block
	* @param cell_size: the cell dimensions
	* @param bin_numbers: number of angles used in the orientation map
	* returns a vector of values, each represents the integral sum of that cell at a specific angle
	*/
	Vector<int> extract(Vector<Mat> img, int r, int c, int wind_width, int wind_height, int block_size, int stride, int cell_size, int bin_numbers) ;
	~HoGFeatureExtractor();
};

