#include "HoGFeatureExtractor.h"



HoGFeatureExtractor::HoGFeatureExtractor()
{
}
Vector<int> HoGFeatureExtractor::extract(Vector<Mat> img, int r, int c, int wind_width, int wind_height, int block_size, int stride, int cell_size, int bin_numbers)
{
	Mat I;
	int S1, S2, S3, S4;
	Vector<int> Features_Vector;
	int ind = 0;
	int cells = block_size / cell_size;
	int totalSum = 0;
	for (int i = c; i <= c + wind_width - block_size; i += stride)
	{
		for (int j = r; j <= r + wind_height - block_size; j += stride)
		{
			for (int k = 0; k < cells; k++)
			{
				for (int q = 0; q < cells; q++)
				{
					for (int b = 0; b < bin_numbers; b++)
					{
						int sum = (int)img[b].at<int>(j + k*cell_size + cell_size, i + q*cell_size + cell_size)
							+ (int)img[b].at<int>(j + k*cell_size, i + q*cell_size)
							- (int)img[b].at<int>(j + k*cell_size, i + q*cell_size + cell_size)
							- (int)img[b].at<int>(j + k*cell_size + cell_size, i + q*cell_size);
						Features_Vector.push_back(sum);
					}
				}
			}
		}
	}
	return Features_Vector;
}

HoGFeatureExtractor::~HoGFeatureExtractor()
{
}
