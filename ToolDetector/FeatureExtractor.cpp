#include "FeatureExtractor.h"



FeatureExtractor::FeatureExtractor()
{
}
/*
* Description: Computes a number of images based on "nbins", each bin represents an angle.
* @param mag: magnitude values of the image
* @param ori: orientation values of the image
* @param nbins: number of angles to represent the image
* @param thresh: the magnitued values above the threshold are only considered in the computation.
* returns a vector of matrices, the vector size is the "nbins" and the matrix represents the magnitude values at this angle
*/
Vector<Mat> FeatureExtractor::orientationMap(const cv::Mat& mag, const cv::Mat& ori, int nbins, double thresh)
{
	Vector<Mat> V;
	for (int i = 0;i<nbins; i++)
	{
		Mat temp(mag.rows, mag.cols, CV_8UC1, Scalar(0));
		V.push_back(temp);
	}

	for (int i = 0; i< mag.rows; i++)
		for (int j = 0; j < mag.cols; j++)
		{
			float magPixel = mag.at<float>(i, j);
			if (magPixel > thresh)
			{
				float oriPixel = ori.at<float>(i, j);
				float bin = cvRound((oriPixel / 360)* (nbins - 1));
				//cout << bin << endl;
				V[bin].at<uchar>(i, j) = cvRound(magPixel);
				//cout << bin << ":"<< V[bin].at<int>(i,j)<< endl;
			}
		}

	return V;
}

Vector<Mat> FeatureExtractor::HoG_integral_images(Mat img, int bins)
{
	Vector<Mat> V;

	cvtColor(img, img, CV_BGR2GRAY);
	//imshow("img", img);
	Mat Sx;
	Sobel(img, Sx, CV_32F, 1, 0, 3);

	Mat Sy;
	Sobel(img, Sy, CV_32F, 0, 1, 3);

	Mat mag, ori;
	magnitude(Sx, Sy, mag);
	phase(Sx, Sy, ori, true);
	Vector<Mat> oriMap = orientationMap(mag, ori, bins, 1.0);
	Vector<Mat> IntegImage;
	for (int i = 0; i< bins; i++)
	{
		Mat m;
		integral(oriMap[i], m, CV_32SC1);
		IntegImage.push_back(m);
	}
	return IntegImage;
}

FeatureExtractor::~FeatureExtractor()
{
}
