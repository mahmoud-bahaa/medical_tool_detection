#pragma once
#include "AllIncludes.h"

using namespace std;
using namespace cv;

class FrameGrabber
{
public:
	FrameGrabber();
	FrameGrabber(string filename, int skipFrames = 0);
	bool isValid();
	void resizeFrame(const cv::Mat& inputFrame, double shrinkPercentage, cv::Mat& resizedFrame);
	bool setFrame(int skipFrames);
	bool readFrame(cv::Mat &frame);

	~FrameGrabber();
private:
	bool valid ;
	VideoCapture *currentVideo;
	int skipFrames ;
};

