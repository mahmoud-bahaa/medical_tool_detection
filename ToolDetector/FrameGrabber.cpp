
#include "FrameGrabber.h"
#include "CommonFunctions.h"

using namespace std;
using namespace cv;

FrameGrabber::FrameGrabber()
{
	skipFrames = 0;
	valid = false;
}
FrameGrabber::FrameGrabber(string filename, int skipFrames)
{
	currentVideo = new VideoCapture(filename);
	currentVideo->open(filename); // open the video file for reading

	if (!currentVideo->isOpened())  // if not success, exit program
	{
		cout << "Cannot open the video file S1" << endl;
		valid = false;
	}
	else {
		valid = true;
	}

	this->skipFrames = skipFrames;



}
bool FrameGrabber::isValid()
{
	return valid;
}
bool FrameGrabber::setFrame(int skipFrames)
{
	this->skipFrames = skipFrames;
	return currentVideo->set(CV_CAP_PROP_POS_FRAMES, skipFrames);

}
void FrameGrabber::resizeFrame(const cv::Mat& inputFrame, double shrinkPercentage, cv::Mat& resizedFrame)
{
	resizedFrame = Myresize(inputFrame, shrinkPercentage);
}
bool FrameGrabber::readFrame(cv::Mat &frame )
{
	
	bool bSuccess = currentVideo->read(frame);
	if (!bSuccess) //if not success, break loop
	{
		cout << "Cannot read the frame from video file" << endl;
		return false;
	}
	setFrame(skipFrames);
	skipFrames++;
		
	return true;
}

FrameGrabber::~FrameGrabber()
{
	delete currentVideo;
}
