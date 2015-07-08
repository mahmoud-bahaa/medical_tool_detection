#include <cmath>
#include <ctype.h>
//#include <thread>



#include "CommonFunctions.h"
#include "RANSAC1.h"
#include "FrameGrabber.h"
#include "Trainer.h"
#include "HoGFeatureExtractor.h"
#include "HAARFeatureExtractor.h"
/*
#include <vl/generic.h>
#include <vl/hog.h>
#include <vl/pgm.h>
*/

using namespace std;
using namespace cv;





void CannyThreshold(Mat src_gray, Mat src, Mat dst)
{
	Mat detected_edges;

	int edgeThresh = 1;
	int lowThreshold;
	int const max_lowThreshold = 100;
	int ratio = 3;
	int kernel_size = 3;
	char* window_name = "Edge Map";

	/// Reduce noise with a kernel 3x3
	blur(src_gray, detected_edges, Size(3, 3));

	/// Canny detector
	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

	/// Using Canny's output as a mask, we display our result
	dst = Scalar::all(0);

	src.copyTo(dst, detected_edges);
	imshow(window_name, dst);
}

Point2d motionEstimation(Mat block, Mat searchWin, int step, double &returnedError) {
	double minRMS = 10000.0;
	Mat greyBlock,greySearchWin,thresholdedBlock,thresholdedSearchWin;
	cvtColor(block, greyBlock, CV_BGR2GRAY);
	cvtColor(searchWin, greySearchWin, CV_BGR2GRAY);
	threshold( greyBlock, thresholdedBlock, 127, 255, CV_THRESH_BINARY_INV);
	threshold(greySearchWin,thresholdedSearchWin , 127, 255, CV_THRESH_BINARY_INV);
	//imshow("Thresholded", thresholdedBlock);
	greyBlock = thresholdedBlock;
	greySearchWin = thresholdedSearchWin;
	
	//CannyThreshold(greyBlock, block, edgeBlock);
	//imshow("Grey block", greyBlock);
	//imshow("Grey window", greySearchWin);
	//cvWaitKey(0);
	Point2d motionVec;
	for (int i = 0; i < (greySearchWin.cols - greyBlock.cols) / step; i+=step) {
		for (int j = 0; j < (greySearchWin.rows - greyBlock.rows) / step; j+=step){
			double rms = 0.0;
			for (int k = 0; k < greyBlock.cols; k++) {
				for (int l = 0; l < greyBlock.rows; l++) {
					rms += pow((double)greyBlock.at<uchar>(l,k) - greySearchWin.at<uchar>(j + l,i + k), 2);
					//cout << "RMS:" << greyBlock.at<uchar>(k, l) - greySearchWin.at<uchar>(i + k, j + l) << endl;
					//cout << "Block(" << k << "," << l << ") == Window(" << i+k << "," << k+l << ") " << endl;
				}	
			}

			/*
			Rect im_roi(i, j, greyBlock.cols, greyBlock.rows);
			Mat image_roi = greySearchWin(im_roi);
			subtract(image)
			*/
			rms = rms / (greyBlock.rows*greyBlock.cols);
			rms = sqrt(rms);
			if (rms < minRMS)
			{
				minRMS = rms;
				motionVec.x = i;
				motionVec.y = j;
			}
		}
	}
	returnedError = minRMS;
	return motionVec;
}

int main(int argc, char** argv)

{
	vector<string> trainingVids;
	//char* fileName = "annotations\\Learning Samples\\video 2.avi";
	char* testVideoFileName = "annotations\\Learning Samples\\video 2.avi";
	string mainDir = "annotations";

	int percentage = 20; // percentage is 20 /100 
	int bins = 9;
	int window_width = 32;
	int window_height = 32;
	int block_size = 16;
	int stride = 8;
	int cell_size = 8;
	bool tracking = false;
	Mat sampleMat2(1, 324, CV_32FC1);
	CvSVM SVMModel;
	Mat M = Myresize(imread("2.png"), percentage);
	vector<Mat> channel;
	split(M, channel);
	Mat GreenCH = channel[1];
	Mat resizedFrame = M.clone();
	Mat debugFrame , debugFrameTip, debugFrameShaft;
	Mat output2(GreenCH.rows, GreenCH.cols, CV_8UC1, Scalar(0));
	Rect estimatedRegion;
	Rect searchWindow;
	CvSVMParams SVMparams;
	string pos, neg;
	Vector<Point> shaftPointList;
	vector<Point> tipPointList;
	Point lastTipPosition(0,0),toolPoint(0,0);
	
	FrameGrabber frameGrabber(testVideoFileName);

	/*
	VideoCapture currentVideo(testVideoFileName); // open the video file for reading
	*/
	int frame_num = 1;
	if (!frameGrabber.isValid())  // if not success, exit program
	{
		cout << "Cannot open video file " << endl;
		return -1;
	}
	bool loadData = true;
	//Initialize the Trainer
	Trainer trainer("training_videos.txt",loadData);

	//Train data 
	trainer.train(SVMModel, mainDir,window_width,window_height,block_size,stride, cell_size,bins,percentage,"training_model.xml",true,false);
		
	//////////////////////////////////////////////////////////////////////////
	double t = (double)getTickCount();
	//int skipFrames = 13500;
	int skipFrames = 4000;
	//int skipFrames = 0;
	//int i = 9207;
	double avg = 0;
	int tk = 0;
	int tipToShaftDist = 30;
	double detectionError = 0.0;
	int evaluationSampleSize = 0;
	Mat f;
	Rect Box;
	//Rect R(window_width / 2, window_height / 2, resizedFrame.cols - 2 * win_size, resizedFrame.rows - 2 * win_size);
	//Changed to center Roi
	Rect R(window_width / 2, window_height / 2, resizedFrame.cols - window_width, resizedFrame.rows - window_height);
	Box = R;
	Mat prevgray;
	Mat imgROI;
	int rectMargin = 70;
	bool potentialFalsePositive = false;
	frameGrabber.setFrame(skipFrames);
	HoGFeatureExtractor featureExtractor;

	for (;;)
	{

		t = (double)getTickCount();

		//bool bSuccess = currentVideo.read(f);

		if (!frameGrabber.readFrame(f)) //if not success, break loop
		{
			cout << "Cannot read the frame from video file" << endl;
			continue;
		}

		//currentVideo.set(CV_CAP_PROP_POS_FRAMES, skipFrames++);
		frameGrabber.resizeFrame(f, percentage, resizedFrame);
		debugFrame = resizedFrame.clone();
		debugFrameTip = resizedFrame.clone();
		debugFrameShaft = resizedFrame.clone();
		
		//resizedFrame = Myresize(f, percentage);

			bool invalidAccess = false;
		
			Vector<Mat> V = featureExtractor.HoG_integral_images(resizedFrame, bins);

			for (int i = Box.y; i < Box.height + Box.y; i += 3)
			{
				if (i >= resizedFrame.rows - window_height / 2)
					continue;
				for (int j = Box.x; j < Box.width + Box.x; j += 3)
				{
					Vector<int> HOGFeatures;
					
					HOGFeatures = featureExtractor.extract(V, max(0,i - window_height / 2), max(0,j - window_width / 2), window_width, window_height, block_size, stride, cell_size, bins);
					
					
					for (int w = 0; w < HOGFeatures.size(); w++)
					{
						sampleMat2.at<float>(0, w) = HOGFeatures[w];

					}
					Mat SD;
					sampleMat2.convertTo(SD, CV_32FC1);

					int response = SVMModel.predict(sampleMat2);

					output2.at<uchar>(i, j) = 255;
					Point P;
					P.x = i;
					P.y = j;
					Point2f p2f;
					p2f.x = i;
					p2f.y = j;

					switch (response)
					{
					case SHAFT:
						shaftPointList.push_back(P);
						circle(debugFrameShaft, Point(j, i), 2, Scalar(255,255,255), 1);
						break;
					case OPENED_TIP:
						tipPointList.push_back(P);
						circle(debugFrameTip, Point(j, i), 2, cv::Scalar(255, 255, 255), 1);
						break;
					case CLOSED_TIP:
						tipPointList.push_back(P);
						circle(debugFrameTip, Point(j, i), 2, cv::Scalar(255, 255, 255), 1);
						break;

					}
				}
				if(invalidAccess)
					break;

			}
			if(invalidAccess)
				continue;

			t = ((double)getTickCount() - t) / getTickFrequency();
			tk++;
			avg += (double)1 / t;
			if (tk % 100 == 0)
				cout << skipFrames << "::" << "SVM finished Testing ..." << t << "s ... rate :" << (double)1 / t << " avg: " << avg / tk << endl;


			std::vector<KeyPoint> shaftBlobs;
			Vec4i shaftLine;
			Point shaftClosestPoint;
			std::vector<KeyPoint> clusterBlobs ;
			Point clusterCenterPoint ;
			
			shaftBlobs = getShaftCenter(debugFrameShaft,100);
			for(int l = 0; l < shaftBlobs.size() ; l++)
			{
				Vector <Point> filteredShaftPointList;
				for(int i = 0;i < shaftPointList.size() ; i++)
				{
					if(computeRMS(shaftPointList[i],Point(shaftBlobs.at(l).pt.y,shaftBlobs.at(l).pt.x)) < tipToShaftDist)
					{
						filteredShaftPointList.push_back(shaftPointList[i]);
					}
				}
				//Fit a line to the shaft using RANSAC
				shaftLine = lineFit(filteredShaftPointList,  filteredShaftPointList.size(),rectMargin * 2, 3, 500, 45.0F, 20.0F);
				cv::line(resizedFrame, Point(shaftLine[1], shaftLine[0]), Point(shaftLine[3], shaftLine[2]), Scalar(0, 0, 255), 4);

				//Use the estiamted line to look for the nearby tip
				//Try to detect if the tool was opened or closed
				int tipSampleSize = std::max(tipPointList.size(), tipPointList.size());
				shaftClosestPoint = Point(shaftLine[0],shaftLine[1]);

				Vector<Point> mostLikelyCluster;
				int initArea = 50;
				
				clusterBlobs = getTooltipCenter(debugFrameTip,initArea);
				if(clusterBlobs.size()  == 0)
				{
					//initArea -= 5;
					break;
				}
				else if(clusterBlobs.size() == 1)
				{
					clusterCenterPoint = Point(clusterBlobs.at(0).pt.y,clusterBlobs.at(0).pt.x);
					break;
				}
				else
				{
					double diff  = computeRMS(shaftClosestPoint, Point(clusterBlobs.at(0).pt.y,clusterBlobs.at(0).pt.x));
					diff = std::min((double)tipToShaftDist,diff);
					for(int i = 0 ; i < clusterBlobs.size() ; i++)
					{
						if(computeRMS(shaftClosestPoint, Point(clusterBlobs.at(i).pt.y,clusterBlobs.at(i).pt.x))<diff)
						{
							clusterCenterPoint = Point(clusterBlobs.at(i).pt.y,clusterBlobs.at(i).pt.x);
							diff = computeRMS(shaftClosestPoint, Point(clusterBlobs.at(i).pt.y,clusterBlobs.at(i).pt.x));
						}
					}
				}
			}
			
			
			Point roiCenter(max(shaftLine[1], shaftLine[3]), max(shaftLine[0], shaftLine[2]));
			Rect ROI(roiCenter.x - rectMargin, roiCenter.y - rectMargin, rectMargin * 1.5, rectMargin * 1.5);
			//	Rect ROI(min(shaftLine[1], shaftLine[3]) - rectMargin / 2, min(shaftLine[0], shaftLine[2]) - rectMargin / 2, abs(shaftLine[1] - shaftLine[3]) + rectMargin, abs(shaftLine[0] - shaftLine[2]) + rectMargin);
			//	cluster(shaftPointList, 4, 10, outputClusters, cv::Point(ROI.x, ROI.y), cv::Point(ROI.x + ROI.width, ROI.y + ROI.height));

			if (shaftLine[0] == 0 || clusterBlobs.size() == 0 || shaftBlobs.size() == 0 || computeRMS(shaftClosestPoint, clusterCenterPoint) > tipToShaftDist)
			{
				Box = R;
				tracking = false;
				potentialFalsePositive = true;
				//continue;
			}
			else
			{
				if (ROI.y < window_height / 2)
					ROI.y = window_height / 2;
				if (ROI.x < window_width / 2)
					ROI.x = window_width / 2;
				if (ROI.y >= resizedFrame.rows - ROI.height - 1)
					ROI.y = resizedFrame.rows - ROI.height - 100;
				if (ROI.x >= resizedFrame.cols - ROI.width - 1)
					ROI.x = resizedFrame.cols - ROI.width - 60;

				Box = ROI;
				//tracking = true;
				imgROI = resizedFrame(Rect(ROI.x + rectMargin / 2, ROI.y + rectMargin / 2, ROI.width - rectMargin, ROI.height - rectMargin));
				estimatedRegion = Rect(ROI.x + rectMargin / 2, ROI.y + rectMargin / 2, ROI.width - rectMargin, ROI.height - rectMargin);
				//imshow("ROI", imgROI);

			}
			

			if(!potentialFalsePositive)
			{
				toolPoint = Point((clusterCenterPoint.x + shaftClosestPoint.x) / 2, (clusterCenterPoint.y + shaftClosestPoint.y) / 2);
				Point refPoint = toolPoint;
				Mat imagePatch;

				if(EVALUATION)
				{

					bool result = trainer.positionBox(resizedFrame, imagePatch, refPoint.y - 10, refPoint.x - 10, refPoint.y + 10, refPoint.x + 10, true);
					if (!result)
						break;
					detectionError += (pow((double)(refPoint.x - toolPoint.x), 2)+ ((refPoint.y - toolPoint.y), 2));
					evaluationSampleSize++;
					lastTipPosition = toolPoint;
					
					
					
				}
				circle(debugFrame, Point(clusterCenterPoint.y, clusterCenterPoint.x), 2, cv::Scalar(0, 0, 255), 1);
				circle(debugFrame, Point(shaftClosestPoint.y, shaftClosestPoint.x), 2, cv::Scalar(255, 0, 0), 1);
				circle(resizedFrame, Point(toolPoint.y, toolPoint.x), 5, cv::Scalar(0, 255, 0), 1);
			//Point toolPoint(shaftClosestPoint.x, shaftClosestPoint.y);
			//Point toolPoint(clusterCenterPoint.x,clusterCenterPoint.y );
			/*
			if (lastTipPosition.x != 0 && lastTipPosition.y != 0)
			{
				double rms = computeRMS(lastTipPosition, toolPoint);
				if ( rms > 10)
				{
					
					toolPoint += (10 / rms)*lastTipPosition ;
					toolPoint.x /=2;
					toolPoint.y /=2;
				}
				
				
			}*/
			
			}
			
			//cv::line(debugFrame, Point(P2[1], P2[0]), Point(P2[3], P2[2]), Scalar(0, 0, 255), 4);
			rectangle(debugFrame, Box, Scalar(0, 0, 255), 3);
			//rectangle (debugFrame,R,Scalar(0,0,255),3);
			//rectangle (debugFrame,ROI,Scalar(0,0,255),3);
			potentialFalsePositive = false;

			
		
		shaftPointList.clear();
		tipPointList.clear();
		
		imshow("Instrument detection", resizedFrame);
		//imshow("Instrument detection", debugFrame);
		waitKey(27);
		//waitKey(0);

	}
	cout << "Evaluation sample: " << evaluationSampleSize << endl;
	cout << "Detection error: " << sqrt(detectionError / evaluationSampleSize) << endl;
	//waitKey(0);
	return 0;
}

