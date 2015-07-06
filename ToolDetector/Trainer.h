#pragma once
#include "AllIncludes.h"
#include "CommonFunctions.h"

using namespace std;
using namespace cv;

class Trainer
{
public:
	Trainer();
	Trainer(string filename,bool offline);
	void loadTrainingVideos();
	Mat CreateTrainingData_integral(String video_name,String video_short_name, String file_name, int window_width, int window_height, int block_size, int stride, int cell_size, int bins, int resize_amount);
	bool positionBox(const Mat& resized, Mat& imagePatch, int x1, int y1, int x2, int y2, bool manual = false);
	void train(CvSVM &SVMModel, string mainDir, int window_width = 32, int window_height = 32, int block_size = 16
		, int stride = 8, int cell_size = 8, int bins = 9, int percentage = 20, string offlineModelFilename = "", bool saveModel = false, bool tempMode = true);
	~Trainer();
private:
	string trainingVideosFilename;
	bool offline;
	vector<string> trainingVids;
	CvSVM SVMModel;
	string shaftDir,tipDir, openTipDir,negDir;
	CvSVMParams SVMparams;
	Mat tipSamples, shaftSamples, negativeSamples, openTipSamples, openTipTempSamples, tipNegativeTempSamples;
	Mat tipNegativeTempLabels, openTipTempLabels,AllData2, AllLabels2, AllDataFloat;

};

