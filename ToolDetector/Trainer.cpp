#include "Trainer.h"
#include "HoGFeatureExtractor.h"
#include "HAARFeatureExtractor.h"



Trainer::Trainer()
{

}
Trainer::Trainer(string filename, bool offline)
{
	this->offline = offline;
	trainingVideosFilename = filename;
}
void Trainer::loadTrainingVideos()
{
	readFromTxt(trainingVideosFilename, trainingVids);
}
bool Trainer::positionBox(const Mat& resized, Mat& imagePatch, int x1, int y1, int x2, int y2, bool manual )
{
	if (!manual)
	{
		imagePatch = resized.rowRange(y1, y2).colRange(x1, x2);
		return true;
	}
	while (true)
	{
		imagePatch = resized.rowRange(y1, y2).colRange(x1, x2);
		cv::imshow("patch", imagePatch);

		int key = cvWaitKey();
		//cout << key << endl;
		switch (key)
		{
		case 13: //enter
			return true;
		case 27: //n
			return false;
		case 2490368: // up
			y1--;
			y2--;
			break;
		case 2621440: // down
			y1++;
			y2++;
			break;
		case 2424832: // left
			x1--;
			x2--;
			break;
		case 2555904: // right
			x1++;
			x2++;
			break;
		}
	}
}
Mat Trainer::CreateTrainingData_integral(String video_name, String video_short_name,String file_name, int window_width, int window_height, int block_size, int stride, int cell_size, int bins, int resize_amount)
{
	
	VideoCapture cap(video_name);
	vector<int> strs = readMultipleFiles(file_name);
	Mat R = cv::Mat(strs.size(), FN, CV_32SC1, Scalar(0));
	vector<vector<float>> returnMatrix;
	int numberOfSamples = strs.size();
	int j = 0;
	for (int i = 0;i<strs.size();i++)
	{
		
		cap.set(CV_CAP_PROP_POS_FRAMES, strs[i]);
		Mat frame;
		bool bSuccess = cap.read(frame); // read a new frame from video		
		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read the frame from video file" << endl;
		}

		std::string groundTruthPath = reformPath(std::string(file_name));
		std::string groundTruthFile = groundTruthPath + to_string((long double)strs[i]) + ".txt";

		std::string line;
		ifstream myfile(groundTruthFile);
		if (myfile.is_open())
		{
			while (getline(myfile, line))
			{
				//cout << line << '\n';
			}
			myfile.close();
		}

		else cout << "Unable to open file";
		vector<int> p = parseString(line, ";");
		int x1 = p.at(0);
		int y1 = p.at(1);
		int x2 = p.at(2);
		int y2 = p.at(3);
		///////////////////// for resized images /////////////////////////
		Mat resized = Myresize(frame, resize_amount);
		int x = cvRound((double)x2*resize_amount / 100);
		int y = cvRound((double)y2*resize_amount / 100);
		int p1_x = x - window_width / 2;
		int p1_y = y - window_width / 2;
		int p2_x = p1_x + window_height;
		int p2_y = p1_y + window_height;
		if (p1_x <0 || p1_y <0 || p2_x >= resized.cols || p2_y >= resized.rows)
			continue;
		//////////////////////////////////////////////////////////////////
		Mat imagePatch;
		bool consider = positionBox(resized, imagePatch, p1_x, p1_y, p2_x, p2_y,TRAINING);
		if (!consider)
		{
			//cout << "rejected" << endl;
			numberOfSamples--;
			continue;
		}
		//cout << strs.at(i) << " - " << "accepted" << endl;
		//Write sample to folder
		double cx = (p1_x + window_width / 2) * (100 / resize_amount);
		double cy = (p1_y + window_height / 2) * (100 / resize_amount);
		ofstream sampleOut(video_name+"\\..\\..\\..\\"+file_name+"\\..\\"+to_string((long double)strs.at(i))+".txt");
		sampleOut << to_string((long double)strs.at(i)) << ";w;2;" << cx << ";" << cy << ";" << cx << ";" << cy << ";-1;-1;-1;-1;whatever;"<<video_short_name;
		sampleOut.close();

		HoGFeatureExtractor extractor;
		Vector<Mat> V = extractor.HoG_integral_images(imagePatch, bins);
		Vector<int> f = ExtractHOGF_ver2(V, 0, 0, window_width, window_height, block_size, stride, cell_size, bins);
		vector<float> tempVec;
		for (int k = 0;k<FN;k++)
		{
			R.at<int>(j, k) = f[k];
			//tempVec.push_back(f[k]);
		}
		returnMatrix.push_back(tempVec);
		j++;
	}
	R.resize(j);
	cout << "# of samples:" << numberOfSamples << endl;
	return R;
}
void Trainer::train(CvSVM &SVMModel,string mainDir, int window_width, int window_height, int block_size, int stride, int cell_size, int bins, int percentage, string offlineModelFilename,bool saveModel,bool tempMode)
{
	if (offline)
	{
		SVMModel.load(offlineModelFilename.c_str());
		return;
	}

	loadTrainingVideos();
	cout << "Training ..." << endl;
	//string trainingVid = "annotations\\Learning Samples\\video 1.avi";
	for (int i = 0; i < trainingVids.size(); i++)
	{
		string trainingVid = mainDir + "\\Learning Samples\\" + trainingVids.at(i) + ".avi";
		int win_size = 32;
		string temp = (tempMode) ? "temp" : "enhanced";
		tipDir = mainDir + "\\DataSetSVM_positive\\Tool_" + trainingVids.at(i) + "\\"+temp;
		openTipDir = mainDir + "\\DataSetSVM_tipOpen\\Tool_" + trainingVids.at(i) + "\\"+temp;
		negDir = mainDir + "\\DataSetSVM_negative\\Tool_" + trainingVids.at(i) + "\\"+temp;
		shaftDir = mainDir + "\\DataSetSVM_shaft\\Tool_" + trainingVids.at(i) + "\\"+temp;

		
		SVMparams.svm_type = CvSVM::C_SVC;
		//SVMparams.kernel_type = CvSVM::RBF;
		SVMparams.kernel_type = CvSVM::LINEAR;
		//SVMparams.svm_type = CvSVM::EPS_SVR;
		SVMparams.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 300, 1e-6);
		

		/*
		SVMparams.svm_type = CvSVM::EPS_SVR;
		SVMparams.kernel_type = CvSVM::RBF;
		SVMparams.term_crit = TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);
		SVMparams.p = 0.1;

		double gammax = 0.8806162477592124;
		double cx = 0.5654417165992166;
		SVMparams.gamma = gammax;
		
		
		svm_param.nu = 0.5; 
		SVMparams.p = 0;
		SVMparams.C = cx;
		SVMparams.term_crit.epsilon = 0.001;
		SVMparams.term_crit.max_iter = 50;
		SVMparams.term_crit.type = CV_TERMCRIT_ITER | CV_TERMCRIT_EPS;
		*/

		//output2 = Mat::zeros(resizedFrame.rows, resizedFrame.cols, CV_8UC1);

		/////////////////////////////// training /////////////////////////////////
		//percentage = 100;
		String video_short_name = trainingVids.at(i) + ".avi";
		cout << i+1 << "- " << trainingVids.at(i) << endl;
		cout << "Closed Tip samples..." << endl;
		tipSamples = CreateTrainingData_integral(trainingVid,video_short_name, tipDir, window_width, window_height, block_size, stride, cell_size, bins, percentage);
		cout << "Opened Tip samples..." << endl;
		openTipSamples = CreateTrainingData_integral(trainingVid, video_short_name, openTipDir, window_width, window_height, block_size, stride, cell_size, bins, percentage);
		cout << "Shaft samples..." << endl;
		shaftSamples = CreateTrainingData_integral(trainingVid, video_short_name, shaftDir, window_width, window_height, block_size, stride, cell_size, bins, percentage);
		cout << "Negative samples..." << endl;
		negativeSamples = CreateTrainingData_integral(trainingVid, video_short_name, negDir, window_width, window_height, block_size, stride, cell_size, bins, percentage);


		Mat tipLabels = CLOSED_TIP * Mat::ones(tipSamples.rows, 1, CV_32FC1);
		Mat openTipLabels = OPENED_TIP * Mat::ones(openTipSamples.rows, 1, CV_32FC1);
		Mat shaftLabels = SHAFT * Mat::ones(shaftSamples.rows, 1, CV_32FC1);
		Mat negativeLabels = Mat::zeros(negativeSamples.rows, 1, CV_32FC1);
		
		vconcat(tipSamples, negativeSamples, tipNegativeTempSamples);
		vconcat(tipLabels, negativeLabels, tipNegativeTempLabels);
		vconcat(tipNegativeTempSamples, shaftSamples, openTipTempSamples);
		vconcat(tipNegativeTempLabels, shaftLabels, openTipTempLabels);
		vconcat(openTipTempSamples, shaftSamples, AllData2);
		vconcat(openTipTempLabels, shaftLabels, AllLabels2);

		// Train the SVM
		AllData2.convertTo(AllDataFloat, CV_32FC1);
		//cout << " Start Learning.... " << endl;
		SVMModel.train(AllDataFloat, AllLabels2, Mat(), Mat(), SVMparams);
		cout << "---------------------------------" << endl;
	}

	if (saveModel)
	{
		SVMModel.save(offlineModelFilename.c_str());
	}
}

Trainer::~Trainer()
{
}
