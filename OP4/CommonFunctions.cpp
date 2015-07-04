#include "CommonFunctions.h"
#include <math.h>
void readFromTxt(string filename, vector<string> &trainingVids)
{
	ifstream trainingVidsFile(filename);
	string line;
	if (trainingVidsFile.is_open())
	{
		while (getline(trainingVidsFile, line))
		{
			trainingVids.push_back(line);
		}
		trainingVidsFile.close();
	}

	else cout << "Unable to open file";

}

int find_min(IplImage *img)
{
	int height = img->height;
	int width = img->width;
	int step = img->widthStep / sizeof(uchar);
	int channels = img->nChannels;
	uchar* data = (uchar *)img->imageData;
	int Min_ELE = 255;
	for (int i = 0; i<height;++i)
	{
		for (int j = 0; j<width;++j)
		{
			if (data[i*step + j*channels]<Min_ELE)
			{
				Min_ELE = data[i*step + j*channels];
			}
		}

		return Min_ELE;
	}
}

int find_max(IplImage *img)
{
	int height = img->height;
	int width = img->width;
	int step = img->widthStep / sizeof(uchar);
	int channels = img->nChannels;
	uchar* data = (uchar *)img->imageData;
	int Max_ELE = -255;
	for (int i = 0; i<height;++i)
	{
		for (int j = 0; j<width;++j)
		{
			if (data[i*step + j*channels]>Max_ELE)
			{
				Max_ELE = data[i*step + j*channels];
			}
		}

		return Max_ELE;
	}
}

Mat getGradient(Mat x)
{
	Mat grad_x, grad_y, grad;
	Mat abs_grad_x, abs_grad_y;

	Sobel(x, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(x, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	return  grad;

}

Mat getGradientDirection(Mat x)
{
	Mat grad_x, grad_y, grad;
	Mat abs_grad_x, abs_grad_y;

	Sobel(x, grad_x, CV_32F, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);

	/// Gradient Y
	//Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
	Sobel(x, grad_y, CV_32F, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);

	/// Total Gradient (approximate)
	//addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad ); 
	Mat orientation = Mat(grad_x.rows, grad_x.cols, CV_32F);

	// Calculate orientations of gradients --> in degrees
	// Loop over all matrix values and calculate the accompagnied orientation
	for (int i = 0; i < grad_x.rows; i++) {
		for (int j = 0; j < grad_x.cols; j++) {
			// Retrieve a single value
			float valueX = grad_x.at<float>(i, j);
			float valueY = grad_y.at<float>(i, j);
			// Calculate the corresponding single direction, done by applying the arctangens function
			float result = atan2(valueX, valueY);
			// Store in orientation matrix element
			orientation.at<float>(i, j) = (int)(180 + result * 180 / 3.1414);
			cout << orientation.at<float>(i, j) << "  ";
		}
		//cout<<orientation.at<float>(i,i);
	}

	return orientation;
}

Mat R_value(IplImage *img)
{
	int height = img->height;
	int width = img->width;
	int step = img->widthStep / sizeof(uchar);
	int channels = img->nChannels;
	uchar* data = (uchar *)img->imageData;
	//int k = channels;

	Mat M(height, width, CV_8UC1, Scalar(0, 0, 255));

	//cout<<step <<":" <<channels <<":" <<img->widthStep;
	for (int i = 0; i<height;++i)
	{
		for (int j = 0; j<width;++j)
		{
			for (int k = 0;k<3;++k)
			{
				M.at<uchar>(i, j) = data[i*step + j*channels + 2];
			}
		}

		return M;
	}
}

Mat CenterOfMasses(Vec4i L, Mat a)
{
	Mat R(200, 2, CV_16UC1);
	int index = 0;
	int deep = 7;
	R.at<ushort>(0, 0) = 0;
	R.at<ushort>(0, 1) = 0;
	if ((0.012 + L[0] - L[2]) == 0) { cout << ":::slope problem:"; return R; }
	float slope = (float(L[1] - L[3])) / (0.012 + L[0] - L[2]);


	int Accumulator_x = 0;
	int Acc_count = 0;
	int Accumulator_y = 0;

	for (int kx = L[0]; kx <= L[2]; kx++)
	{
		int ky = cvRound(slope*(kx - L[0]) + L[1]);

		for (int j = kx - 2 * deep; j< kx + deep; j++)
		{
			int i = cvRound((-1) / slope*(j - kx) + ky);
			if (i<a.rows && i>0 && j>0 && j<a.cols)
			{
				if (a.at<uchar>(i, j) == 255)
				{
					Acc_count = Acc_count + 1;
					Accumulator_x = Accumulator_x + j;
					Accumulator_y = Accumulator_y + i;
				}
			}
			//a.at<uchar>(i,j) = 250;
		}
		if (Acc_count>0)
		{

			ushort x0 = (int)cvRound(Accumulator_x / Acc_count);
			ushort y0 = (int)cvRound(Accumulator_y / Acc_count);
			R.at<ushort>(index, 0) = x0;
			R.at<ushort>(index, 1) = y0;
			//a.at<uchar>(R.at<ushort>(index,1),R.at<ushort>(index,0)) = 0;
			index = index + 1;
		}

		Accumulator_x = 0;
		Acc_count = 0;
		Accumulator_y = 0;
	}
	//cout<<"CoM:"<<L[0]<<"::::"<<L[1]<<":::index::::"<<index<<endl;
	R.at<ushort>(index, 0) = 0;
	R.at<ushort>(index, 1) = 0;
	return R;

}

float AVG(ushort x[], int n)
{
	float f = 0;
	int sum = 0;
	for (int i = 0; i<n; i++)
	{
		sum = sum + x[i];
	}
	f = (float)sum / n;
	return f;
}

float WeightedAVG(ushort x[], int n)
{
	float f = 0;
	int sum = 0;
	for (int i = 0; i<n; i++)
	{
		sum = sum + x[i];
	}
	f = (float)sum / n;
	f = f / n;
	return f;
}

float WeightedAVG1(int x[], int n)
{
	float f = 0;
	int sum = 0;
	for (int i = 0; i<n; i++)
	{
		sum = sum + x[i];
	}
	f = (float)sum / n;
	f = f / n;
	return f;
}

float WeightedAVG1(vector<int> x, int n)
{
	float f = 0;
	int sum = 0;
	for (int i = 0; i<n; i++)
	{
		sum = sum + x[i];
	}
	f = (float)sum / n;
	f = f / n;
	return f;
}

Mat getToolTipInfo(Mat a, ushort start_x, ushort start_y, float slope, ushort x1, ushort y1)
{
	Mat Tooltip(1, 3, CV_16UC1);
	ushort tip_topx = 0;
	ushort tip_topy = 0;
	ushort max_wdth = 0;
	ushort accumulator = 0;
	ushort deep = 7;

	for (int x = start_x - 10; x < x1; x++)
	{
		ushort y = cvRound(slope*(x - x1) + y1);
		for (int j = x - deep; j < x + deep; j++)
		{
			ushort i = cvRound((-1) / slope*(j - x) + y);

			if (i<a.rows && i>0 && j>0 && j<a.cols)

				if (a.at<uchar>(i, j) != 0)
				{
					accumulator = accumulator + 1;
					if (tip_topx == 0 && tip_topy == 0)
					{
						tip_topx = x;
						tip_topy = y;

					}
				}

		}


		if (accumulator>max_wdth)
		{
			max_wdth = accumulator;
		}
		accumulator = 0;
	}
	if (tip_topx == 0)
	{
		tip_topx = start_x;
		tip_topy = cvRound(slope*(tip_topx - x1) + y1);
		max_wdth = 12;

	}
	Tooltip.at<ushort>(1, 1) = tip_topx;
	Tooltip.at<ushort>(1, 2) = tip_topy;
	Tooltip.at<ushort>(1, 3) = max_wdth;

	return Tooltip;
}

Mat getToolBottomInfo(Mat a, ushort tip_topx, ushort tip_topy, float slope, ushort x1, ushort y1, ushort max_wdth)
{
	Mat Toolbottom(1, 3, CV_16UC1);

	ushort tip_bottomx = tip_topx + 20;
	ushort tip_bottomy = cvRound(slope*(tip_bottomx - x1) + y1);;
	ushort accumulator = 0;
	ushort zeros_found = 0;
	ushort ones_found = 0;
	ushort tip_width = max_wdth;
	int deep = 7;
	ushort cons = 0;
	for (int x = tip_topx;x <x1;x++)
	{
		int y = cvRound(slope*(x - x1) + y1);
		for (int j = x - 2 * deep; j< x + 2 * deep;j++)
		{
			ushort i = cvRound((-1) / slope*(j - x) + y);
			if (i<a.rows&& i>0 && j>0 && j<a.cols)


				if (a.at<uchar>(i, j) == 255)
				{
					tip_width = (tip_width>abs(j - x)) ? tip_width : abs(j - x);
					if (zeros_found == 1)
					{
						ones_found = 2;
					}
					else
					{
						ones_found = 1;
						accumulator = accumulator + 1;
					}
				}
				else
				{
					if (ones_found == 1)
					{
						zeros_found = 1;
					}

				}

		}

		if (accumulator>0.60*max_wdth)
		{

			cons = cons + 1;
			if (cons >= 6)
			{
				tip_bottomx = x - 5;
				tip_bottomy = cvRound(slope*(x - 5 - x1) + y1);
				// cout<<"yes"<<endl;
				break;
			}
		}
		else
		{
			cons = 0;

		}

		accumulator = 0;
		zeros_found = 0;
		ones_found = 0;
	}

	Toolbottom.at<ushort>(1, 1) = tip_bottomx;
	Toolbottom.at<ushort>(1, 2) = tip_bottomy;
	Toolbottom.at<ushort>(1, 3) = tip_width;

	return Toolbottom;
}

void drawline(IplImage* M, ushort tipx, ushort tipy, float slope, ushort toolWidth)
{

	ushort xc2 = M->width;
	ushort yc2 = cvRound(slope*(xc2 - tipx)) + tipy;
	if (yc2>M->height)
	{
		yc2 = M->height;
		xc2 = cvRound((yc2 - tipy) / slope) + tipx;
	}


	cvLine(M, Point(tipx, tipy), Point(xc2, yc2), Scalar(255, 0, 0), 4, CV_AA);
}

bool isInsideTheBox(ushort x1, ushort y1, ushort x2, ushort y2, ushort x3, ushort y3, ushort x4, ushort y4, ushort px, ushort py)
{
	float m1 = (float(y1 - y2) / (0.001 + (x1 - x2)));
	float m2 = (float(y3 - y4) / (0.001 + (x3 - x4)));
	float m3 = (float(y1 - y3) / (0.001 + (x1 - x3)));
	float m4 = (float(y2 - y4) / (0.001 + (x2 - x4)));

	int b1 = cvRound(y1 - m1*x1);
	int b2 = cvRound(y3 - m2*x3);
	int b3 = cvRound(y1 - m3*x1);
	int b4 = cvRound(y2 - m4*x2);
	/*cout<<x1<<"::"<<y1<<"::____"<<x2<<"::"<<y2<<":::------"<<x3<<"::"<<y3<<"::____"<<x4<<"::"<<y4<<":::::-----"<<px<<"::"<<py<<endl;
	cout<<m1<<"::"<<m2<<"::"<<m3<<"::"<<m4<<endl;
	cout<<b1<<"::"<<b2<<"::"<<b3<<"::"<<b4<<endl;*/

	/*if ((py - m1*px - b1) >=0)
	cout<<"t1:"<<1<<endl;

	if ((py - m2*px - b2) <=0)
	cout<<"t2:"<<1<<endl;

	if ((py - m3*px - b3) <=0)
	cout<<"t3:"<<1<<endl;

	if ((py - m4*px - b4) >=0)
	cout<<"t4:"<<1<<endl;*/

	if (((py - m1*px - b1) >= 0) && ((py - m2*px - b2) <= 0) && ((py - m3*px - b3) <= 0) && ((py - m4*px - b4) >= 0))
	{
		return true;
	}
	else
	{
		return false;
	}


}

cv::vector<int> parseString(string s, string delimiter)
{
	//std::string s = "scott>=tiger>=mushroom";
	//std::string delimiter = ">=";
	cv::vector<int> points;
	size_t pos = 0;
	int counter = 0;
	int index = 0;
	std::string token;
	while ((pos = s.find(delimiter)) != std::string::npos) {
		token = s.substr(0, pos);
		if (counter >= 3 && counter <= 6)
			points.push_back(stoi(token));
		//std::cout << token << std::endl;
		s.erase(0, pos + delimiter.length());
		counter++;
		index++;
	}

	return points;
}
/////////////////////////////////////////////////////////////

std::string reformPath(std::string path)
{
	std::replace(path.begin(), path.end(), '\\', '/');
	int lastIndex = path.find_last_of('/', path.size());
	if (lastIndex < (int)path.size() - 1)
		path += "/";
	return path;
}
//////////////////////////////////////////////////////////////

vector<int> readMultipleFiles(string dir)
{
	vector<int> VIS, VIS1;
	ifstream fin;
	string filepath;
	int num;
	DIR *dp;
	struct dirent *dirp;
	struct stat filestat;

	// cout << "dir to get files of: " << flush;

	dp = opendir(dir.c_str());
	if (dp == NULL)
	{
		cout << "Error(" << errno << ") opening " << dir << endl;
	}

	while ((dirp = readdir(dp)))
	{
		filepath = dir + "/" + dirp->d_name;

		if (stat(filepath.c_str(), &filestat)) continue;
		if (S_ISDIR(filestat.st_mode))         continue;

		VIS1 = parseStringOne(dirp->d_name, ".");
		VIS.push_back(VIS1[0]);
	}

	closedir(dp);

	return VIS;
}

////////////////////////////////////////////////////////////////////
cv::vector<int> parseStringOne(string s, string delimiter)
{
	//std::string s = "scott>=tiger>=mushroom";
	//std::string delimiter = ">=";
	cv::vector<int> points;
	size_t pos = 0;
	int counter = 0;
	int index = 0;
	std::string token;
	while ((pos = s.find(delimiter)) != std::string::npos) {
		token = s.substr(0, pos);
		if (counter >= 0 && counter <= 1)
			points.push_back(stoi(token));
		//std::cout << token << std::endl;
		s.erase(0, pos + delimiter.length());
		counter++;
		index++;
	}

	return points;
}
//////////////////////////////////////////////////////////////////////

Mat Myresize(Mat I, int f)
{
	int percent = f;
	IplImage *b;
	b = new IplImage(I);
	// declare a destination IplImage object with correct size, depth and channels
	///////////////////////////////////////////////////////////////////////////
	IplImage *destination = cvCreateImage
		(cvSize((int)((I.cols*percent) / 100), (int)((I.rows*percent) / 100)),
			b->depth, b->nChannels);

	cvResize(b, destination, 0);

	Mat ret = cvarrToMat(destination);

	return ret;
}

void cluster(Vector<cv::Point> pointList,int numberOfClusters, int trials , cv::Mat &outputClusters, cv::Point p1, cv::Point p2)
{
	cv::Mat samples = cv::Mat::zeros(p2.y - p1.y, p2.x - p1.x,CV_8U);
	
	for (int i = 0; i < pointList.size(); i++)
	{
		cv::Point p = pointList[i];
		samples.at<int>(p.x,p.y) = 1;
	}

	int clusterCount = numberOfClusters;
	Mat labels;
	int attempts = trials;
	Mat centers;
	kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 10000, 0.0001), attempts, KMEANS_PP_CENTERS, centers);


	//kmeans()
}
double distToLine(vector<float> line, Point p)
{
	double x1 = line.at(2);
	double y1 = line.at(3);
	double x2 = line.at(2) + line.at(0);
	double y2 = line.at(3) + line.at(1);
	double x0 = p.x;
	double y0 = p.y;
	double denom = sqrt(pow(x2 - x1,2) + pow(y2 - y1,2) );
	double d = abs((x2 - x1)*(y1 - y0) - (x1 - x0)*(y2 - y1)) / denom;
	return d;

}

double computeRMS(Point p1, Point p2)
{
	double rms = (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
	return sqrt(rms / 2);
}
Mat extractFeatureVector(Vector<Vector<Mat>>& V, int r, int c, int win_size)
{

	Mat R(1, 4 * V.size(), CV_32FC1, Scalar(0));
	int index = 0;
	int x = r - win_size / 2;
	int y = c - win_size / 2;
	int x1 = x + win_size;
	int y1 = y + win_size;
	for (int i = 0;i< V.size(); i++)
	{
		Vector<Mat> G = V[i];
		for (int j = 0;j<G.size(); j++)
		{
			Mat m = G[j];
			if (x<0 || y <0 || x1 >= m.rows || y1 >= m.cols)
				return R;
			float tf = ((float)(m.at<int>(x1, y1) - m.at<int>(x, y1) - m.at<int>(x1, y) + m.at<int>(x, y)) / (win_size * win_size));
			R.at<float>(0, index) = tf;
			index++;
		}
	}
	return R;
}
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
Vector<int> ExtractHOGF_ver2(Vector<Mat> img, int r, int c, int wind_width, int wind_height, int block_size, int stride, int cell_size, int bin_numbers)
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