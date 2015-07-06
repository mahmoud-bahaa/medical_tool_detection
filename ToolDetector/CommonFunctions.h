#ifndef HEADER_H
#define HEADER_H

#include "AllIncludes.h"
#define PI 3.14159265
#define FN 324

using namespace std;
using namespace cv;

void readFromTxt(string filename, vector<string> &vids);
int find_min(IplImage *img);
int find_max(IplImage *img);

Mat getGradient(Mat x);
Mat getGradientDirection(Mat x);
std::vector<KeyPoint> getTooltipCenter( Mat &im, int blobArea = 50);

Mat R_value(IplImage *img);
Mat CenterOfMasses(Vec4i L, Mat a);
float AVG(ushort x[], int n);
float WeightedAVG(ushort x[], int n);
float WeightedAVG1(int x[], int n);
float WeightedAVG1(vector<int> x, int n);
Mat getToolTipInfo(Mat a, ushort start_x, ushort start_y, float slope, ushort x1, ushort y1);
Mat getToolBottomInfo(Mat a, ushort tip_topx, ushort tip_topy, float slope, ushort x1, ushort y1, ushort max_wdth);

void drawline(IplImage* M, ushort tipx, ushort tipy, float slope, ushort toolWidth);
bool isInsideTheBox(ushort x1, ushort y1, ushort x2, ushort y2, ushort x3, ushort y3, ushort x4, ushort y4, ushort px, ushort py);


cv::vector<int> parseString(string s, string delimiter);
std::string reformPath(std::string path);
vector<int> readMultipleFiles(string dir);
cv::vector<int> parseStringOne(string s, string delimiter);
Mat Myresize(Mat I, int f);
double computeRMS(Point p1, Point p2);
double distToLine(vector<float> line, Point p);
Mat extractFeatureVector(Vector<Vector<Mat>>& V, int r, int c, int win_size);

Vector<int> ExtractHOGF_ver2(Vector<Mat> img, int r, int c, int wind_width, int wind_height, int block_size, int stride, int cell_size, int bin_numbers);





#endif // !HEADER_H
