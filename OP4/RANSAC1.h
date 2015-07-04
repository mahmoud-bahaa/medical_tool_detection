#ifndef RANSAC_H
#define RANSAC_H
#define PI 3.14159265

#include "CommonFunctions.h"
using namespace std;
using namespace cv;


Vec4s RANSAC1 ( Mat& XY , ushort size, ushort epsilon, ushort threshold , ushort Iterations )
{
	//cout<<size<<":inside ransac"<<endl;
Vec4s L;
ushort num = Iterations;
ushort Inliers[200];
ushort Inliers_size = 0;
ushort min_distance = 100;
L[0] = 0;
L[1] = 0;
L[2] = 0;
L[3] = 0;
float Avg = 0;
ushort r1,r2;
cout<<size<<"-----------------------------------------";
	if (size <20)
		return L;

cout<<endl;
    while(num>0)
	{
    r1 = 1;
    r2 = 1;


    while(abs(r1 - r2)<2)
	{		
    r1 = cvRound(size/2) + rand()%cvRound(1+size/2);
    r2 = cvRound(size/2) + rand()%cvRound(1+size/2);
	}
	
	
    ushort p1x = XY.at<ushort>(r1,0);
    ushort p1y = XY.at<ushort>(r1,1);
    
    ushort p2x = XY.at<ushort>(r2,0);
    ushort p2y = XY.at<ushort>(r2,1);

	if( abs(p1x-p2x)<4 || abs(p1y-p2y)<4)
		continue;
    
    for (int i = 1; i< size ; i++)
	{
        ushort p0x = XY.at<ushort>(i,0);
        ushort p0y = XY.at<ushort>(i,1);        
		ushort d = abs((p2x-p1x)*(p1y-p0y)-(p1x-p0x)*(p2y-p1y))/sqrt((p2x-p1x)^2 + (p2y-p1y)^2);
        if(d<=epsilon)
		{
            Inliers[Inliers_size] = d;
			Inliers_size = Inliers_size + 1;
		}
        
	}
    if(Inliers_size>=threshold)
	{
        float Avg = WeightedAVG(Inliers,Inliers_size) ;
        if (Avg < min_distance)
		{
            L[0] = p1x;
            L[1] = p1y;
            L[2] = p2x;
            L[3] = p2y;
			
            min_distance = Avg;
			cout<<"Avg:"<<Avg<<" last AVG:"<<min_distance<<":size:"<<Inliers_size<<endl;
		}
        
	}
    
    num = num - 1;
    Inliers_size = 0;

	}
	if (L[0]==0)
		cout<<"Inliers_size:"<<Inliers_size<<"min_distance:"<<min_distance<<endl;
	cout<<"final Avg:"<<Avg<<" last AVG:"<<min_distance<<":size:"<<Inliers_size<<endl;
	return L;
}
Vec4i RANSAC1 ( Vector<Point>& XY , int size, int epsilon/*Model fitting threshold*/, int threshold /*acceptance percentage*/, int Iterations  /*Minimum iterations*/, float predefinedAngle = 45,float angleRange = 30, float predefinedLength = 20)
{

static float prevAngle = 0.0f;
float lineLength = 0.0;
Vec4i L;
int num = Iterations;
//cout<<size<<":inside ransac"<<endl;
float angle = 0.0f;
vector<int> Inliers;
int Inliers_size = 0;
int minInliersSize = 0;
float min_distance = 100;
L[0] = 0;
L[1] = 0;
L[2] = 0;
L[3] = 0;
float Avg = 0;
int r1,r2;
//cout<<size<<"-----------------------------------------";
	if (size <20)
		return L;

//cout<<endl;
    while(num>0)
	{
    r1 = 1;
    r2 = 1;


    while(abs(r1 - r2)<2)
	{	
		num = num - 1;
		r1 =  rand()%cvRound(size);
		r2 =  rand()%cvRound(size);
	}
	
	Point P1 = XY[r1] ;
	Point P2 = XY[r2] ;
    int p1x = P1.x;
    int p1y = P1.y;
    
    int p2x = P2.x;
    int p2y = P2.y;

	
    
    for (int i = 0; i< size ; i++)
	{
        int p0x = XY[i].x;
        int p0y = XY[i].y;
		int d = abs((p2x-p1x)*(p1y-p0y)-(p1x-p0x)*(p2y-p1y))/sqrt((p2x-p1x)*(p2x-p1x) + (p2y-p1y)*(p2y-p1y));
        if(d<=epsilon)
		{

			Inliers.push_back(d);
			Inliers_size++;
		}

		//cout<< i << " distance " << d << endl;
        
	}
    if(Inliers_size>=threshold)
	{
        float Avg = WeightedAVG1(Inliers,Inliers_size) ;
		angle = atan2(p2y - p1y, p2x - p1x) * 180 / PI;
		lineLength = sqrt(pow(p2x - p1x, 2) + pow(p2y - p1y, 2));
		
        if ( (abs(angle-prevAngle) <= 5 || prevAngle == 0 ) && lineLength >= predefinedLength && abs(angle-predefinedAngle) <= angleRange  && (Inliers_size > minInliersSize || (Inliers_size == minInliersSize && Avg < min_distance)))
		{
            L[0] = p1x;
            L[1] = p1y;
            L[2] = p2x;
            L[3] = p2y;
			
            min_distance = Avg;
			minInliersSize = Inliers_size;
			//cout<<"Avg:"<<Avg<<" last AVG:"<<min_distance<<":size:"<<Inliers_size<<"::::"<<r1 <<"::::"<<r2<<endl;
			//cout << "Angle: " << angle <<" - Previous Angle: "<< prevAngle<< endl;
			prevAngle = angle;

		}
        
	}
	
	Inliers.clear();
    Inliers_size = 0;

	}
	
	/*if (L[0]==0)
		cout<<"Inliers_size:"<<Inliers_size<<"min_distance:"<<min_distance<<endl;*/
	//cout<<"final Avg:"<<Avg<<" last AVG:"<<min_distance<<":size:"<<Inliers_size<<"::::"<<r1 <<"::::"<<r2<<endl;

	return L;
}
Vec4i lineFit(Vector<Point>& XY, int size, int epsilon, int threshold, int Iterations, float predefinedAngle = 45, float angleRange = 30, float predefinedLength = 20)
{
	Vec4i line = RANSAC1(XY, size, epsilon, threshold, Iterations);
	/*
	vector<float> fittedLine;
	vector<Point> inliers;
	Point lineCenter(0, 0);

	std::vector<Point> points;

	
	for (int i = 0;i < XY.size();i++) {
		
		points.push_back(XY[i]);
	}
	
	
	//cv::fitLine(Mat(points), fittedLine, CV_DIST_L12, 0.0, 0.01, 0.01);

	
	*/
	return line;
}
#endif // !HEADER_H







