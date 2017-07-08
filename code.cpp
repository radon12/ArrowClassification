	#include "opencv2/highgui/highgui.hpp"
	#include "opencv2/imgproc/imgproc.hpp"
	#include <iostream>
	#include <stdio.h>
	#include <fstream>

	using namespace cv;
	using namespace std;

	/// Global variables

	Mat src, src_gray, temp;
	Mat dst, detected_edges, mask;

	int edgeThresh = 1;
	int lowThreshold = 300;
	int highThreshold = 430;
	int const max_lowThreshold = 500;
	int ratio = 3;
	int kernel_size = 3;
	char window_name[] = "Edge Map";
	Point2f *center;
	float *radius;
	int ind;
	double maxarea;
	vector<Point2f> corners;

	void CannyThreshold(int, void*)
	{
		/// Reduce noise with a kernel 3x3
		blur(src_gray, detected_edges, Size(3, 3));
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		/// Canny detector
		Canny(detected_edges, detected_edges, lowThreshold, highThreshold, kernel_size);

		imshow("detector",detected_edges);
		dilate(detected_edges, mask, getStructuringElement(MORPH_RECT, Size(5, 5)), Point(-1, -1), 3);
		erode(mask, mask, getStructuringElement(MORPH_RECT, Size(5, 5)), Point(-1, -1), 3);

		findContours(mask, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE, Point(0, 0));

		int csize = contours.size();
		if (csize)
		{
			maxarea = contourArea(contours[0]);
			ind = 0;

			for (int i = 1; i < csize; i++)
			{
				double area = contourArea(contours[i]);
				if (area > maxarea)
				{
					ind = i;
					maxarea = area;
				}
			}
			//vector<RotatedRect> minRect(1);
			Rect boundRect;
			boundRect = boundingRect(Mat(contours[ind]));

			rectangle(temp, boundRect.tl(), boundRect.br(), Scalar(255,0,0), 1, 8, 0);
			Point2f corner1 = boundRect.br();
			Point2f corner2 = boundRect.tl();
			int a = corner1.x - corner2.x;
			int b = corner1.y - corner2.y;
			line(temp, Point2f((corner2.x + a/3), corner2.y), Point2f((corner2.x + a / 3),corner1.y), Scalar(0, 255, 0), 1, 8, 0);
			line(temp, Point2f((corner2.x + 2 * a / 3), corner2.y), Point2f((corner2.x + 2 * a / 3),corner1.y), Scalar(0, 255, 0), 1, 8, 0);
			line(temp, Point2f(corner2.x,(corner2.y + b / 3) ), Point2f(corner1.x,(corner2.y + b / 3)), Scalar(0, 255, 0), 1, 8, 0);
			line(temp, Point2f(corner2.x,(corner2.y + 2 * b / 3)), Point2f(corner1.x,(corner2.y + 2 * b / 3)), Scalar(0, 255, 0), 1, 8, 0);

			Mat r[9];
		 	int avg[9];
			int rindex=0;
			for(int i=0;i<3;i++)
			{
				for(int j=0;j<3;j++)
				{
					r[rindex]=Mat(temp,Rect(corner2.x+(j*a)/3,corner2.y+(i*b)/3,a/3,b/3));
					Scalar avgPixelIntensity=mean(r[rindex]);
					avg[rindex++] = (avgPixelIntensity.val[0] + avgPixelIntensity.val[1] + avgPixelIntensity.val[2]) / 3;
				}
			}

			for(int i=0;i<9;i++)
						cout<<avg[i]<<",";
			cout<<1<<"\n";

		}

		imshow("temp", temp);
		imshow("canny", detected_edges);
		imshow("mask", mask);
		/// Using Canny's output as a mask, we display our result
		dst = Scalar::all(0);
		src.copyTo(dst, detected_edges);
		imshow(window_name, dst);

	}


	/** @function main */
	int main(int argc, char** argv)
	{
		freopen("outright.txt","w",stdout);
		VideoCapture cam(0);
		while (waitKey(10) != 'q') {
			/// Load an image

			cam.read(src);
			temp = src;
			if (!src.data)
			{
				return -1;
			}

			/// Create a matrix of the same type and size as src (for dst)
			dst.create(src.size(), src.type());

			/// Convert the image to grayscale
			cvtColor(src, src_gray, CV_BGR2GRAY);

			/// Create a window
			namedWindow(window_name, CV_WINDOW_AUTOSIZE);

			/// Create a Trackbar for user to enter threshold
			createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
			createTrackbar("Max Threshold:", window_name, &highThreshold, max_lowThreshold, CannyThreshold);

			///Apply canny and show image
			CannyThreshold(0, 0);
		}
		fclose(stdout);
		return 0;
	}
