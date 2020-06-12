// 15_2.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include <opencv.hpp>
#include <dnn.hpp>
using namespace cv;
using namespace std;
using namespace dnn;

#define OPENPOSE_VIDEO "..\\testImage\\sport.mp4"

int POSE_PAIRS[3][20][2] = {
	{   // COCO body
		{ 1,2 },{ 1,5 },{ 2,3 },
		{ 3,4 },{ 5,6 },{ 6,7 },
		{ 1,8 },{ 8,9 },{ 9,10 },
		{ 1,11 },{ 11,12 },{ 12,13 },
		{ 1,0 },{ 0,14 },
		{ 14,16 },{ 0,15 },{ 15,17 }
	},
	{   // MPI body
		{ 0,1 },{ 1,2 },{ 2,3 },
		{ 3,4 },{ 1,5 },{ 5,6 },
		{ 6,7 },{ 1,14 },{ 14,8 },{ 8,9 },
		{ 9,10 },{ 14,11 },{ 11,12 },{ 12,13 }
	},
	{   // hand
		{ 0,1 },{ 1,2 },{ 2,3 },{ 3,4 },         // thumb
		{ 0,5 },{ 5,6 },{ 6,7 },{ 7,8 },         // pinkie
		{ 0,9 },{ 9,10 },{ 10,11 },{ 11,12 },    // middle
		{ 0,13 },{ 13,14 },{ 14,15 },{ 15,16 },  // ring
		{ 0,17 },{ 17,18 },{ 18,19 },{ 19,20 }   // small
	} };

int openpose()
{

	//读入网络模型和权重文件
	String modelTxt = "D:\\openpose_pose_coco.prototxt";
	String modelBin = "D:\\openpose_caffe_models\\caffe_models\\pose\\coco\\pose_iter_440000.caffemodel";

	cv::dnn::Net net = cv::dnn::readNetFromCaffe(modelTxt, modelBin);

	int W_in = 368;
	int H_in = 368;
	float thresh = 0.1;

	VideoCapture cap;
	cap.open(OPENPOSE_VIDEO);

	if (!cap.isOpened())return -1;

	while (1) {
		cv::Mat frame;
		cap >> frame;
		if (frame.empty()) {
			std::cout << "frame is empty!!!" << std::endl;
			return -1;
		}

		//创建输入
		Mat inputBlob = blobFromImage(frame, 1.0 / 255, Size(W_in, H_in), Scalar(0, 0, 0), false, false);

		//输入
		net.setInput(inputBlob);

		//得到网络输出结果，结果为热力图
		Mat result = net.forward();

		int midx, npairs;
		int H = result.size[2];
		int W = result.size[3];

		//得到检测结果的关键点点数
		int nparts = result.size[1];

		// find out, which model we have
		//判断输出的模型类别
		if (nparts == 19)
		{   // COCO body
			midx = 0;
			npairs = 17;
			nparts = 18; // skip background
		}
		else if (nparts == 16)
		{   // MPI body
			midx = 1;
			npairs = 14;
		}
		else if (nparts == 22)
		{   // hand
			midx = 2;
			npairs = 20;
		}
		else
		{
			cerr << "there should be 19 parts for the COCO model, 16 for MPI, or 22 for the hand one, but this model has " << nparts << " parts." << endl;
			return (0);
		}

		// 获得身体各部分坐标
		vector<Point> points(22);
		for (int n = 0; n < nparts; n++)
		{
			// Slice heatmap of corresponding body's part.
			Mat heatMap(H, W, CV_32F, result.ptr(0, n));
			// 找到最大值的点
			Point p(-1, -1), pm;
			double conf;
			minMaxLoc(heatMap, 0, &conf, 0, &pm);
			//判断置信度
			if (conf > thresh) {
				p = pm;
			}
			points[n] = p;
		}

		//连接身体各个部分，并且绘制
		float SX = float(frame.cols) / W;
		float SY = float(frame.rows) / H;
		for (int n = 0; n < npairs; n++)
		{
			Point2f a = points[POSE_PAIRS[midx][n][0]];
			Point2f b = points[POSE_PAIRS[midx][n][1]];

			//如果前一个步骤没有找到相应的点，则跳过
			if (a.x <= 0 || a.y <= 0 || b.x <= 0 || b.y <= 0)
				continue;

			// 缩放至图像的尺寸
			a.x *= SX; a.y *= SY;
			b.x *= SX; b.y *= SY;

			//绘制
			line(frame, a, b, Scalar(0, 200, 0), 2);
			circle(frame, a, 3, Scalar(0, 0, 200), -1);
			circle(frame, b, 3, Scalar(0, 0, 200), -1);
		}
		imshow("frame", frame);
		waitKey(30);
	}
	return 0;
}
int main()
{
	openpose();
    return 0;
}

