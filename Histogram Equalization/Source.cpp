#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
void self_equalizeHist(InputArray _src,OutputArray _dst);

class self_EqualizeHistCalcHist_Invoker : public cv::ParallelLoopBody
{
private:
	self_EqualizeHistCalcHist_Invoker& operator=(const self_EqualizeHistCalcHist_Invoker&);
	cv::Mat& src_;
	int* globalHistogram_;
	cv::Mutex* histogramLock_;
public:
	enum { HIST_SZ = 256 };
	self_EqualizeHistCalcHist_Invoker(cv::Mat& src, int* histogram, cv::Mutex* histogramLock)
		:src_(src), globalHistogram_(histogram), histogramLock_(histogramLock)
	{

	}
	void operator()(const cv::Range& rowRange) const CV_OVERRIDE
	{
		int localHistogram[HIST_SZ] = { 0, };
		const size_t sstep = src_.step;

		int width = src_.cols;
		int height = rowRange.end - rowRange.start;

		if (src_.isContinuous())
		{
			width *= height;
			height = 1;
		}
		for (const uchar* ptr = src_.ptr<uchar>(rowRange.start); height--; ptr += sstep)////
		{
			int x = 0;
			for (; x <= width - 4; x += 4)
			{
				int t0 = ptr[x], t1 = ptr[x + 1];
				localHistogram[t0]++;
				localHistogram[t1]++;
				t0 = ptr[x + 2];
				t1 = ptr[x + 3];
				localHistogram[t0]++;
				localHistogram[t1]++;
			}
			for (; x < width; ++x)
				localHistogram[ptr[x]]++;
		}
		cv::AutoLock lock(*histogramLock_);
		for (int i = 0; i < HIST_SZ; i++)
			globalHistogram_[i] += localHistogram[i];
	}
	static bool isWorthParallel(const cv::Mat& src)
	{
		return (src.total() >= 640 * 480);
	}
};

class self_EqualizeHistLut_Invoker : public cv::ParallelLoopBody
{
private:
	self_EqualizeHistLut_Invoker& operator=(const self_EqualizeHistLut_Invoker&);

	cv::Mat& src_;
	cv::Mat& dst_;
	int* lut_;

public:
	self_EqualizeHistLut_Invoker(cv::Mat& src, cv::Mat& dst, int* lut)
		: src_(src), dst_(dst), lut_(lut)
	{ }
	void operator()(const cv::Range& rowRange) const CV_OVERRIDE
	{
		const size_t sstep = src_.step;//each row length
		const size_t dstep = dst_.step;

		int width = src_.cols;
		int height = rowRange.end - rowRange.start;
		int* lut = lut_;

		if (src_.isContinuous() && dst_.isContinuous())
		{
			width *= height;
			height = 1;
		}

		const uchar* sptr = src_.ptr<uchar>(rowRange.start);
		uchar* dptr = dst_.ptr<uchar>(rowRange.start);

		for (; height--; sptr += sstep, dptr += dstep)
		{
			int x = 0;
			for (; x <= width - 4; x += 4)
			{
				int v0 = sptr[x];
				int v1 = sptr[x + 1];
				int x0 = lut[v0];
				int x1 = lut[v1];
				dptr[x] = (uchar)x0;
				dptr[x + 1] = (uchar)x1;

				v0 = sptr[x + 2];
				v1 = sptr[x + 3];
				x0 = lut[v0];
				x1 = lut[v1];
				dptr[x + 2] = (uchar)x0;
				dptr[x + 3] = (uchar)x1;
			}

			for (; x < width; ++x)
				dptr[x] = (uchar)lut[sptr[x]];
		}
	}
	static bool isWorthParallel(const cv::Mat& src)
	{
		return (src.total() >= 640 * 480);
	}
};

int main()
{
	Mat img,out,out2;
	img = imread("D:/self_opencv/HE/Lenna.jpg");
	cvtColor(img,img,COLOR_BGR2GRAY);
	self_equalizeHist(img, out);
	equalizeHist(img,out2);

	imshow("before", img);


	namedWindow("after");
	moveWindow("after", 40, 30);
	imshow("after", out);

	imshow("after_offical", out2);

	waitKey(0);
	return 0;
}
void self_equalizeHist(InputArray _src, OutputArray _dst)
{
	Mat src = _src.getMat();
	_dst.create(src.size(),src.type());
	Mat dst = _dst.getMat();

	Mutex histogramLockInstance;
	const int hist_sz = self_EqualizeHistCalcHist_Invoker::HIST_SZ;
	int hist[hist_sz] = { 0, };
	int lut[hist_sz];

	self_EqualizeHistCalcHist_Invoker calcBody(src,hist,&histogramLockInstance);
	self_EqualizeHistLut_Invoker      lutBody(src, dst, lut);
	cv::Range heightRange(0,src.rows);
	
	/*if (self_EqualizeHistCalcHist_Invoker::isWorthParallel(src))
		parallel_for_(heightRange, calcBody);
	else*/
		calcBody(heightRange);
	
	int i = 0;
	while (!hist[i])
		++i;

	int total = (int)src.total();//total pixel
	if (hist[i] == total)
	{
		dst.setTo(i);//set all pixel value is i
		return;
	}

	float scale = (hist_sz - 1.f) / (total - hist[i]);
	int sum = 0;

	for (lut[i++] = 0; i < hist_sz; ++i)
	{
		sum += hist[i];
		lut[i] = saturate_cast<uchar>(sum * scale);
		//if value<0 value=0 value>255 value=255
	}
	/*if (self_EqualizeHistLut_Invoker::isWorthParallel(src))
		parallel_for_(heightRange, lutBody);
	else*/
		lutBody(heightRange);
}
