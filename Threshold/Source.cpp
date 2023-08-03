#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include<iostream>

using namespace std;
using namespace cv;

struct res
{
	double thres;
	Mat out;
};

struct res self_threshold(InputArray _src, OutputArray _dst, double thresh, double maxval, int type);
static void self_thresh_8u(const Mat& _src,Mat& _dst,uchar thresh,uchar maxval,int type);
static double self_getThreshVal_Ostu_8u(const Mat& _src);

template<typename T, size_t BinsOnStack = 0u>
static double self_getThreshVal_Ostu(const Mat& _src, const Size& size)
{
	const int N = std::numeric_limits<T>::max() + 1;////
	int i, j;

	AutoBuffer<int, BinsOnStack>hBuf(N);////
	memset(hBuf.data(), 0, hBuf.size() * sizeof(int));
	int* h = hBuf.data();
	for (i = 0; i < size.height; i++)
	{
		const T* src = _src.ptr<T>(i, 0);
		j = 0;
		for (; j < size.width; j++)
			h[src[j]]++;
	}

	double mu = 0, scale = 1. / (size.width * size.height);
	for (i = 0; i < N; i++)
	{
		mu += i * (double)h[i];
	}

	mu *= scale;
	double mu1 = 0, q1 = 0;
	double max_sigma = 0, max_val = 0;

	for (i = 0; i < N; i++)
	{
		double p_i, q2, mu2, sigma;

		p_i = h[i] * scale;
		mu1 *= q1;
		q1 += p_i;//omega0
		q2 = 1. - q1;//omega1

		if (std::min(q1, q2) < FLT_EPSILON || std::max(q1, q2) > 1. - FLT_EPSILON)
			continue;

		mu1 = (mu1 + i * p_i) / q1;
		mu2 = (mu - q1 * mu1) / q2;

		sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
		if (sigma > max_sigma)
		{
			max_sigma = sigma;
			max_val = i;
		}
	}
	return max_val;
}

int main()
{
	Mat img, out,out3;
	struct res self_result;
	img = imread("D:/self_opencv/Threshold/Ichika.jpg");
	double thresh = 100.0;
	double maxval = 255.0;
	cvtColor(img, img, COLOR_BGR2GRAY);
	imshow("before", img);

	double offical_thresh,self_thresh=100;
	offical_thresh=threshold(img, out3, thresh, maxval,THRESH_OTSU|THRESH_BINARY);//
	self_result = self_threshold(img, out, thresh, maxval,THRESH_OTSU|THRESH_BINARY);// ;
	

	cout << offical_thresh << endl;
	cout << self_result.thres<< endl;


	imshow("after",self_result.out);
	imshow("after_offical",out3);

	waitKey(0);


	/*if (img.depth() == CV_8U)
		cout << "0" << endl;
	else if(img.depth() == CV_16S)
		cout << "1" << endl;
	else if(img.depth() == CV_16U)
		cout << "2" << endl;
	else if (img.depth() == CV_32F)
		cout << "3" << endl;
	else if (img.depth() == CV_64F)
		cout << "4" << endl;*/


	/*
	src = _src.ptr();
	dst = _dst.ptr();
	for (int i = 0; i < roi.height; i++, src += src_step, dst += dst_step)////
	{
		j = j_scalar;
		for (; j < roi.width; j++)
			dst[j] = tab[src[j]];
	}*/
	return 0;
	
}
static void self_thresh_8u(const Mat& _src, Mat& _dst, uchar thresh, uchar maxval, int type)
{
	Size roi = _src.size();
	roi.width *= _src.channels();
	size_t src_step = _src.step;
	size_t dst_step = _dst.step;

	if (_src.isContinuous() && _dst.isContinuous())
	{
		roi.width *= roi.height;
		roi.height = 1;
		src_step = dst_step = roi.width;
	}

	int j = 0;
	const uchar* src = _src.ptr();
	uchar* dst = _dst.ptr();
	int j_scalar = j;

	if (j_scalar < roi.width)
	{
		const int thresh_pivot = thresh + 1;
		uchar tab[256] = { 0 };
		switch (type)
		{
			case THRESH_BINARY:
				memset(tab, 0, thresh_pivot);
				if (thresh_pivot < 256)
					memset(tab+thresh_pivot, maxval, 256 - thresh_pivot);
			break;
			case THRESH_BINARY_INV:
				memset(tab, maxval,thresh_pivot);
				if (thresh_pivot < 256)
					memset(tab+thresh_pivot,0,256-thresh_pivot);
			break;
			case THRESH_TRUNC:
				for (int i = 0; i <= thresh; i++)
					tab[i] = (uchar)i;
				if (thresh_pivot < 256)
					memset(tab + thresh_pivot, thresh, 256 - thresh_pivot);
			break;
			case THRESH_TOZERO:
				memset(tab, 0, thresh_pivot);
				for(int i=thresh_pivot;i<256;i++)
					tab[i] = (uchar)i;
			break;
			case THRESH_TOZERO_INV:
				for (int i = 0; i < thresh_pivot; i++)
					tab[i] = (uchar)i;
				if (thresh_pivot < 256)
					memset(tab + thresh_pivot, 0, 256 - thresh_pivot);
			break;
			case THRESH_OTSU:
			break;
		}
		src = _src.ptr();
		dst = _dst.ptr();
		for (int i = 0; i < roi.height; i++, src += src_step, dst += dst_step)////
		{
			j = j_scalar;
			for (; j < roi.width; j++)
				dst[j] = tab[src[j]];
		}
	}
}


static double self_getThreshVal_Ostu_8u(const Mat& _src)
{
	Size size = _src.size();
	int step = (int)_src.step;
	if (_src.isContinuous())
	{
		size.width *= size.height;
		size.height = 1;
		step = size.width;
	}
	return self_getThreshVal_Ostu<uchar,256u>(_src,size);
}

struct res self_threshold(InputArray _src,OutputArray _dst,double thresh, double maxval, int type)
{
	struct res answer;
	Mat src = _src.getMat();
	
	int automatic_thresh = (type &~THRESH_MASK);
	type &= THRESH_MASK;

	_dst.create(src.size(), src.type());
	Mat dst = _dst.getMat();

	if (automatic_thresh == THRESH_OTSU)
	{
		int src_type = src.type();
		thresh = self_getThreshVal_Ostu_8u(src);

		int ithresh = cvFloor(thresh);
		thresh = ithresh;
		int imaxval = cvRound(maxval);

		answer.thres = thresh;
	}
	if (src.depth() == CV_8U)
	{
		int ithresh = cvFloor(thresh);
		thresh = ithresh;
		int imaxval = cvRound(maxval);

		if (type == THRESH_TRUNC)
			imaxval = ithresh;
		imaxval = saturate_cast<uchar>(imaxval);

		if (ithresh < 0 || ithresh >= 255)
		{
			if (type == THRESH_BINARY || type == THRESH_BINARY_INV ||
				((type == THRESH_TRUNC || type == THRESH_TOZERO_INV) && ithresh < 0) ||
				(type == THRESH_TOZERO && ithresh >= 255))
			{
				int v = type == THRESH_BINARY ? (ithresh >= 255 ? 0 : imaxval) :
					type == THRESH_BINARY_INV ? (ithresh >= 255 ? imaxval : 0) :0;
				dst.setTo(v);
			}
			else
				src.copyTo(dst);

			answer.thres = thresh;
			answer.out = dst;
			return answer;//thresh;
		}
		thresh = ithresh;
		maxval = imaxval;
	}

	Range R(0,dst.rows);
	int row0 = R.start;//0
	int row1 = R.end;

	Mat srcStrip = src.rowRange(row0, row1);
	Mat dstStrip = src.rowRange(row0, row1);

	self_thresh_8u(srcStrip, dstStrip,(uchar)thresh,(uchar)maxval,type);
	
	answer.thres = thresh;
	answer.out = dstStrip;

	return answer;//thresh;
}