#include<iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"


using namespace std;
using namespace cv;

typedef ushort HT;

/**
 * This structure represents a two-tier histogram. The first tier (known as the
 * "coarse" level) is 4 bit wide and the second tier (known as the "fine" level)
 * is 8 bit wide. Pixels inserted in the fine level also get inserted into the
 * coarse bucket designated by the 4 MSBs of the fine bucket value.
 *
 * The structure is aligned on 16 bits, which is a prerequisite for SIMD
 * instructions. Each bucket is 16 bit wide, which means that extra care must be
 * taken to prevent overflow.
 */
typedef struct
{
	HT coarse[16];
	HT fine[16][16];
}Histogram;


void self_medianBlur(const Mat& src, Mat& dst, int ksize);
static void medianBlur_8u_O1(const Mat& _src, Mat& _dst, int ksize);


static inline void histogram_add(const HT x[16], HT y[16])
{
	int i;
	for (i = 0; i < 16; ++i)
		y[i] = (HT)(y[i] + x[i]);
}

static inline void histogram_sub(const HT x[16], HT y[16])
{
	int i;
	for (i = 0; i < 16; ++i)
		y[i] = (HT)(y[i] - x[i]);
}

static inline void histogram_muladd(int a, const HT x[16],HT y[16])
{
	for (int i = 0; i < 16; ++i)
		y[i] = (HT)(y[i] + a * x[i]);
}
int main()
{
	Mat img, out,out2;
	int i = 7;
	img = imread("D:/self_opencv/Filter/Lenna2.jpg");
	cvtColor(img,img, COLOR_BGR2GRAY);

	imshow("before", img);
	self_medianBlur(img,out,i/2);
	medianBlur(img, out2,i);//11

	imshow("after", out);
	imshow("after_offical", out2);



	waitKey(0);
	return 0;
}

static void
medianBlur_8u_O1(const Mat& _src, Mat& _dst, int ksize)
{
#define COP(c,j,x,op) \
    h_coarse[ 16*(n*c+j) + (x>>4) ] op, \
    h_fine[ 16 * (n*(16*c+(x>>4)) + j) + (x & 0xF) ] op

	int cn = _dst.channels();
	int m = _dst.rows;
	int r = (ksize - 1) / 2;
	size_t sstep = _src.step,dstep=_dst.step;

	Histogram CV_DECL_ALIGNED(16) H[4];////
	HT CV_DECL_ALIGNED(16) luc[4][16];////

	int STRIPE_SIZE = std::min(_dst.cols,512/cn);

	vector<HT> _h_coarse(1 * 16 * (STRIPE_SIZE + 2 * r) * cn + 16);
	vector<HT> _h_fine(16 * 16 * (STRIPE_SIZE + 2 * r) * cn + 16);
	HT* h_coarse = alignPtr(&_h_coarse[0], 16);
	HT* h_fine = alignPtr(&_h_fine[0], 16);

	for (int x = 0; x < _dst.cols; x +=STRIPE_SIZE)
	{
		int i, j, k, c;
		int n = std::min(_dst.cols - x, STRIPE_SIZE) + r * 2;
		const uchar* src = _src.data + x * cn;
		uchar* dst = _dst.data + (x - r) * cn;

		memset(h_coarse, 0, 16 * n * cn * sizeof(h_coarse[0]));
		memset(h_fine, 0, 16 * 16 * n * cn * sizeof(h_fine[0]));
	
		for (c = 0; c < cn; c++)
		{
			for (j = 0; j < n; j++)
				COP(c, j, src[cn * j + c], += r + 2);


			for (i = 1; i < r; i++)
			{
				const uchar* p = src + sstep * std::min(i, m - 1);
				for (j = 0; j < n; j++)
					COP(c, j, p[cn * j + c], ++);
			}
		}

		for (i = 0; i < m; i++)
		{
			const uchar* p0 = src + sstep * std::max(0, i - r - 1);
			const uchar* p1 = src + sstep * std::min(m - 1, i + r);

			memset(H, 0, cn * sizeof(H[0]));
			memset(luc, 0, cn * sizeof(luc[0]));

			for (c = 0; c < cn; c++)
			{
				// Update column histograms for the entire row.
				for (j = 0; j < n; j++)
				{
					COP(c, j, p0[j * cn + c], --);
					COP(c, j, p1[j * cn + c], ++);
				}

				// First column initialization
				for (k = 0; k < 16; ++k)
					histogram_muladd(2 * r + 1, &h_fine[16 * n * (16 * c + k)], &H[c].fine[k][0]);

			
				for (j = 0; j < 2 * r; ++j)
					histogram_add(&h_coarse[16 * (n * c + j)], H[c].coarse);
				
				for (j = r; j < n - r; j++)
				{
					int t = 2 * r * r + 2 * r, b, sum = 0;
					HT* segment;

					histogram_add(&h_coarse[16 * (n * c + std::min(j + r, n - 1))], H[c].coarse);

					// Find median at coarse level
					for (k = 0; k < 16; ++k)
					{
						sum += H[c].coarse[k];
						if (sum > t)
						{
							sum -= H[c].coarse[k];
							break;
						}
					}
					assert(k < 16);

					/* Update corresponding histogram segment */
					if (luc[c][k] <= j - r)
					{
						memset(&H[c].fine[k], 0, 16 * sizeof(HT));
						for (luc[c][k] = j - r; luc[c][k] < MIN(j + r + 1, n); ++luc[c][k])
							histogram_add(&h_fine[16 * (n * (16 * c + k) + luc[c][k])], H[c].fine[k]);

						if (luc[c][k] < j + r + 1)
						{
							histogram_muladd(j + r + 1 - n, &h_fine[16 * (n * (16 * c + k) + (n - 1))], &H[c].fine[k][0]);
							luc[c][k] = (HT)(j + r + 1);
						}
					}
					else
					{
						for (; luc[c][k] < j + r + 1; ++luc[c][k])
						{
							histogram_sub(&h_fine[16 * (n * (16 * c + k) + MAX(luc[c][k] - 2 * r - 1, 0))], H[c].fine[k]);
							histogram_add(&h_fine[16 * (n * (16 * c + k) + MIN(luc[c][k], n - 1))], H[c].fine[k]);
						}
					}

					histogram_sub(&h_coarse[16 * (n * c + MAX(j - r, 0))], H[c].coarse);

					/* Find median in segment */
					segment = H[c].fine[k];
					for (b = 0; b < 16; b++)
					{
						sum += segment[b];
						if (sum > t)
						{
							dst[dstep * i + cn * j + c] = (uchar)(16 * k + b);
							break;
						}
					}
					assert(b < 16);
				}
			}
		}
	}
#undef COP
}
void self_medianBlur(const Mat& src, Mat& dst, int ksize)
{
	dst.create(src.size(), src.type());

	for (int i = ksize; i < src.rows - ksize; i++)
	{
		uchar* p_dst = dst.ptr<uchar>(i);
		int median = 0;
		int nm = 0;
		int hist[256] = {0};
		for (int j = ksize; j < src.cols - ksize; j++)
		{
			if (j == ksize)
			{
				for (int tmp_i = -ksize; tmp_i <= ksize; tmp_i++)
				{
					const uchar* p_src = src.ptr<uchar>(i + tmp_i);
					for (int tmp_j = -ksize; tmp_j <= ksize; tmp_j++)
						hist[p_src[j + tmp_j]]+=1;
				}
				nm = 0;
				int element_nums =(2 * ksize  + 1) * (2 * ksize + 1);
				int gray_level = 0;
				for (gray_level = 0; gray_level <= 255; gray_level++)
				{
					if (hist[gray_level] > 0)
						nm += hist[gray_level];
					if (nm >= element_nums / 2)
						break;
				}
				median = gray_level;
				p_dst[j] = (uchar)median;
				continue;
			}
			for (int tmp_i = -ksize; tmp_i <= ksize; tmp_i++)
			{
				const uchar* p_src = src.ptr<uchar>(i + tmp_i);
				hist[p_src[j -ksize-1]] -= 1;

				if (p_src[j - ksize - 1] <= median)
					nm--;
				hist[p_src[j + ksize]] += 1;
				if (p_src[j + ksize] <= median)
					nm++;
			}

			int element_nums = (2 * ksize + 1) * (2 * ksize + 1);
			while (nm > element_nums / 2)
			{
				nm -= hist[median];
				median--;
			}
			while (nm < element_nums / 2)
			{
				median++;
				nm += hist[median];
			}

			p_dst[j] = (uchar)median;
		}
	}
	//medianBlur_8u_O1(src, dst, ksize);
	
}

