// Author: Jiawang Bian, Postdoctoral Researcher
// https://github.com/JiawangBian/GMS-Feature-Matcher/blob/master/include/gms_matcher.h

#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <ctime>
using namespace std;
using namespace cv;

#define THRESH_FACTOR 6



// 8 possible rotation and each one is 3 X 3 
const int mRotationPatterns[8][9] = {
	1,2,3,
	4,5,6,
	7,8,9,

	4,1,2,
	7,5,3,
	8,9,6,

	7,4,1,
	8,5,2,
	9,6,3,

	8,7,4,
	9,5,1,
	6,3,2,

	9,8,7,
	6,5,4,
	3,2,1,

	6,9,8,
	3,5,7,
	2,1,4,

	3,6,9,
	2,5,8,
	1,4,7,

	2,3,6,
	1,5,9,
	4,7,8
};

// 5 level scales
const double mScaleRatios[5] = { 1.0, 1.0 / 2, 1.0 / sqrt(2.0), sqrt(2.0), 2.0 };




// ++++++++++++++++++++++++++++ GMS MATCHER ++++++++++++++++++++++++++++++  //
class gms_matcher
{
public:

	/** GMS Matcher
	* @pre       Two valid images have been opened, and their keypoints
	*            have been matched.
	* @post      Normalizes the matches. Creates a grid.
	* @param	 vkp1 is the keypoints from image 1
	* @param     size1 is the dimensions of image 1.
	* @param	 vkp2 is the keypoints from image 2.
	* @param	 size2 is the dimensions of image 2.
	* @param	 vDMatches is the matches between the keypoints,
	*            which is detected through brute force nearest neighbor matching.
	*/
	gms_matcher(const vector<KeyPoint>& vkp1, const Size size1, 
		const vector<KeyPoint>& vkp2, const Size size2, const vector<DMatch>& vDMatches)
	{
		// Input initialize
		NormalizePoints(vkp1, size1, mvP1);
		NormalizePoints(vkp2, size2, mvP2);
		mNumberMatches = vDMatches.size();
		ConvertMatches(vDMatches, mvMatches);

		// Grid initialize
		mGridSizeLeft = Size(20, 20); // The default grid size is 20 by 20
		mGridNumberLeft = mGridSizeLeft.width * mGridSizeLeft.height;

		// Initialize the neighbor of left grid 
		mGridNeighborLeft = Mat::zeros(mGridNumberLeft, 9, CV_32SC1); //rows, columns, and type
		InitalizeNeighbors(mGridNeighborLeft, mGridSizeLeft);
	};
	~gms_matcher() {};

private:

	// Normalized Points - filled during the NormalizePoints function
	vector<Point2f> mvP1, mvP2;

	// Matches - filled with pairs of points during the ConvertMatches function
	vector<pair<int, int> > mvMatches;

	// Number of Matches
	size_t mNumberMatches;

	// Grid Size
	Size mGridSizeLeft, mGridSizeRight; // 20 by 20
	int mGridNumberLeft;
	int mGridNumberRight;

	// x	  : left grid idx
	// y      :  right grid idx
	// value  : how many matches from idx_left to idx_right
	Mat mMotionStatistics;

	// 
	vector<int> mNumberPointsInPerCellLeft;

	// Inldex  : grid_idx_left
	// Value   : grid_idx_right
	vector<int> mCellPairs;

	// Every Matches has a cell-pair 
	// first  : grid_idx_left
	// second : grid_idx_right
	vector<pair<int, int> > mvMatchPairs;

	// Inlier Mask for output
	vector<bool> mvbInlierMask;

	//
	Mat mGridNeighborLeft;
	Mat mGridNeighborRight;

public:

	// Get Inlier Mask
	// Return number of inliers 
	int GetInlierMask(vector<bool>& vbInliers, 
		bool WithScale = false, bool WithRotation = false);

private:
	
	/** Normalize Key Points to Range(0 - 1)
	* @pre       Matching was performed between two images.
	*            DMatch is full of matches.
	* @post      Converts from DMatch to Match pairs of points
	* @param	 kp is the keypoints from one image.
	* @param     size is the dimensions of the image.
	* @param	 npts will be filled with normalized points from the image.
	*/
	void NormalizePoints(const vector<KeyPoint>& kp, 
		const Size& size, vector<Point2f>& npts) {
		
		const size_t numP = kp.size();
		const int width = size.width;
		const int height = size.height;
		npts.resize(numP);

		for (size_t i = 0; i < numP; i++)
		{
			npts[i].x = kp[i].pt.x / width;
			npts[i].y = kp[i].pt.y / height;
		}
	}

	/** Convert OpenCV DMatch to Match (pair<int, int>)
	* @pre       Matching was performed between two images.
	*            DMatch is full of matches.
	* @post      Converts from DMatch to Match pairs of points
	* @param	 vDMatches is a vector of matches from brute force matching.
	*            It contains query and train indexes.
	* @param     vMatches is a vector to be filled with pairs of points
	*/
	void ConvertMatches(const vector<DMatch>& vDMatches, vector<pair<int, int> >& vMatches) {
		
		vMatches.resize(mNumberMatches);
		for (size_t i = 0; i < mNumberMatches; i++)
		{
			//Fill vMatches with pairs of points from vDMatches
			vMatches[i] = pair<int, int>(vDMatches[i].queryIdx, vDMatches[i].trainIdx);
		}
	}

	/** Shift the grid a half cell width in the x, y, and xy directions.
	* @pre       mv1 and mvMatches have been filled.
	*			 A valid point is passed to the function.
	* @post      Return the index of the leftmost part of the grid.
	* @param	 pt is the left point (x, y) coordinates.
	* @param     type is the orientation (x, y, or xy) to shift the grid over
	*/
	int GetGridIndexLeft(const Point2f& pt, int type) {
		int x = 0, y = 0;

		//NO SHIFTING
		if (type == 1) {
			x = floor(pt.x * mGridSizeLeft.width);
			y = floor(pt.y * mGridSizeLeft.height);

			if (y >= mGridSizeLeft.height || x >= mGridSizeLeft.width) {
				return -1;
			}
		}

		//SHIFT IN X DIRECTION
		if (type == 2) {
			x = floor(pt.x * mGridSizeLeft.width + 0.5);
			y = floor(pt.y * mGridSizeLeft.height);

			if (x >= mGridSizeLeft.width || x < 1) {
				return -1;
			}
		}

		//SHIFT IN THE Y DIRECTION
		if (type == 3) {
			x = floor(pt.x * mGridSizeLeft.width);
			y = floor(pt.y * mGridSizeLeft.height + 0.5);

			if (y >= mGridSizeLeft.height || y < 1) {
				return -1;
			}
		}

		//SHIFT IN THE X AND Y DIRECTION
		if (type == 4) {
			x = floor(pt.x * mGridSizeLeft.width + 0.5);
			y = floor(pt.y * mGridSizeLeft.height + 0.5);

			if (y >= mGridSizeLeft.height || y < 1 || x >= mGridSizeLeft.width || x < 1) {
				return -1;
			}
		}

		return x + y * mGridSizeLeft.width;
	}

	int GetGridIndexRight(const Point2f& pt) {
		int x = floor(pt.x * mGridSizeRight.width);
		int y = floor(pt.y * mGridSizeRight.height);

		return x + y * mGridSizeRight.width;
	}

	// Assign Matches to Cell Pairs 
	void AssignMatchPairs(int GridType);

	// Verify Cell Pairs
	void VerifyCellPairs(int RotationType);

	// Get Neighbor 9
	vector<int> GetNB9(const int idx, const Size& GridSize) {
		vector<int> NB9(9, -1);

		int idx_x = idx % GridSize.width;
		int idx_y = idx / GridSize.width;

		for (int yi = -1; yi <= 1; yi++)
		{
			for (int xi = -1; xi <= 1; xi++)
			{
				int idx_xx = idx_x + xi;
				int idx_yy = idx_y + yi;

				if (idx_xx < 0 || idx_xx >= GridSize.width || idx_yy < 0 || idx_yy >= GridSize.height)
					continue;

				NB9[xi + 4 + yi * 3] = idx_xx + idx_yy * GridSize.width;
			}
		}
		return NB9;
	}

	/** 
	* @pre       
	* @post      
	* @param	 neighbor is
	* @param     GridSize is
	*/
	void InitalizeNeighbors(Mat& neighbor, const Size& GridSize) {
		for (int i = 0; i < neighbor.rows; i++)
		{
			vector<int> NB9 = GetNB9(i, GridSize);
			int* data = neighbor.ptr<int>(i);
			memcpy(data, &NB9[0], sizeof(int) * 9);
		}
	}

	/** 
	* @pre
	* @post
	* @param	 Scale is 
	*/
	void SetScale(int Scale) {
		// Set Scale
		mGridSizeRight.width = mGridSizeLeft.width * mScaleRatios[Scale];
		mGridSizeRight.height = mGridSizeLeft.height * mScaleRatios[Scale];
		mGridNumberRight = mGridSizeRight.width * mGridSizeRight.height;

		// Initialize the neighbor of right grid 
		mGridNeighborRight = Mat::zeros(mGridNumberRight, 9, CV_32SC1);
		InitalizeNeighbors(mGridNeighborRight, mGridSizeRight);
	}

	// Run 
	int run(int RotationType);
};

int gms_matcher::GetInlierMask(vector<bool>& vbInliers, bool WithScale, bool WithRotation) {

	int max_inlier = 0;

	if (!WithScale && !WithRotation)
	{
		SetScale(0);
		max_inlier = run(1);
		vbInliers = mvbInlierMask;
		return max_inlier;
	}

	if (WithRotation && WithScale)
	{
		for (int Scale = 0; Scale < 5; Scale++)
		{
			SetScale(Scale);
			for (int RotationType = 1; RotationType <= 8; RotationType++)
			{
				int num_inlier = run(RotationType);

				if (num_inlier > max_inlier)
				{
					vbInliers = mvbInlierMask;
					max_inlier = num_inlier;
				}
			}
		}
		return max_inlier;
	}

	if (WithRotation && !WithScale)
	{
		SetScale(0);
		for (int RotationType = 1; RotationType <= 8; RotationType++)
		{
			int num_inlier = run(RotationType);

			if (num_inlier > max_inlier)
			{
				vbInliers = mvbInlierMask;
				max_inlier = num_inlier;
			}
		}
		return max_inlier;
	}

	if (!WithRotation && WithScale)
	{
		for (int Scale = 0; Scale < 5; Scale++)
		{
			SetScale(Scale);

			int num_inlier = run(1);

			if (num_inlier > max_inlier)
			{
				vbInliers = mvbInlierMask;
				max_inlier = num_inlier;
			}

		}
		return max_inlier;
	}

	return max_inlier;
}

void gms_matcher::AssignMatchPairs(int GridType) {

	for (size_t i = 0; i < mNumberMatches; i++)
	{
		Point2f& lp = mvP1[mvMatches[i].first];
		Point2f& rp = mvP2[mvMatches[i].second];

		//
		int lgidx = mvMatchPairs[i].first = GetGridIndexLeft(lp, GridType);
		int rgidx = -1;

		if (GridType == 1)
		{
			rgidx = mvMatchPairs[i].second = GetGridIndexRight(rp);
		}
		else
		{
			rgidx = mvMatchPairs[i].second;
		}

		if (lgidx < 0 || rgidx < 0)	continue;

		mMotionStatistics.at<int>(lgidx, rgidx)++;
		mNumberPointsInPerCellLeft[lgidx]++;
	}

}

void gms_matcher::VerifyCellPairs(int RotationType) {

	const int* CurrentRP = mRotationPatterns[RotationType - 1];

	for (int i = 0; i < mGridNumberLeft; i++)
	{
		if (sum(mMotionStatistics.row(i))[0] == 0)
		{
			mCellPairs[i] = -1;
			continue;
		}

		int max_number = 0;
		for (int j = 0; j < mGridNumberRight; j++)
		{
			int* value = mMotionStatistics.ptr<int>(i);
			if (value[j] > max_number)
			{
				mCellPairs[i] = j;
				max_number = value[j];
			}
		}

		int idx_grid_rt = mCellPairs[i];

		const int* NB9_lt = mGridNeighborLeft.ptr<int>(i);
		const int* NB9_rt = mGridNeighborRight.ptr<int>(idx_grid_rt);

		int score = 0;
		double thresh = 0;
		int numpair = 0;

		for (size_t j = 0; j < 9; j++)
		{
			int ll = NB9_lt[j];
			int rr = NB9_rt[CurrentRP[j] - 1];
			if (ll == -1 || rr == -1)	continue;

			score += mMotionStatistics.at<int>(ll, rr);
			thresh += mNumberPointsInPerCellLeft[ll];
			numpair++;
		}

		thresh = THRESH_FACTOR * sqrt(thresh / numpair);

		if (score < thresh)
			mCellPairs[i] = -2;
	}
}

int gms_matcher::run(int RotationType) {

	mvbInlierMask.assign(mNumberMatches, false);

	// Initialize Motion Statisctics
	mMotionStatistics = Mat::zeros(mGridNumberLeft, mGridNumberRight, CV_32SC1);
	mvMatchPairs.assign(mNumberMatches, pair<int, int>(0, 0));

	for (int GridType = 1; GridType <= 4; GridType++)
	{
		// initialize
		mMotionStatistics.setTo(0);
		mCellPairs.assign(mGridNumberLeft, -1);
		mNumberPointsInPerCellLeft.assign(mGridNumberLeft, 0);

		AssignMatchPairs(GridType);
		VerifyCellPairs(RotationType);

		// Mark inliers
		for (size_t i = 0; i < mNumberMatches; i++)
		{
			if (mvMatchPairs[i].first >= 0) {
				if (mCellPairs[mvMatchPairs[i].first] == mvMatchPairs[i].second)
				{
					mvbInlierMask[i] = true;
				}
			}
		}
	}
	int num_inlier = sum(mvbInlierMask)[0];
	return num_inlier;
}