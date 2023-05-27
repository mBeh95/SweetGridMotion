// 
// Original Author: Jiawang Bian, Postdoctoral Researcher
// 
// We refactored the following names to be more descriptive:
//		"mvP1" as "normalizedPoints1" and "mvP2" as "normalizedPoins2"
//      "mvMatches" and "vMatches" (in ConvertMatches) to be "initialMatches"
//		"mGridNumberLeft" as "totalNumberOfCellsLeft"
//      "mGridNumberRight" as "totalNumberOfCellsRight"
// 
// We considered renaming "mGridSizeRight" and "mGridSizeLeft", but
//      just remember that any time you see "left" it is referring to the 
//      first image / first grid and any time you see "right" it is referring
//      to the second image / second grid.
// 
//		For the scale and rotation changes, the grid of the first image 
//      remains fixed in place, while the grid of the second image 
//      (the "right" image) will alter according to scale and rotation changes.
// 
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
	*            Initializes the neighbors
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
		// Input initialization
		NormalizePoints(vkp1, size1, normalizedPoints1); //Fills normalizedPoints1
		NormalizePoints(vkp2, size2, normalizedPoints2); //Fills normalizedPoints2
		mNumberMatches = vDMatches.size();		//How many matches were found?
		ConvertMatches(vDMatches, initialMatches);	//Fill initialMatches with pairs of points

		// Grid size initialization
		mGridSizeLeft = Size(20, 20); // The default grid size for the first image is 20 by 20
		
		// Total number of cells in the grid
		totalNumberOfCellsLeft = mGridSizeLeft.width * mGridSizeLeft.height; 

		// Initialize the neighbors of the left grid / image
		// The zeros function takes in the number of rows, columns, and data type
		// and fills the matrix with 0s.
		mGridNeighborLeft = Mat::zeros(totalNumberOfCellsLeft, 9, CV_32SC1);
		InitalizeNeighbors(mGridNeighborLeft, mGridSizeLeft);
	};
	~gms_matcher() {};

private:

	// Normalized Points - filled during the NormalizePoints function
	vector<Point2f> normalizedPoints1, normalizedPoints2;

	// Matches - filled with pairs of points during the ConvertMatches function
	vector<pair<int, int> > initialMatches;

	// Number of Matches
	size_t mNumberMatches;

	// Grid Size - 20 by 20
	Size mGridSizeLeft, mGridSizeRight; // 20 by 20
	int totalNumberOfCellsLeft;
	int totalNumberOfCellsRight;

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
	Mat mGridNeighborLeft; //Initialized in the GMS constructor
	Mat mGridNeighborRight; //Initialized in the SetScale function

public:

	// Get Inlier Mask
	// Return number of inliers 
	int GetInlierMask(vector<bool>& vbInliers, 
		bool WithScale = false, bool WithRotation = false);

private:
	
	/** Normalize Key Points to Range (0 - 1)
	* @pre       Matching was performed between two images.
	*            This method will be called twice,
	*            to normalize the points for both images.
	* @post      normalize the points between 0 and 1
	*			 and fill one of the two normalizedPoints vectors.
	* @param	 kp is the keypoints from one image.
	* @param     size is the dimensions of one image.
	* @param	 npts will be filled with normalized points from one image.
	*/
	void NormalizePoints(const vector<KeyPoint>& kp, 
		const Size& size, vector<Point2f>& npts) {
		
		const size_t numP = kp.size();  //How many keypoints were there?
		const int width = size.width;   //What was the width of the image?
		const int height = size.height; //What was the heigth of the image?
		npts.resize(numP);              //Resize the normalizedPoints vector to be the same
		                                // size as the original keypoint vector

		for (size_t i = 0; i < numP; i++)
		{
			npts[i].x = kp[i].pt.x / width;	  //Fill one of the normalizedPoints vectors
			npts[i].y = kp[i].pt.y / height;  //Fill one of the normalizedPoints vectors
		}
	}

	/** Convert OpenCV DMatch to Match (pair<int, int>)
	* @pre       Brute force matching was performed between two images.
	*            DMatch is full of matches.
	* @post      Converts from a DMatch vector to a vector of <pair<int, int>> of points
	*            so that the algorithm can use pairs of points instead.
	* @param	 vDMatches is a vector of matches from the brute force matching.
	*            It contains query and train indexes.
	* @param     initialMatches is a vector to be filled with pairs of points
	*/
	void ConvertMatches(const vector<DMatch>& vDMatches, vector<pair<int, int> >& initialMatches) {
		
		initialMatches.resize(mNumberMatches);
		for (size_t i = 0; i < mNumberMatches; i++)
		{
			//Fill initialMatches with pairs of points from vDMatches
			initialMatches[i] = pair<int, int>(vDMatches[i].queryIdx, vDMatches[i].trainIdx);
		}
	}

	/** Shift the grid a half cell width in the x, y, and xy directions.
	* @pre       normalizedPoints1 and normalizedPoints2 and initialMatches have been filled.
	*			 A valid point is passed to the function.
	* @post      Return the index of the left image.
	*            
	* @param	 pt is the left point (x, y) coordinates.
	* @param     type is the orientation (x, y, or xy) to shift the grid over
	* @return    x + y * mGridSizeLeft.width
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

		//Return the index of the leftmost point of the grid
		return x + y * mGridSizeLeft.width;
	}

	/** Return the
	* @pre       normalizedPoints1 and normalizedPoints2 and initialMatches have been filled.
	*			 A valid point is passed to the function.
	* @post      Return the index of the rightmost part of the grid.
	* @param	 pt is the right point (x, y) coordinates.
	*/
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
		totalNumberOfCellsRight = mGridSizeRight.width * mGridSizeRight.height;

		// Initialize the neighbor of right grid 
		mGridNeighborRight = Mat::zeros(totalNumberOfCellsRight, 9, CV_32SC1);
		InitalizeNeighbors(mGridNeighborRight, mGridSizeRight);
	}

	// Run 
	int run(int RotationType);
};

/** 
* @pre
* @post      Run either without scale or rotation, 
*            with scale, with rotation, with both scale and rotation.
* @param	 
* @param     
* @return    return the max_inlier
*/
int gms_matcher::GetInlierMask(vector<bool>& vbInliers, bool WithScale, bool WithRotation) {

	int max_inlier = 0;

	if (!WithScale && !WithRotation)
	{
		SetScale(0);				//No scaling
		max_inlier = run(1);		//run(1) indicates no rotation
		vbInliers = mvbInlierMask;
		return max_inlier;
	}

	if (WithRotation && WithScale)
	{

		//REPEAT FOR ALL 5 SCALES
		for (int Scale = 0; Scale < 5; Scale++)
		{
			SetScale(Scale);

			//REPEAT FOR ALL 8 ROTATION TYPES
			for (int RotationType = 1; RotationType <= 8; RotationType++)
			{
				int num_inlier = run(RotationType);

				if (num_inlier > max_inlier)
				{
					//Set the max_inlier
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

			int num_inlier = run(1); //run(1) indicates no rotation

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


/**
* @pre       
* @post      The right grid index is initialized to -1;
*            The left grid index is initialized to 
* @param     GridType is determined by how the grid is shifted
*            to ensure that keypoints that fall on the grid border
*            of the original grid are not excluded
* @return    
*/
void gms_matcher::AssignMatchPairs(int GridType) {

	for (size_t i = 0; i < mNumberMatches; i++)
	{
		Point2f& lp = normalizedPoints1[initialMatches[i].first];
		Point2f& rp = normalizedPoints2[initialMatches[i].second];

		//Indexes depend on the GridType
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

/**
* @pre
* @post
* @param     RotationType is one of 8 rotation patterns.
* @return
*/
void gms_matcher::VerifyCellPairs(int RotationType) {

	//Set the rotation pattern
	const int* CurrentRP = mRotationPatterns[RotationType - 1];

	for (int i = 0; i < totalNumberOfCellsLeft; i++)
	{
		if (sum(mMotionStatistics.row(i))[0] == 0)
		{
			mCellPairs[i] = -1;
			continue;
		}

		int max_number = 0;
		for (int j = 0; j < totalNumberOfCellsRight; j++)
		{
			int* value = mMotionStatistics.ptr<int>(i);
			if (value[j] > max_number)
			{
				//Set the maximum
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
			//Use the motion kernel (2020 paper, page 1584)
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

/** RUN GMS
* @pre
* @post      All inliers in mvbInlierMask
*            will be initialized to false.
*            As the algorithm goes through each iteration,
*            more inliers are found and added.
* @param     RotationType is one of 8 rotation patterns.
*            This is needed for the VerifyCellPairs method.
* @return    The number of inliers
*/
int gms_matcher::run(int RotationType) {

	// Initialize all matches to false at first
	mvbInlierMask.assign(mNumberMatches, false);

	// Initialize Motion Statisctics
	mMotionStatistics = Mat::zeros(totalNumberOfCellsLeft, totalNumberOfCellsRight, CV_32SC1);
	mvMatchPairs.assign(mNumberMatches, pair<int, int>(0, 0));

	// Repeat for each of the 4 grid types -- original, shifted x, shifted y, shifted xy
	for (int GridType = 1; GridType <= 4; GridType++)
	{
		// initialize
		mMotionStatistics.setTo(0);
		mCellPairs.assign(totalNumberOfCellsLeft, -1);
		mNumberPointsInPerCellLeft.assign(totalNumberOfCellsLeft, 0);

		AssignMatchPairs(GridType);
		VerifyCellPairs(RotationType);

		// Mark inliers
		for (size_t i = 0; i < mNumberMatches; i++)
		{
			if (mvMatchPairs[i].first >= 0) {
				if (mCellPairs[mvMatchPairs[i].first] == mvMatchPairs[i].second)
				{
					// By setting the inlier mask to false initially,
					// only true matches will be found.
					mvbInlierMask[i] = true;
				}
			}
		}
	}
	int num_inlier = sum(mvbInlierMask)[0];
	return num_inlier;
}