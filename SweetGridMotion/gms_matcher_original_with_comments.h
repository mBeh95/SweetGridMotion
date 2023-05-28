// 
// Original Author: Jiawang Bian, Postdoctoral Researcher
// 
// The constructor and GetInlierMask are public
// all other methods are private.
// 
// We refactored the following names to be more descriptive:
//		"mvP1" as "normalizedPoints1" and "mvP2" as "normalizedPoins2"
//      "mvMatches" and "vMatches" (in ConvertMatches) to be "initialMatches"
//		"mGridNumberLeft" as "totalNumberOfCellsLeft"
//      "mGridNumberRight" as "totalNumberOfCellsRight"
//      "type" to be "GridType" in the GetGridIndexLeft function
//      "InitalizeNeighbors" to "InitializeNeighbors" (fixed typo)
//      "vbInliers" to "inliersToReturn"
// 
// We considered renaming "mGridSizeRight" and "mGridSizeLeft" but decided to keep them the same.
//      Just remember that any time you see 
//      "left" it is referring to the first image / first grid
//      and any time you see 
//      "right" it is referring to the second image / second grid.
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

// +++++++++++++++++++++++++++++++ CONSTANTS +++++++++++++++++++++++++++++++//
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
	* @post      Normalizes the keypoints between the images. 
	*            Converts the matches to pairs of points.
	*            Creates one grid per image.
	*            Initializes the neighbor vectors for both grids.
	* @param	 vkp1 is the keypoints from image 1 (left image)
	* @param     size1 is the dimensions of image 1 (left image).
	* @param	 vkp2 is the keypoints from image 2 (right image).
	* @param	 size2 is the dimensions of image 2 (right image).
	* @param	 vDMatches is the matches between the keypoints,
	*            which is detected through brute force nearest neighbor matching.
	*/
	gms_matcher(const vector<KeyPoint>& vkp1, const Size size1, 
		const vector<KeyPoint>& vkp2, const Size size2, const vector<DMatch>& vDMatches)
	{
		// Input initialization (keypoints and matches)
		NormalizePoints(vkp1, size1, normalizedPoints1);	//Fills normalizedPoints1
		NormalizePoints(vkp2, size2, normalizedPoints2);	//Fills normalizedPoints2
		mNumberMatches = vDMatches.size();					//How many matches were found?
		ConvertMatches(vDMatches, initialMatches);			//Fill initialMatches with pairs of points

		// Grid size initialization
		// Note that mGridSizeLeft has a width and a height
		mGridSizeLeft = Size(20, 20); // The default grid size for the first image is 20 by 20
		
		// Total number of cells in the grid (20 * 20)
		totalNumberOfCellsLeft = mGridSizeLeft.width * mGridSizeLeft.height; 

		// The mGridNeighborLeft matrix is size 400 by 9 by default
		// The zeros function takes in the number of rows, columns, and data type
		// and fills the matrix with 0s.
		mGridNeighborLeft = Mat::zeros(totalNumberOfCellsLeft, 9, CV_32SC1);

		// Fill in the matrixes of the 400 by 9 cells with indexes to the neighbors per cell
		InitializeNeighbors(mGridNeighborLeft, mGridSizeLeft);
	};

	//Destructor
	~gms_matcher() {};

private:

	// Normalized Points - filled during the NormalizePoints function
	vector<Point2f> normalizedPoints1, normalizedPoints2;

	// Matches - filled with pairs of points during the ConvertMatches function
	vector<pair<int, int> > initialMatches;

	// The original number of matches found between two images - initialized from the size of vDMatches 
	size_t mNumberMatches;

	// Grid Size - 20 by 20
	// Note: left is the first image; right is the second image
	// mGridSizeLeft has a width and a height -- 20 by 20 by default
	// mGridSizeRight has a width and a height too -- 20 by 20
	Size mGridSizeLeft, mGridSizeRight;

	// How many cells total are in the left image's grid?
	int totalNumberOfCellsLeft;

	// How many cells total are in the right image's grid?
	int totalNumberOfCellsRight;

	// All possible neighbors for all possible cells in each grid (left and right grid / image)
	Mat mGridNeighborLeft; //Initialized in the GMS constructor - 400 by 9 matrix
	Mat mGridNeighborRight; //Initialized in the SetScale function from GetInlierMask - depends on scale

	// x	  : left grid idx
	// y      : right grid idx
	// value  : how many matches from idx_left to idx_right
	// Note   : incremented in the AssignMatchPairs function
	Mat mMotionStatistics;

	// incremented in the AssignMatchPairs function
	vector<int> mNumberPointsInPerCellLeft;

	// mCellPairs - a one-dimensional vector that holds an index to the RIGHT image if there is a match.
	// If the value is -1 there were NO MATCHES between the left and the right grids for this cell.
	// If the value is NOT -1, there was a match between the left and the right grids for this cell.
	// Index  : grid_idx_left - mCellPairs[i] is the grid index from the LEFT image
	// Value  : grid_idx_right - mCellPairs[i] = j is the grid index from the RIGHT image (or -1 if no matches)
	// Size   : the total number of cells in the grid
	vector<int> mCellPairs;

	// Every match between two points has a corresponding cell-pair too
	// This is initialized in the AssignMatchPairs function
	// first  : grid_idx_left - mvMatchPairs[i].first = LEFT
	// second : grid_idx_right - mvMatchPairs[i].second = RIGHT
	// Size   : the total number of matches found initially
	vector<pair<int, int> > mvMatchPairs;

	// Inlier Mask for output
	// Size   : the total number of matches found initially
	vector<bool> mvbInlierMask;



public:

	// TODO: ADD THE COMMENTS ABOVE ALL DECLARATIONS
	int GetInlierMask(vector<bool>& inliersToReturn, 
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
		
		const size_t numP = kp.size();  // How many keypoints were there?
		const int width = size.width;   // What was the width of the image?
		const int height = size.height; // What was the heigth of the image?
		npts.resize(numP);              // Resize the normalizedPoints vector to be the same
		                                // size as the original keypoint vector

		for (size_t i = 0; i < numP; i++)
		{
			npts[i].x = kp[i].pt.x / width;	  // Fill one of the normalizedPoints vectors
			npts[i].y = kp[i].pt.y / height;  // Fill one of the normalizedPoints vectors
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

	/** Return the starting index for the left grid / image
	* @pre       normalizedPoints1 and normalizedPoints2 and initialMatches have been filled.
	*			 A valid point is passed to the function.
	* @post      Shift the left image's grid a half cell width in the x, y, and xy directions,
	*            depending on the GridType
	*		     Return the starting index of the left image's grid.
	* @param	 pt is the left point (x, y) coordinates.
	* @param     GridType is the direction (x, y, or xy) to shift the grid over
	* @return    x + y * mGridSizeLeft.width
	*/
	int GetGridIndexLeft(const Point2f& pt, int GridType) {
		int x = 0, y = 0;

		//NO SHIFTING
		if (GridType == 1) {
			x = floor(pt.x * mGridSizeLeft.width);
			y = floor(pt.y * mGridSizeLeft.height);

			if (y >= mGridSizeLeft.height || x >= mGridSizeLeft.width) {
				return -1;
			}
		}

		//SHIFT IN X DIRECTION
		if (GridType == 2) {
			x = floor(pt.x * mGridSizeLeft.width + 0.5);
			y = floor(pt.y * mGridSizeLeft.height);

			if (x >= mGridSizeLeft.width || x < 1) {
				return -1;
			}
		}

		//SHIFT IN THE Y DIRECTION
		if (GridType == 3) {
			x = floor(pt.x * mGridSizeLeft.width);
			y = floor(pt.y * mGridSizeLeft.height + 0.5);

			if (y >= mGridSizeLeft.height || y < 1) {
				return -1;
			}
		}

		//SHIFT IN THE X AND Y DIRECTION
		if (GridType == 4) {
			x = floor(pt.x * mGridSizeLeft.width + 0.5);
			y = floor(pt.y * mGridSizeLeft.height + 0.5);

			if (y >= mGridSizeLeft.height || y < 1 || x >= mGridSizeLeft.width || x < 1) {
				return -1;
			}
		}

		//Return the index of the leftmost point of the grid
		return x + y * mGridSizeLeft.width;
	}

	/** Return the starting index for the right grid / image
	* @pre       normalizedPoints1 and normalizedPoints2 and initialMatches have been filled.
	*			 A valid point is passed to the function.
	* @post      Return the starting index of the right image's grid.
	* @param	 pt is the right point (x, y) coordinates.
	*/
	int GetGridIndexRight(const Point2f& pt) {
		int x = floor(pt.x * mGridSizeRight.width);
		int y = floor(pt.y * mGridSizeRight.height);

		return x + y * mGridSizeRight.width;
	}

	// TODO: ADD THE COMMENTS ABOVE ALL DECLARATIONS
	void AssignMatchPairs(int GridType);

	// TODO: ADD THE COMMENTS ABOVE ALL DECLARATIONS
	void VerifyCellPairs(int RotationType);

	/** Get Neighbor 9
	* @pre       There is a grid on an image.
	*            InitializeNeighbors calls this function.
	* @post      Fill in NB9 with indexes for the neighbors for one cell.
	* @param	 idx is the index of ONE CELL in the grid.
	* @param     GridSize is the dimensions of the grid (20 by 20)
	*/
	vector<int> GetNB9(const int idx, const Size& GridSize) {

		//A vector of 9 slots filled with -1's
		vector<int> NB9(9, -1); 

		//Find out what cell to look at within the 20 by 20 grid
		int idx_x = idx % GridSize.width; //What part of the grid - in the x dimension?
		int idx_y = idx / GridSize.width; //What part of the grid - in the y dimension?

		//Repeat for yi equals -1, 0, and 1
		for (int yi = -1; yi <= 1; yi++)
		{
			//Repeat for xi equals -1, 0, and 1
			for (int xi = -1; xi <= 1; xi++)
			{

				//Look left, center, right and up, center, down for each cell
				int idx_xx = idx_x + xi; // -1 would be left; 0 is center; 1 is right
				int idx_yy = idx_y + yi; // -1 would be up; 0 is center; 1 is down

				//Make sure you do not go out of bounds
				if (idx_xx < 0 || 
					idx_xx >= GridSize.width || 
					idx_yy < 0 || 
					idx_yy >= GridSize.height)
					continue;

				// Fill in the NB9 vector
				// When xi is -1 and yi is -1, this indexes to NB9[0]
				// When xi is  0 and yi is -1, this indexes to NB9[1]
				// When xi is  1 and yi is -1, this indexes to NB9[2]
				// When xi is -1 and yi is  0, this indexes to NB9[3]
				// When xi is  0 and yi is  0, this indexes to NB9[4]
				// When xi is  1 and yi is  0, this indexes to NB9[5]
				// When xi is -1 and yi is  1, this indexes to NB9[6]
				// When xi is  0 and yi is  1, this indexes to NB9[7]
				// When xi is  1 and yi is  1, this indexes to NB9[8]
				NB9[xi + 4 + yi * 3] = idx_xx + idx_yy * GridSize.width;
			}
		}
		return NB9;
	}

	/** Initialize the neighbor matrices.
	* @pre       The GridSize is known.
	*            This function will be called twice, 
	*			 once for the left image and once for the right image.
	* @post      Fill the neighbors matrices with indexes to the neighbors for each cell.
	* @param	 neighbor is the matrix of neighbors (400 by 9) for one grid / image
	* @param     GridSize is the dimensions of one grid 
	*/
	void InitializeNeighbors(Mat& neighbor, const Size& GridSize) {

		//Repeat for ALL CELLS in the grid (400 cells if 20 by 20)
		for (int i = 0; i < neighbor.rows; i++)
		{
			// Grab the neighbor indexes for the cell
			vector<int> NB9 = GetNB9(i, GridSize);

			// The data pointer points to the neighbor for 
			int* data = neighbor.ptr<int>(i);

			// data is the destination; NB9 is the source to copy over
			// Fill the neighbor vector with the indexes of all its neighbors
			memcpy(data, &NB9[0], sizeof(int) * 9);
		}
	}

	/** Set the scale for image 2 (the right image)
	* @pre		 Image 1, the left image has a grid and 
	*            This is called within the GetInlierMask function
	*            to make sure that 5 different scales are tried.
	* @post      Initialize the neighbor vector for the right image.
	*            In other words, fill the mGridNeighborRight vector.
	* @param	 Scale is one of 5 possible scales.
	*/
	void SetScale(int Scale) {

		// Set Scale
		mGridSizeRight.width = mGridSizeLeft.width * mScaleRatios[Scale];
		mGridSizeRight.height = mGridSizeLeft.height * mScaleRatios[Scale];
		totalNumberOfCellsRight = mGridSizeRight.width * mGridSizeRight.height;

		// Initialize the neighbors of right grid 
		mGridNeighborRight = Mat::zeros(totalNumberOfCellsRight, 9, CV_32SC1);
		InitializeNeighbors(mGridNeighborRight, mGridSizeRight);
	
	}

	// TODO: ADD THE COMMENTS ABOVE ALL DECLARATIONS
	int run(int RotationType);

};

/** Get the inliers between two images
* @pre       The GetInlierMask public method is called.
* 
* @post      This public method will run GMS.
*            Depending on the settings provided when the GetInlierMask is called,
*            this will either run without scale or rotation, 
*            with scale OR with rotation, or with BOTH scale AND rotation,
*
*            Fill the inliersToReturn vector with true correspondences.
*            Return the count of inliers found.
* 
* @param	 inliersToReturn is the true correspondences between the images
* @param     WithScale if true indicates the 2nd image is scaled
* @param     WithRotation if true indicates the 2nd image is rotated
* @return    return the max_inlier (count of inliers found)
*/
int gms_matcher::GetInlierMask(vector<bool>& inliersToReturn, bool WithScale, bool WithRotation) {

	int max_inlier = 0;

	if (!WithScale && !WithRotation)
	{
		SetScale(0);						//SetScale(0) indicates NO scaling
		max_inlier = run(1);				//run(1) indicates no rotation
		inliersToReturn = mvbInlierMask;
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
					inliersToReturn = mvbInlierMask;
					max_inlier = num_inlier;
				}
			}
		}
		return max_inlier;
	}

	if (WithRotation && !WithScale)
	{
		SetScale(0);				//SetScale(0) indicates NO scaling

		for (int RotationType = 1; RotationType <= 8; RotationType++)
		{
			int num_inlier = run(RotationType);

			if (num_inlier > max_inlier)
			{
				inliersToReturn = mvbInlierMask;
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
				inliersToReturn = mvbInlierMask;
				max_inlier = num_inlier;
			}

		}
		return max_inlier;
	}

	return max_inlier;
}


/** Assign Match Pairs
* @pre       The public GetInlierMask function called the run function,
*            which called this function.
* @post      Get the grid indexes for the pairs of points in every match.
*            Fill the mMotionStatistics and mNumberPointsInPerCellLeft vectors.
*            Fill in the mvMatchPairs[i].first and mvMatchPairs[i].second points.
* @param     GridType is determined by how the grid is shifted
*            to ensure that keypoints that fall on the grid border
*            of the original grid are not excluded.
*/
void gms_matcher::AssignMatchPairs(int GridType) {

	//For all the initial matches between the two images (including incorrect matches)
	for (size_t i = 0; i < mNumberMatches; i++)
	{
		// Look at one pair of normalized points from the left and right images
		Point2f& lp = normalizedPoints1[initialMatches[i].first];
		Point2f& rp = normalizedPoints2[initialMatches[i].second];

		// Get the grid index for that pair of points.
		// Index locations depend on the GridType.
		// Get the grid index for the left point (.first indicates LEFT)
		// Simultaneously, set mvMatchPairs[i].first
		int lgidx = mvMatchPairs[i].first = GetGridIndexLeft(lp, GridType);
		int rgidx = -1;

		// GridType == 1 indicates no movement of the grid position
		if (GridType == 1)
		{
			//Get the grid index for the right point (.second indicates RIGHT)
			// Simultaneously, set mvMatchPairs[i].second
			rgidx = mvMatchPairs[i].second = GetGridIndexRight(rp);
		}
		else
		{
			//Get the grid index for the right point from ........
			rgidx = mvMatchPairs[i].second;
		}

		//Ensure that neither index is out of bounds.
		if (lgidx < 0 || rgidx < 0)	continue; 

		// Increment the motion statistics vector for each match found inside those corresponding cells
		mMotionStatistics.at<int>(lgidx, rgidx)++; 

		// Increment the number of matched points per cell for the left image
		mNumberPointsInPerCellLeft[lgidx]++;
	}

}

/** Verify Cell Pairs
* @pre       AssignMatchPairs was called to fill the mMotionStatistics and mNumberPointsInPerCellLeft vectors.
* @post      Sets mCellPairs to -1 if no matches were found between a cell in the left and a cell in the right.
*            Sets mCellPairs[i] to j (the index of the cell in the right image) if there is a match.
* @param     RotationType is one of 8 rotation patterns.
*/
void gms_matcher::VerifyCellPairs(int RotationType) {

	// Set the rotation pattern
	const int* CurrentRP = mRotationPatterns[RotationType - 1];

	// For all the cells in the left grid
	for (int i = 0; i < totalNumberOfCellsLeft; i++)
	{
		// If there were NO MATCHES here, set mCellPairs to -1 and try the next cell.
		// (Note: row is looking at one match between the left and right image)
		if (sum(mMotionStatistics.row(i))[0] == 0)
		{
			mCellPairs[i] = -1; // Set the index to 0; no matches were found
			continue;
		}

		int max_number = 0;

		// For all cells in the right grid
		for (int j = 0; j < totalNumberOfCellsRight; j++)
		{
			// Look at the mMotionStatistics vector for this cell
			int* value = mMotionStatistics.ptr<int>(i);

			// If there is a match between the left and right grids ...
			if (value[j] > max_number)
			{
				// For the grid pair i, j
				// Set the value of mCellPairs[i] to equal the index j from the 2nd grid.
				mCellPairs[i] = j;     
				max_number = value[j]; // Set the new maximum
			}
		}

		// Get the index within the right grid
		int idx_grid_rt = mCellPairs[i];

		// Get the indexes of the neighbor cells surrounding that cell
		const int* NB9_lt = mGridNeighborLeft.ptr<int>(i);
		const int* NB9_rt = mGridNeighborRight.ptr<int>(idx_grid_rt);

		int score = 0;		// Motion statistics score (from LEFT and RIGHT images)
		double thresh = 0;	// Threshold (just in the LEFT image)
		int numpair = 0;	// How many cells were there that contained matches?

		// For each of the 9 neighbors
		for (size_t j = 0; j < 9; j++)
		{
			// Get the index for the left image
			int ll = NB9_lt[j];

			// For the right image, grab the neighbor indexes (in case of rotation changes)
			int rr = NB9_rt[CurrentRP[j] - 1];

			// Check to make sure the indexes are not out of bounds
			if (ll == -1 || rr == -1)	continue;

			// Increment the score, using the number of matches found within that cell (from mMotionStatistics)
			// and all the neighboring cells around it within both the LEFT and the RIGHT images.
			score += mMotionStatistics.at<int>(ll, rr);

			// The threshold is a function of how many matches were found within that cell
			// and all the neighboring cells around it within the LEFT image alone.
			thresh += mNumberPointsInPerCellLeft[ll];

			numpair++; // Counts the number of cells that did contain matches
		}

		// Evaluate the threshold
		thresh = THRESH_FACTOR * sqrt(thresh / numpair);

		// Bad match
		if (score < thresh)
			mCellPairs[i] = -2;
	}
}

/** RUN GMS
* @pre       run is called from the public GetInlierMask function.
* @post      All inliers in mvbInlierMask
*            will be initialized to false.
*            As the algorithm goes through each iteration,
*            more inliers are found and added.
*            This calls the AssignMatchPairs and VerifyCellPairs functions.
* @param     RotationType is one of 8 rotation patterns.
*            This is needed for the VerifyCellPairs method.
* @return    The number of inliers
*/
int gms_matcher::run(int RotationType) {

	// Initialize all matches to false at first
	mvbInlierMask.assign(mNumberMatches, false);

	// Initialize mMotionStatistics to 0s for 400 by 400 cells
	mMotionStatistics = Mat::zeros(totalNumberOfCellsLeft, totalNumberOfCellsRight, CV_32SC1);

	// Initialize mvMatchPairs to 0s for each set of matches
	mvMatchPairs.assign(mNumberMatches, pair<int, int>(0, 0));

	// Repeat for each of the 4 grid types -- original, shifted x, shifted y, shifted xy
	for (int GridType = 1; GridType <= 4; GridType++)
	{
		// Set motion statistics vector to all 0s
		mMotionStatistics.setTo(0);

		// Initialize mCellPairs with -1s for all the cells in the grid
		mCellPairs.assign(totalNumberOfCellsLeft, -1);

		// Initialize mNumberPointsInPerCellLeft with 0s for all the cells in the grid
		mNumberPointsInPerCellLeft.assign(totalNumberOfCellsLeft, 0);

		// Fill the mMotionStatistics and mNumberPointsInPerCellLeft vectors
		AssignMatchPairs(GridType);

		// Fill in the mCellPairs vector
		VerifyCellPairs(RotationType);

		// Mark inliers
		for (size_t i = 0; i < mNumberMatches; i++)
		{
			// If there was a match between the cells
			if (mvMatchPairs[i].first >= 0) {

				// There should be an equal number of matches per cell pair (if the cells match)
				if (mCellPairs[mvMatchPairs[i].first] == mvMatchPairs[i].second)
				{
					// By setting the inlier mask to false initially,
					// only true matches will be found.
					// Fill mvbInlierMask with true if the match was true
					mvbInlierMask[i] = true; 
				}
			}
		}
	}

	// Return the total number of inliers found
	int num_inlier = sum(mvbInlierMask)[0];
	return num_inlier;
}