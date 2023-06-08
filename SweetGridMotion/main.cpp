// Breanna Powell and Melody Behdarvandian
// Changes and additions on lines: 42 to 82; 88 to 90; 127 to 143; 150 to 170; 184 to 191;
// Additional commented out code from 245 to 310

#include <iostream>
// #include <opencv2/xfeatures2d.hpp>          // For running the OpenCV version of GMS, which works poorly
// #include <opencv2/xfeatures2d/cuda.hpp>     // For running the OpenCV version of GMS, which works poorly
// #include <opencv2/xfeatures2d/nonfree.hpp>  // For running the OpenCV version of GMS, which works poorly
#include <opencv2/core/hal/interface.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <vector>

#include "calculateMetrics.h"

//#include "gms_matcher.h"
//#include "gms_matcher_highres.h"
//#include "gms_matcher_borders.h"
#include "gms_matcher_rotation_complexity.h" //THIS ONE IS THE BEST FOR TESTING
//#include "gms_matcher_increased_Precision.h"

using namespace cv;
// using namespace xfeatures2d; // For running the OpenCV version of GMS, which works poorly
// using namespace cuda;        // For running the OpenCV version of GMS, which works poorly
using namespace std;
using namespace std::chrono;



void gmsMatch(Mat& img1, Mat& img2);
Mat drawInlier(Mat& src1, Mat& src2, vector<KeyPoint>& kpt1, vector<KeyPoint>& kpt2, vector<DMatch>& inlier, int type);

// siftKeyDes
// Summary: Use SIFT feature detection to get the decriptors and keypoints for the images
// precondition: The images should not be empty, while the decriptors and keypoint parametes will be.
// postconditions: The descriptor and keypoint parameters will now store the values given by SIFT
void siftKeyDes(Mat* descriptOne, Mat* descriptTwo,
	const Mat imgOne, const Mat imgTwo,
	vector<KeyPoint>* keyOne, vector<KeyPoint>* keyTwo, vector<DMatch>* matches, int kpNum)
{
	//Create the SIFT detector
	Ptr<SIFT> detector = SIFT::create(kpNum);

	//Get the keypoints and descriptors from both images
	detector->detectAndCompute(imgOne, Mat(), *keyOne, *descriptOne);
	detector->detectAndCompute(imgTwo, Mat(), *keyTwo, *descriptTwo);

	BFMatcher matcher;
	matcher.match(*descriptOne, *descriptTwo, *matches);
}


// orbKeyDes
// Summary: Use ORB feature detection to get the decriptors and keypoints for the images
// precondition: The images should not be empty, while the decriptors and keypoint parametes will be.
// postconditions: The descriptor and keypoint parameters will now store the values given by ORB
//		

void orbKeyDes(Mat* descriptOne, Mat* descriptTwo,
	const Mat imgOne, const Mat imgTwo,
	vector<KeyPoint>* keyOne, vector<KeyPoint>* keyTwo, vector<DMatch>* matches, int kpNum)
{
	//Create the ORB detector
	Ptr<ORB> detector = ORB::create(kpNum);
	detector->setFastThreshold(0);

	//Get the keypoints and descriptors from both images
	detector->detectAndCompute(imgOne, Mat(), *keyOne, *descriptOne);
	detector->detectAndCompute(imgTwo, Mat(), *keyTwo, *descriptTwo);

	BFMatcher matcher(NORM_HAMMING);
	matcher.match(*descriptOne, *descriptTwo, *matches);
}


/** runImagePair
* @pre       Two valid images exist in the solution
* @post      Reads in the two images         
*/
void runImagePair() {
	//Both images are retrieved.
	//Must be in the same folder as cpp file
	// This string is a collection of images used for testing.
	//                        0         1          2                 3             4                5            6              7                8
	String photoNames[9] = { "01.jpg", "02.jpg", "02_FlipH.jpg", "02_FlipV.jpg", "02_Half.jpg", "02_Half2.jpg", "02_R45.jpg", "02_R90.jpg", "02_Zoomed.jpg" };
	
	//Homography imageset
	Mat img1 = imread("EiffelA.png");
	Mat img2 = imread("EiffelB.png");

	//Run the GMS matching
	gmsMatch(img1, img2);
}


int main()
{
	//Start the program
	runImagePair();

	return 0;
}


///** gmsMatch
//* @pre       Two valid images have been opened, and their keypoints
//*            have been matched. This will output the true matches
//			   between the two images
//* 
//* @post      none.
//*            
//* @param	 img1 is the Mat of image 1 (Left image)
//* @param	 img2 is the Mat of image 2 (Right image)
//*
//*            
//*/

void gmsMatch(Mat& img1, Mat& img2) {
	vector<KeyPoint> kp1, kp2;
	Mat d1, d2, output;
	vector<DMatch> matches_all, matches_gms;

	int kpTotal = 10000;

	//siftKeyDes(&d1, &d2, img1, img2, &kp1, &kp2, &matches_all, kpTotal);

	orbKeyDes(&d1, &d2, img1, img2, &kp1, &kp2, &matches_all, kpTotal);

	drawMatches(img1, kp1, img2, kp2, matches_all, output);

	// Create a window that will show the output result from the feature detector and matching techinique
	namedWindow("output", WINDOW_NORMAL);

	// Scale down the window due to the large image
	resizeWindow("output", output.cols, output.rows);

	// Show window
	imshow("output", output);
	waitKey(0);


	// GMS filter
	//A vector used to store correspondences
	std::vector<bool> vbInliers;

	// Get starting timepoint
	auto start = high_resolution_clock::now();

	//Call the gms function
	gms_matcher gms(kp1, img1.size(), kp2, img2.size(), matches_all);

	//get the number of inliers
	int num_inliers = gms.getInlierMask(vbInliers, false, false);

	// Get ending timepoint
	auto stop = high_resolution_clock::now();

	// Get duration. Substart timepoints to
	// get duration. To cast it to proper unit
	// use duration cast method
	auto duration = duration_cast<microseconds>(stop - start);

	cout << "Time taken by function: "
		<< duration.count() << " microseconds" << endl;

	cout << "Get total " << num_inliers << " matches." << endl;

	// collect matches
	for (size_t i = 0; i < vbInliers.size(); ++i)
	{
		if (vbInliers[i] == true)
		{
			matches_gms.push_back(matches_all[i]);
		}
	}

	// draw matching to a window
	Mat show = drawInlier(img1, img2, kp1, kp2, matches_gms,2);
	// Create a window that will show the output result from the feature detector and matching techinique
	namedWindow("show", WINDOW_NORMAL);

	// Scale down the window due to the large image
	resizeWindow("show", show.cols/2, show.rows/2);
	imshow("show", show);
	//const vector<KeyPoint>& GMSkptsLeft, const vector<KeyPoint>& GMSkptsRight, vector<DMatch> good_matches, Mat& img1, Mat& img2
	useHomography(kp1, kp2, matches_gms, img1, img2);
	waitKey(0);
}

///** drawInlier
//* @pre       Two valid Mat images need to opened along with their keypoints
//*			   A vector containing all true match is also required and the type of the graph
//* 
//* @post      none.
//*            
//* @param	 src1 is the Mat of image 1 (Left image)
//* @param	 src2 is the Mat of image 2 (Right image)
//* @param	 kpt1 is the keypoints from image 1 (Left image).
//* @param	 kpt2 is the keypoints from image 2 (right image).
//* @param	 vector inlier is the true inlier matches between the keypoints,
//* @param   type is the type of graph the user want to view (Only 1 and 2 are availiable)
//*            
//*/
Mat drawInlier(Mat& src1, Mat& src2, vector<KeyPoint>& kpt1, vector<KeyPoint>& kpt2, vector<DMatch>& inlier, int type) {
	const int height = max(src1.rows, src2.rows);
	const int width = src1.cols + src2.cols;
	Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));
	src1.copyTo(output(Rect(0, 0, src1.cols, src1.rows)));
	src2.copyTo(output(Rect(src1.cols, 0, src2.cols, src2.rows)));

	if (type == 1)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			line(output, left, right, Scalar(0, 255, 255));
		}
	}
	else if (type == 2)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			line(output, left, right, Scalar(255, 0, 0));
		}

		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			circle(output, left, 1, Scalar(0, 255, 255), 2);
			circle(output, right, 1, Scalar(0, 255, 0), 2);
		}
	}

	return output;
}

/*************************** NOTE FOR THIS CODE BELOW **************************************
* This section of the code was testing done to run the OpenCV version of GMS
* Because the results of the OpenCV gms were poor, this code was rendered unusable
* We have decided to keep it as a comment to show the work that was done during the
* eperimental phase of our project
*/

/*
int main()
{
    std::cout << "Hello World!\n";

	//Get both the foreground and the background image
	Mat imgOne = imread("01.jpg");
	Mat imgTwo = imread("02.jpg");

	//Keypoint vectors
	vector<KeyPoint> keyOne;
	vector<KeyPoint> keyTwo;

	//Descriptors
	Mat descriptOne;
	Mat descriptTwo;

	Ptr<ORB> detector = ORB::create(10000);

	detector->setFastThreshold(0);

	//Get the keypoints and descriptors from both images
	detector->detectAndCompute(imgOne, Mat(), keyOne, descriptOne);
	detector->detectAndCompute(imgTwo, Mat(), keyTwo, descriptTwo);

	Mat output;
	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING);

	// Create a vector called matches to store all matches
	vector<DMatch> matches;
	vector<DMatch> matchesGMSV;

	// The match function will compare the descriptors and output all resulting matches into the DMatch vector
	matcher->match(descriptOne, descriptTwo, matches);


	matchGMS(imgOne.size(), imgTwo.size(), keyOne, keyTwo, matches, matchesGMSV, true, true, 0.6);


	// Draw the matches between each image based on the keypoints and the matches vector
	// results will be saved in the output Mat variable
	drawMatches(imgOne, keyOne, imgTwo, keyTwo, matchesGMSV, output, Scalar::all(-1), Scalar::all(-1),
		std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);


	Mat show = DrawInlier(imgOne, imgTwo, keyOne, keyTwo, matchesGMSV, 1);
	imshow("show", show);
	waitKey(0);


	/// Create a window that will show the output result from the feature detector and matching techinique
	namedWindow("output", WINDOW_NORMAL);

	// Show window
	imshow("output", output);
	waitKey(0);

}*/
