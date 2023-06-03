// ConsoleApplication5.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>
#include <iostream>

#include "gms_matcher.h"
//#include "gms_matcher_mb.h"
//#include "gms_matcher_bp.h"
using namespace cv;
using namespace xfeatures2d;
using namespace cuda;
using namespace std;



void gmsMatch(Mat& img1, Mat& img2);
Mat drawInlier(Mat& src1, Mat& src2, vector<KeyPoint>& kpt1, vector<KeyPoint>& kpt2, vector<DMatch>& inlier, int type);

void runImagePair() {
	//Both images are retrieved.
	//Must be in the same folder as cpp file
	Mat img1 = imread("01.jpg");
	Mat img2 = imread("02.jpg");

	//Run the GMS matching
	gmsMatch(img1, img2);
}


int main()
{
	//Start the program
	runImagePair();

	return 0;
}


///** drawInlier
//* @pre       Two valid images have been opened, and their keypoints
//*            have been matched. This will output the true matches
//			   between the two images
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

void gmsMatch(Mat& img1, Mat& img2) {
	vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	vector<DMatch> matches_all, matches_gms;

	Ptr<ORB> orb = ORB::create(10000);
	orb->setFastThreshold(0);

	orb->detectAndCompute(img1, Mat(), kp1, d1);
	orb->detectAndCompute(img2, Mat(), kp2, d2);


	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all);


	// GMS filter
	//A vector used to store correspondences
	std::vector<bool> vbInliers;

	//Call the gms function
	gms_matcher gms(kp1, img1.size(), kp2, img2.size(), matches_all);

	//get the number of inliers
	int num_inliers = gms.GetInlierMask(vbInliers, false, false);
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
	imshow("show", show);
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
