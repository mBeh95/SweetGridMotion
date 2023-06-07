// 
// Use the homography between two images to calculate how well GMS performed.

#pragma once
#include <opencv2/core.hpp>             // namespace cv
#include <opencv2/core/persistence.hpp> // https://docs.opencv.org/3.4/da/d56/classcv_1_1FileStorage.html
#include <opencv2/features2d.hpp>       // Brute force matcher
#include <opencv2/highgui.hpp>          // imshow
#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>

#include <iostream>

// Modified code from 
// https://towardsdatascience.com/improving-your-image-matching-results-by-14-with-one-line-of-code-b72ae9ca2b73
// https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html#tutorial_homography_Demo3

using namespace cv;
using namespace std;

/** Use homography to calculate the metrics
* @pre       A homography file exists.
*            A validation set of points exists.
* @post      Open the homography file.
*            Calculate the true inliers using ground truth homography
*            established by the makers of the dataset. 
*            Print the precision.
* @param	 GMSkptsLeft is the keypoints GMS located from the first image (left image).
* @param     GMSkptsRight is the keypoints GMS located from the second image (right image).
* @param	 img1 is the first image (left image).
* @param     img2 is the second image (right image).
*/
void useHomography(const vector<KeyPoint>& GMSkptsLeft, const vector<KeyPoint>& GMSkptsRight, 
    vector<DMatch> matchesFoundByGMS, Mat& img1, Mat& img2) {

    // Load homography file - not working
    //FileStorage file = FileStorage("Eiffel_vpts.mat", 0);
    //Mat homography = file.getFirstTopLevelNode().mat();

    /*double homography[3][3] = 
      { {1.32601002878971, -0.0583865106212613, -934.618313266433},
        {0.293840970834713,	1.08257312730755, 484.497536919587},
        {0.000336792169880890, -0.000200624987739184, 1} };*/

    double homography[3][3] = { 
        {0.929315345255432, -0.121244736991080, 50.7244433172911},
        {-0.0275532282755572, 0.808809574501722, 832.934983027886},
        {-0.000014365821012690, -0.000142061737799082, 1} };

    Mat homographyMat(3, 3, CV_64F, homography);

    std::cout << "Homography from img1 to img2" << homographyMat << endl;

    // Hold the keypoint coordinates of only the matches from GMS
    vector<KeyPoint> leftMatchesFromGMS, rightMatchesFromGMS;
    
    // Hold any matches that were correct (true positives) that GMS found 
    vector<DMatch> inlier_matches;

    // Fill the matches vectors with the keypoints that were given to GMS from ORB
    // These are matches that GMS found to be good matches.
    for (size_t i = 0; i < matchesFoundByGMS.size(); i++) {
        leftMatchesFromGMS.push_back(GMSkptsLeft[matchesFoundByGMS[i].queryIdx]);
        rightMatchesFromGMS.push_back(GMSkptsRight[matchesFoundByGMS[i].trainIdx]);

        cout << "matchesFoundByGMS at queryIdx: " << matchesFoundByGMS[i].queryIdx << endl;
        cout << "matchesFoundByGMS at trainIdx: " << matchesFoundByGMS[i].trainIdx << endl << endl;

        cout << "GMSkptsLeft[matchesFoundByGMS[i].queryIdx]: " << GMSkptsLeft[matchesFoundByGMS[i].queryIdx].pt << endl;
        cout << "GMSkptsRight[matchesFoundByGMS[i].trainIdx]: " << GMSkptsRight[matchesFoundByGMS[i].trainIdx].pt << endl<< endl;

    }

   


    // Hold only the inliers (true positives) that GMS found.
    vector<KeyPoint>inliersLeftImage, inliersRightImage;

    // Distance threshold:
    // A good match is fewer than 2.5 pixels
    // from where the homography says it should be
    double inlier_threshold = 10;

    // How many inliers were true positives?
    int count = 0;

    // For all the matches, check to see if they
    // fall within the area that the homography says they will
    for (int i = 0; i < leftMatchesFromGMS.size(); i++) {

        // Create a mat where the calculation can happen
        // This will be one column with 3 rows of doubles
        // [ ]
        // [ ]
        // [ ]
        //Mat col = Mat::ones(3, 1, CV_64F);
        Mat col = (Mat_<double>(3, 1) << leftMatchesFromGMS[i].pt.x, leftMatchesFromGMS[i].pt.y, 1);

        // Fill the mat with the x, y coordinates of image1
        // [x]
        // [y]
        // [1]
        //col.at<double>(0, 0) = leftMatchesFromGMS[i].pt.x;
        //col.at<double>(1, 0) = leftMatchesFromGMS[i].pt.y;
        //col.at<double>(2, 0) = 1;

        cout << "THIS IS COL AT X Y Z before mulitiplication" << endl 
            << col.at<double>(0, 0) << endl 
            << col.at<double>(1, 0) << endl 
            << col.at<double>(2, 0) << endl;

        // Project the point from image1 to image2
        // [x] * homographyMat
        // [y]
        // [1]
        col = homographyMat * col;

        cout << "THIS IS COL AT X Y Z after multiplying by the homography" << endl
            << col.at<double>(0, 0) << endl
            << col.at<double>(1, 0) << endl
            << col.at<double>(2, 0) << endl;

        // Project the point from image1 to image2
        // [x2 * scaling factor]
        // [y2 * scaling factor]
        // [scaling factor]
        col /= col.at<double>(2, 0);

        cout << "THESE ARE THE PROJECTED POINTS ON IMAGE 2:" << endl
            << col.at<double>(0, 0) << endl
            << col.at<double>(1, 0) << endl;

        // Find the euclidean distance between the projected point
        // on image2 and the match that was found in image2.

        cout << rightMatchesFromGMS[i].pt.x << endl;
        cout << rightMatchesFromGMS[i].pt.y << endl;

        double dist = 

            // square root of    (x2 - GMSx)^2 + (y2 - GMSy)^2
            sqrt(pow(col.at<double>(0, 0) - rightMatchesFromGMS[i].pt.x, 2)
               + pow(col.at<double>(1, 0) - rightMatchesFromGMS[i].pt.y, 2));

        cout << "The euclidean distance is " << dist << endl;
        cout << "If the distance was within" << inlier_threshold << "pixels count it as an inlier" << endl;

        // If the distance was within 2.5 pixels count it as an inlier
        if (dist < inlier_threshold) {
            count++;
        }
    }

    // Print the results
    cout << "T P = true positives  = good matches that were found by GMS" << endl;
    cout << "F P = false positives = bad  matches that were found and mistaken for good matches by GMS" << endl;
    cout << "F N = false negatives = good matches that were not found by GMS" << endl;

    cout << "Total keypoints found in image 1: " << GMSkptsLeft.size() << endl;
    cout << "Total keypoints found in image 2: " << GMSkptsRight.size() << endl;
    cout << "Total matches found (T P + F P):  " << matchesFoundByGMS.size() << endl;
    cout << "Total inliers found (T P):        " << count << endl;
    cout << "Precision = T P / (T P + F P):    " << 100 * count / matchesFoundByGMS.size() << "%" << endl;

}
