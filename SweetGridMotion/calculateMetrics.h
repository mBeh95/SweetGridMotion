// Use the homography between two images to calculate how well GMS performed.

#pragma once
#include <opencv2/core.hpp>             // namespace cv
#include <opencv2/core/persistence.hpp> // https://docs.opencv.org/3.4/da/d56/classcv_1_1FileStorage.html
#include <opencv2/features2d.hpp>       // Brute force matcher
#include <opencv2/highgui.hpp>          // imshow
#include <iostream>

// Modified code from 
// https://towardsdatascience.com/improving-your-image-matching-results-by-14-with-one-line-of-code-b72ae9ca2b73

using namespace cv;
using namespace std;

/** Use homography to calculate the metrics
* @pre       A homography file exists.
*            A validation set of points exists.
* @post      Open the homography file.
*            Do KNN matching to filter out bad matches found by GMS.
*            Pass those matches through Lowe's Ratio Test 
*            to narrow down even more to just the good matches that GMS found.
*            Calculate metrics, using ground truth homography 
*            established by the makers of the dataset. 
*            Print the metrics: Precision and Recall.
* @param	 GMSkptsLeft is the keypoints GMS located from the first image (left image).
* @param     GMSkptsRight is the keypoints GMS located from the second image (right image).
* @param	 img1 is the first image (left image).
* @param     img2 is the second image (right image).
*/
void useHomography(const vector<KeyPoint>& GMSkptsLeft, const vector<KeyPoint>& GMSkptsRight,
    Mat& img1, Mat& img2) {

    // Load homography file
    FileStorage file = FileStorage("Eiffel_vpts.mat", 0);
    Mat homography = file.getFirstTopLevelNode().mat();
    cout << "Homography from img1 to img2" << homography << endl;

    // Hold the descriptors for images 1 and 2.
    Mat descriptors1, descriptors2;

    // Use a brute force hamming distance matcher
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);

    // Collect all the matches between the two images
    vector<vector<DMatch>> matches;

    // Do KNN matching and fill the descriptors1 and descriptors2 vectors.
    matcher->knnMatch(descriptors1, descriptors2, matches, 2);

    // Hold all the keypoints that pass Lowe's Ratio Test
    vector<KeyPoint> matched1, matched2;

    // Lowe's Ratio test threshold
    double nearestNeighborMatchingRatio = 0.8;

    // Use Lowe's Ratio test to ensure the descriptors are similar.
    for (size_t i = 0; i < matches.size(); i++) {
        DMatch first = matches[i][0];
        float dist1 = matches[i][0].distance;
        float dist2 = matches[i][1].distance;
        if (dist1 < nearestNeighborMatchingRatio * dist2) {
            matched1.push_back(GMSkptsLeft[first.queryIdx]);
            matched2.push_back(GMSkptsRight[first.trainIdx]);
        }
    }

    // Hold only the inliers 
    vector<KeyPoint>inliers1, inliers2;

    // Hold only the good matches
    vector<DMatch> good_matches;

    // Distance threshold:
    // A good match is fewer than 2.5 pixels
    // from where the homography says it should be
    double inlier_threshold = 2.5;

    // For all the matches, check to see if they
    // fall within the area that the homography says they will
    for (int i = 0; i < matched1.size(); i++) {

        // Create a mat where the calculation can happen
        Mat col = Mat::ones(3, 1, CV_64F);

        // Fill the mat with the x, y coordinates of image1
        col.at<double>(0, 0) = matched1[i].pt.x;
        col.at<double>(1, 0) = matched1[i].pt.y;

        // Project the point from image1 to image2
        col = homography * col;
        col /= col.at<double>(2, 0);

        // Find the euclidean distance between the projected point
        // on image2 and the match that was found in image2.
        double dist = sqrt(pow(col.at<double>(0, 0) - matched2[i].pt.x, 2)
            + pow(col.at<double>(1, 0) - matched2[i].pt.y, 2));

        // If the distance was within 2.5 pixels add it to the inliers vectors.
        if (dist < inlier_threshold) {
            good_matches.push_back(DMatch(inliers1.size(), inliers2.size(), 0));
            inliers1.push_back(matched1[i]);
            inliers2.push_back(matched2[i]);
        }
    }

    Mat trueInliers = file.operator[]("validation.pts").mat();

    // Print the results
    cout << "T P = true positives  = good matches that were found by GMS" << endl;
    cout << "F P = false positives = bad  matches that were found and mistaken for good matches by GMS" << endl;
    cout << "F N = false negatives = good matches that were not found by GMS" << endl;

    cout << "Total keypoints found in image 1: " << GMSkptsLeft.size() << endl;
    cout << "Total keypoints found in image 2: " << GMSkptsRight.size() << endl;
    cout << "Total matches found (T P + F P):  " << matches.size() << endl;
    cout << "Total inliers found (T P):        " << inliers1.size() << endl;
    cout << "Precision = T P / (T P + F P):    " << 100 * inliers1.size() / matches.size() << "%" << endl;
    cout << "Recall = T P / (T P + F N):       " << 100 * inliers1.size() / trueInliers.size().width << "%" << endl;

}
