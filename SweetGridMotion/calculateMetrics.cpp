//Precision = T P / (T P + F P)
//recall = T P / (T P + F N)

#include <opencv2/core.hpp>             // namespace cv
#include <opencv2/core/persistence.hpp> // https://docs.opencv.org/3.4/da/d56/classcv_1_1FileStorage.html
#include <opencv2/features2d.hpp>       // Brute force matcher
#include <iostream>

// Modified code from 
// https://towardsdatascience.com/improving-your-image-matching-results-by-14-with-one-line-of-code-b72ae9ca2b73

using namespace cv;
using namespace std;

// Use homography
void useHomography(const vector<KeyPoint>& vkp1, const vector<KeyPoint>& vkp2) {

    // Load homography file
    FileStorage file = FileStorage("homography file.xml", 0);
    Mat homography = file.getFirstTopLevelNode().mat();
    cout << "Homography from img1 to img2" << homography << endl;

    Mat descriptors1;
    Mat descriptors2;

    vector<KeyPoint>matched1;
    vector<KeyPoint>matched2;

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE_HAMMING);
    vector<vector<DMatch>> matches;
    matcher->knnMatch(descriptors1, descriptors2, matches, 2);
    vector<KeyPoint> matched1, matched2;
    double nearestNeighborMatchingRatio = 0.8;

    for (size_t i = 0; i < matches.size(); i++) {
        DMatch first = matches[i][0];
        float dist1 = matches[i][0].distance;
        float dist2 = matches[i][1].distance;
        if (dist1 < nearestNeighborMatchingRatio * dist2) {
            matched1.push_back(vkp1[first.queryIdx]);
            matched2.push_back(vkp2[first.trainIdx]);
        }
    }

    vector<KeyPoint>inliers1;
    vector<KeyPoint>inliers2;

}

/*

We will consider a mach as valid if its point in image 2 and its point 
from image 1 projected to image 2 are less than 2.5 pixels away.

good_matches = []
inlier_threshold = 2.5  # Distance threshold to identify inliers with homography check
for i, m in enumerate(matched1):
    # Create the homogeneous point
    col = np.ones((3, 1), dtype=np.float64)
    col[0:2, 0] = m.pt
    # Project from image 1 to image 2
    col = np.dot(homography, col)
    col /= col[2, 0]
    # Calculate euclidean distance
    dist = sqrt(pow(col[0, 0] - matched2[i].pt[0], 2) + pow(col[1, 0] - matched2[i].pt[1], 2))
    if dist < inlier_threshold:
        good_matches.append(cv.DMatch(len(inliers1), len(inliers2), 0))
        inliers1.append(matched1[i])
        inliers2.append(matched2[i])

Now that we have the correct matches inside inliers1 and inliers2 variables, we can evaluate the results qualitative using cv.drawMatches. Each of the corresponding points can help us in higher level tasks such as homography estimation, Perspective-n-Point, plane tracking, real-time pose estimation or images stitching.

res = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)
cv.drawMatches(img1, inliers1, img2, inliers2, good_matches, res)
plt.figure(figsize=(15, 5))
plt.imshow(res)

Since it is hard to compare qualitative this kind of results, lets plot some quantitative evaluation metrics. The metric that best reflects how reliable is our descriptor is the percentage of inliers:


Matching Results (BEBLID)
*******************************
# Keypoints 1:                          9105
# Keypoints 2:                          9927
# Matches:                              660
# Inliers:                              512
# Percentage of Inliers:                77.57%
Using the BEBLID descriptor obtains a 77.57% of inliers. If we comment BEBLID and uncomment ORB descriptor in the description cell, the results drop to 63.20%:


Matching Results (ORB)
*******************************
# Keypoints 1:                          9105
# Keypoints 2:                          9927
# Matches:                              780
# Inliers:                              493
# Percentage of Inliers:                63.20%

# Comment or uncomment to use ORB or BEBLID
# descriptor = cv.xfeatures2d.BEBLID_create(0.75)
descriptor = cv.ORB_create()
kpts1, desc1 = descriptor.compute(img1, kpts1)
kpts2, desc2 = descriptor.compute(img2, kpts2)


In conclusion, changing only one line of code to replace ORB descriptor with BEBLID we can improve the matching result of these two images a 14%. This can have a big impact in higher level tasks that need local feature matching to work so do not hesitate, give BEBLID a try!

*/