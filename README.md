# SweetGridMotion

This project builds off of the work published as 
*    GMS: Grid-based Motion Statistics for Fast, Ultra-robust Feature Correspondence.
*    JiaWang Bian, Wen-Yan Lin, Yasuyuki Matsushita, Sai-Kit Yeung, Tan Dat Nguyen, Ming-Ming Cheng
*    IEEE CVPR, 2017
*    ProjectPage: http://jwbian.net/gms

Our goal is to improve the algorithm.

## Process
We started by commenting on the code to fully understand how GMS works.

1) First, we found the .h version of GMS from Github and added comments.

> https://github.com/JiawangBian/GMS-Feature-Matcher/blob/master/include/gms_matcher.h

> We saved that as "gms_matcher_original_with_comments.h"

> We saved the unrefactored .h file as "gms_matcher.h"

2) Next, we found the version of GMS that is implemented in OpenCV.

> https://docs.opencv.org/3.4/db/dd9/group__xfeatures2d__match.html

> OpenCV did a little refactoring, just cleaning up a few for loops and changing the naming to camel case.

> However, for some reason, the OpenCV version does not seem to have the same results as the Github version.

> We made a copy of that and called it gms_commented_refactored.cpp

> Then we moved the comments from the .h file over to the .cpp file and made some small edits to refactor the code.

> There were also a few typos that we caught from the original code, including "initialize" as "initalize" and "Statistics" as "Statisctics".

> Note that we commented out the .cpp file, because the OpenCV code performs poorly compared to the Github version of the original GMS code. 

3) We set up main.cpp and attempted to replicate the results from the 2017 GMS paper.

4) Finally, we made our own implementations.

> gms_matcher_borders.h - for testing our borderCells addition

> gms_matcher_highres.h - for testing a dynamic grid resizing function

> gms_matcher_increased_precision.h - for testing a sub-grid idea

> gms_matcher_rotation_complexity.h - for testing early stopping on rotation idea

## Setup
> You will need Visual Studio Community Edition 2022

> You will need OpenCV

## Presentation Links

> We did an initial design presentation

> https://docs.google.com/presentation/d/e/2PACX-1vRSaX5HrOn5_VuWyQrgkC5JoR9sxFf_q_bZ0U7x1-LWBfZiVuNsLrzPmNeBJ1h20QT54x0Tnn2xMPiB/pub?start=false&loop=false&delayms=3000

> We did a paper presentation

https://docs.google.com/presentation/d/e/2PACX-1vRW-apSE4G-ncZbMjFRHZUvIQV3pD3p6KrNcjKStPsbGqPBYFsa7zXlpeiDlZNIwtm-b8AyzUUPsU_S/pub?start=false&loop=false&delayms=3000

> We did a demo presentation

> https://docs.google.com/presentation/d/e/2PACX-1vTz5A-RDIrVEh2NzzJgNdTV2joOdnR6uMreHi_BtJvpnkbUW_OlGxozH6bG16J-Vu-Ya4aY-cWKXGNq/pub?start=false&loop=false&delayms=3000
