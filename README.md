# SweetGridMotion

## Process
We started by commenting on the code to fully understand how GMS works.

1) First, we found the .h version of GMS from Github and added comments.

https://github.com/JiawangBian/GMS-Feature-Matcher/blob/master/include/gms_matcher.h

We saved that as gms_matcher_original_with_comments.h

2) Next, we found the version of GMS that is implemented in OpenCV.

OpenCV did a little refactoring, just cleaning up a few for loops and changing the naming to camel case.

We made a copy of that called gms.cpp

We made another copy of that and called it gms_commented_refactored.cpp

Then we moved the comments from the .h file over to the .cpp file.

There were also a few typos that we caught from the original code, including "initialize" as "initalize" and "Statistics" as "Statisctics".

3) We set up main.cpp and attempted to replicate the results from the 2017 GMS paper.

4) Finally, we made our own implementations.
gms_matcher_mb.h – Melody (Prarin) Behdarvandian
gms_matcher_bp.h – Breanna Powell

## Setup
You will need CMake
You will need OpenCV