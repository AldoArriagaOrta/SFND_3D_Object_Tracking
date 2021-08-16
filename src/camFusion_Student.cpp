#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 0);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                              std::vector<cv::DMatch> &kptMatches)
{
    // Add all the current matched keypoints and their corresponding match to the bounding box
    std::vector<cv::DMatch> matches;

    for (auto& match : kptMatches) // loop over all the matched keypoint pairs between the current and previous frame 
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt))
            matches.push_back(match);

    //Calculate the Euclidean Distance of each matched pair within the bounding box and store it in a vector
    std::vector<double> euDist;
    for (const auto& match : matches)
        euDist.push_back(cv::norm(kptsCurr[match.queryIdx].pt - kptsPrev[match.trainIdx].pt));

    //find the median distance between the keypoint matches inside the box
    std::sort(euDist.begin(), euDist.end());
    double medIndex = floor(euDist.size()/2.0);
    double medianDistance = euDist.size() % 2 == 0 ? (euDist[medIndex - 1] + euDist[medIndex]) / 2.0 : euDist[medIndex]; 

    // delete outliers, or rather push inliers
    const double distThr = 15; 
 
    for (auto& match : matches)
    {
        if (abs(cv::norm(kptsCurr[match.queryIdx].pt - kptsPrev[match.trainIdx].pt) - medianDistance) > distThr)
        {
            boundingBox.keypoints.push_back(kptsCurr[match.trainIdx]);
            boundingBox.kptMatches.push_back(match);
        }
    }  
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // based on the lesson's code
    double dT = 1 / frameRate;

    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto& it1 : kptMatches)
    { 
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1.trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1.queryIdx);

        for (auto& it2 : kptMatches)
        { 
            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2.trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2.queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }
  
    if (distRatios.size() == 0) // only continue if list of distance ratios is not empty
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();
  	std::sort(distRatios.begin(), distRatios.end());
    double medIndex = floor(distRatios.size()/2.0);
  	double medianDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : 
                                                           distRatios[medIndex];                        
    TTC = -dT / (1 - medianDistRatio);
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dT = 1 / frameRate;
    static double oldTTC = 1e9;             //Time to collision initialization
    const double minDistDetection = 0.001;  //minimum distance between frames of 1 mm
    constexpr const double laneWidth = 4.0; //lane width in m

    // Custom lambda expression for sorting algorithm.    
    auto comparePoints = [](LidarPoint a, LidarPoint b) { return a.x < b.x && a.y < laneWidth/2.0 && a.y > laneWidth/-2.0 && 
                                                                 b.y < laneWidth/2.0 && b.y > laneWidth/-2.0; };

    // Median Filter
    // sort lidar point vectors based on custom condition
    std::sort(lidarPointsPrev.begin(), lidarPointsPrev.end(), comparePoints);
    std::sort(lidarPointsCurr.begin(), lidarPointsCurr.end(), comparePoints);

    const int sampleSize = 300; //Based on the number of lidar points from the preceding vehicle
    double medIndex = floor(sampleSize/2.0);

    double medianDistPrev = sampleSize % 2 == 0 ? (lidarPointsPrev[medIndex - 1].x + lidarPointsPrev[medIndex].x) / 2.0 : lidarPointsPrev[medIndex].x; 
    double medianDistCurr = sampleSize % 2 == 0 ? (lidarPointsCurr[medIndex - 1].x + lidarPointsCurr[medIndex].x) / 2.0 : lidarPointsCurr[medIndex].x;

    // Minimum distance detection and division by zero protection
    double distDiff = abs(medianDistPrev - medianDistCurr);
    if (distDiff > std::numeric_limits<double>::epsilon() &&  distDiff > minDistDetection)
    {
        TTC = medianDistCurr * dT / (medianDistPrev - medianDistCurr);
        oldTTC = TTC;
    }
    else
    {
        TTC = oldTTC;
    }
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::multimap <int, int> keypt2BB;

    for (auto& match : matches) // loop over all the matched keypoint pairs between the current and previous frame   
        for(auto& prevBB : prevFrame.boundingBoxes) // loop over all the bounding boxes in the current frame       
            for(auto& currBB : currFrame.boundingBoxes) // loop over all the bounding boxes in the previous frame             
                if (currBB.roi.contains(currFrame.keypoints[match.trainIdx].pt) && prevBB.roi.contains(prevFrame.keypoints[match.queryIdx].pt))                  
                    keypt2BB.insert(std::pair<int,int>(prevBB.boxID, currBB.boxID));

    // usage of multimap for counting pairs based on: https://www.cplusplus.com/reference/map/multimap/equal_range/
    for(auto& prevBox : prevFrame.boundingBoxes)
    {
        std::pair <std::multimap<int,int>::iterator, std::multimap<int,int>::iterator> ret = keypt2BB.equal_range(prevBox.boxID);       
        vector<int> countMatches(currFrame.boundingBoxes.size(),0);  // vector to store the number of matches for each BB in the current frame

        //increase by 1 the count in the vector element denoted by the second member of the iterator (i.e  countMatches[currBoxID])
        for (std::multimap<int,int>::iterator it = ret.first; it != ret.second; ++it)            
            countMatches[it->second] += 1;         

        //select the current bounding box ID with the highest count
        int currBoxID = std::distance(countMatches.begin(), std::max_element(countMatches.begin(), countMatches.end()));
        bbBestMatches.insert(std::pair<int,int>(prevBox.boxID,currBoxID));
        cout  << "prevBoxID:" << prevBox.boxID << " matched with currBoxID: " << currBoxID << " with " << countMatches[currBoxID] << " matches" << endl;    
    }

    // // Alternative approach without using multimap
    // for(auto& prevBB : prevFrame.boundingBoxes) // loop over all the bounding boxes in the current frame
    // {
    //     int keyptsMax = 0;
    //     int currBoxID = -1;
    //     for(auto& currBB : currFrame.boundingBoxes) // loop over all the bounding boxes in the previous frame 
    //     {   
    //         int  keyptsCount = 0;            
    //         for (auto& match : matches) // loop over all the matched keypoint pairs betweein the current and previous frame
    //             if (currBB.roi.contains(currFrame.keypoints[match.trainIdx].pt) && prevBB.roi.contains(prevFrame.keypoints[match.queryIdx].pt) )
    //                 keyptsCount++;

    //         if (keyptsCount > keyptsMax)
    //         {
    //             keyptsMax = keyptsCount;
    //             currBoxID = currBB.boxID;
    //         }
    //     }
        
    //     if (keyptsMax > 0 && currBoxID!= -1) // check if there is a match for the current bounding box
    //     {
    //         bbBestMatches.insert(std::pair<int,int>(prevBB.boxID,currBoxID));
    //         cout  << "prevBoxID:" << prevBB.boxID << " matched with currBoxID: " << currBoxID << endl;  
    //     }             
    // }  
}
