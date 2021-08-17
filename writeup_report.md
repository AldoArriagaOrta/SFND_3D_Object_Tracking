#  Track an Object in 3D Space

---


## [Rubric](https://review.udacity.com/#!/rubrics/2550/view) Points


[//]: # (Image References)

[image1]: ./writeup_images/TTC_lidar.png "TTC_lidar"
[image2]: ./writeup_images/TTC_camera.png "TTC_camera"
[image3]: ./writeup_images/TTC_camera.png "TTC_camera"
[image11]: ./writeup_images/11.png "frame11"
[image12]: ./writeup_images/12.png "frame12"
[image13]: ./writeup_images/13.png "frame13"
[image16]: ./writeup_images/16.png "frame16"
[image17]: ./writeup_images/17.png "frame17"



---

### FP.1 Match 3D Objects

#### Criteria

Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences. 

#### Meets Specifications

Code is functional and returns the specified output, where each bounding box is assigned the match candidate with the highest number of occurrences. 

#### Solution

The solution uses of a multimap to store the matched pairs counts.

```c++
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
}
```

An alternative approach was investigated without using multimap. 

```c++
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // Alternative approach without using multimap
    for(auto& prevBB : prevFrame.boundingBoxes) // loop over all the bounding boxes in the current frame
    {
        int keyptsMax = 0;
        int currBoxID = -1;
        for(auto& currBB : currFrame.boundingBoxes) // loop over all the bounding boxes in the previous frame 
        {   
            int  keyptsCount = 0;            
            for (auto& match : matches) // loop over all the matched keypoint pairs betweein the current and previous frame
                if (currBB.roi.contains(currFrame.keypoints[match.trainIdx].pt) && prevBB.roi.contains(prevFrame.keypoints[match.queryIdx].pt) )
                    keyptsCount++;

            if (keyptsCount > keyptsMax)
            {
                keyptsMax = keyptsCount;
                currBoxID = currBB.boxID;
            }
        }
        
        if (keyptsMax > 0 && currBoxID!= -1) // check if there is a match for the current bounding box
        {
            bbBestMatches.insert(std::pair<int,int>(prevBB.boxID,currBoxID));
            cout  << "prevBoxID:" << prevBB.boxID << " matched with currBoxID: " << currBoxID << endl;  
        }             
    }  
}
```
The results are similar, with the disadvantage that the actual count for each possible matching pair is not stored. With the previous approach we could try to make a one-to-one mapping instead of the current one-to-many mapping. It would depend on the final application. 


### FP.2 Compute Lidar-based TTC

#### Criteria

Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.

#### Meets Specifications

Code is functional and returns the specified output. Also, the code is able to deal with outlier Lidar points in a statistically robust way to avoid severe estimation errors. 

#### Solution

The solution was implemented in the computeTTCLidar function of camFusion_Student.cpp

```c++
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

```
The custom lambda expression for sorting takes into account the longitudinal distance, but also the lane width like the example from the lesson.
This is based on the assumption that obstacles would have to enter the ego lane to become a potential driving hazard. This logic considers that the ego vehicle drives in a straight line and it would have to be adjusted in order to take into account the curvature radius in case that the ego vehicle is turning. 

### FP.3 Associate Keypoint Correspondences with Bounding Boxes

#### Criteria

Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.

#### Meets Specifications

Code performs as described and adds the keypoint correspondences to the "kptMatches" property of the respective bounding boxes. Also, outlier matches have been removed based on the euclidean distance between them in relation to all the matches in the bounding box. 

#### Solution

The solution was implemented in the clusterKptMatchesWithROI function of camFusion_Student.cpp

A median filter was used to enhance robustness.

```c++
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

```

### FP.4 Compute Camera-based TTC

#### Criteria

Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

#### Meets Specifications

Code is functional and returns the specified output. Also, the code is able to deal with outlier correspondences in a statistically robust way to avoid severe estimation errors. 

#### Solution

The implementation is based directly on the code studied in the lesson. It uses a median filter to reject outliers.

```c++
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
  	std::sort(distRatios.begin(), distRatios.end());
    double medIndex = floor(distRatios.size()/2.0);
  	double medianDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : 
                                                           distRatios[medIndex];                        
    TTC = -dT / (1 - medianDistRatio);
}

```

### FP.5 Performance Evaluation 1

#### Criteria

Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.

#### Meets Specifications

Several examples (2-3) have been identified and described in detail. The assertion that the TTC is off has been based on manually estimating the distance to the rear of the preceding vehicle from a top view perspective of the Lidar points.

#### Solution

Between frames 11 and 12 there is a noisy lidar point that introduces an important error in the TTC computation.

![alt text][image11] ![alt text][image12] ![alt text][image13]

This point effectively makes it look like the preceding vehicle changed direction between frames, giving a negative TTC = -10.38571429 s

A second example of this effect is shown in the last two frames

![alt text][image16] ![alt text][image17]

Giving also a negative TTC = -9.857142857 s

Thanks to the median filter in the implementation, the TTC is much more robust to this effect:

![alt text][image1]

There seems to be also an issue in frame 4. The distance between frames seems to be rather small and might indicate a latency issue. Is the time synchronization guaranteed? Is dt truly consistent?


### FP.6 Performance Evaluation 2

#### Criteria

Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

#### Meets Specifications

All detector / descriptor combinations implemented in previous chapters have been compared with regard to the TTC estimate on a frame-by-frame basis. To facilitate comparison, a spreadsheet and graph should be used to represent the different TTCs. 

#### Solution

The data for this analysis is available in this [spreadsheet](https://docs.google.com/spreadsheets/d/1VBbLL6KB2y7tE6Bmjy8iFbTGy8eLFMegTuukmtArr5c/edit#gid=1837838976)

The most stable detector-descriptor pairs are summarized in the next plot:

![alt text][image2]

It is clear that most detector-descriptor pairs struggle with frame 4. Further investigation is needed to determine the cause of the spike in the TTC values. The fact that it is the same frame as with the lidar detection might give some support to my hypothesis about the inconsistent delta time.

The execution times for these pairs was also analyzed and summarized in the following plot:

![alt text][image3]

All the pairs here presented using a FAST detector show stable performance with great execution time. Surprisingly SHITOMASI-SIFT and SHITOMASI-ORB give very reasonable results as well. All these would be suitable options for further development.