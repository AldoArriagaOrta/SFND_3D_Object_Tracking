#  Track an Object in 3D Space

---


## [Rubric](https://review.udacity.com/#!/rubrics/2550/view) Points


[//]: # (Image References)

[image3]: ./writeup_images/image1.png "image1"


---

### FP.1 Match 3D Objects

## Criteria

Implement the method "matchBoundingBoxes", which takes as input both the previous and the current data frames and provides as output the ids of the matched regions of interest (i.e. the boxID property). Matches must be the ones with the highest number of keypoint correspondences. 

## Meets Specifications

Code is functional and returns the specified output, where each bounding box is assigned the match candidate with the highest number of occurrences. 

## Solution

The initial implementation 


```c++

```

The 

```c++

```


#### FP.2 Compute Lidar-based TTC

## Criteria

Compute the time-to-collision in second for all matched 3D objects using only Lidar measurements from the matched bounding boxes between current and previous frame.

## Meets Specifications

Code is functional and returns the specified output. Also, the code is able to deal with outlier Lidar points in a statistically robust way to avoid severe estimation errors. 

## Solution

The 

```c++

```

#### FP.3 Associate Keypoint Correspondences with Bounding Boxes

## Criteria

Prepare the TTC computation based on camera measurements by associating keypoint correspondences to the bounding boxes which enclose them. All matches which satisfy this condition must be added to a vector in the respective bounding box.

## Meets Specifications

Code performs as described and adds the keypoint correspondences to the "kptMatches" property of the respective bounding boxes. Also, outlier matches have been removed based on the euclidean distance between them in relation to all the matches in the bounding box. 

## Solution

The 

```c++

```

#### FP.4 Compute Camera-based TTC

## Criteria

Compute the time-to-collision in second for all matched 3D objects using only keypoint correspondences from the matched bounding boxes between current and previous frame.

## Meets Specifications

Code is functional and returns the specified output. Also, the code is able to deal with outlier correspondences in a statistically robust way to avoid severe estimation errors. 

## Solution

The 

```c++

```

#### FP.5 Performance Evaluation 1

## Criteria

Find examples where the TTC estimate of the Lidar sensor does not seem plausible. Describe your observations and provide a sound argumentation why you think this happened.

## Meets Specifications

Several examples (2-3) have been identified and described in detail. The assertion that the TTC is off has been based on manually estimating the distance to the rear of the preceding vehicle from a top view perspective of the Lidar points.

## Solution

The 

```c++

```

#### FP.6 Performance Evaluation 2

## Criteria

Run several detector / descriptor combinations and look at the differences in TTC estimation. Find out which methods perform best and also include several examples where camera-based TTC estimation is way off. As with Lidar, describe your observations again and also look into potential reasons.

## Meets Specifications

All detector / descriptor combinations implemented in previous chapters have been compared with regard to the TTC estimate on a frame-by-frame basis. To facilitate comparison, a spreadsheet and graph should be used to represent the different TTCs. 

## Solution

The following code presents the implementation :

```c++


```

