#include <iostream>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char** argv ){

 // Q3) loding the images 0001.jpg , 0199.jpg into cv::Mat objects img1 img2 

 cv::Mat img1 = cv::imread("../data/Fish/img/0001.jpg",CV_LOAD_IMAGE_COLOR);

 cv::Mat img2 = cv::imread("../data/Fish/img/0199.jpg",CV_LOAD_IMAGE_COLOR);

 if (!img1.data || !img2.data) {
		std::cout << "Error occured while reading images" << std::endl;
		return -1;
	}



// Q4) Croping the image1 and displaying it in  a seperate window
  
 cv::Mat image1_crop =img1;
 cv::Rect rect(134,55,60,88);
 image1_crop = image1_crop(rect);

 cv::namedWindow("Image1_cropped",cv::WINDOW_AUTOSIZE);
 cv::imshow("Image1_cropped",image1_crop);
 

// Q5) creating  SurfFeatureDetector of  minHessian=400 && and writting keypoint vectors to csv files
	
 int minHessian = 400;
 Ptr<SURF> detector = SURF::create(minHessian);

 std::vector<KeyPoint> keypoints1, keypoints2;

 detector->detect(img1, keypoints1);
 detector->detect(img2, keypoints2);
 
 cv::Mat keypointsvector_img1 , keypointsvector_img2;
 
 drawKeypoints(img1, keypoints1, keypointsvector_img1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
 drawKeypoints(img2, keypoints2, keypointsvector_img2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
 
 String img1_filename = "features0001.csv", img2_filename = "features0199.csv";
 
 FileStorage fileStorage1(img1_filename, FileStorage::WRITE);
 FileStorage fileStorage2(img2_filename, FileStorage::WRITE);

 cv::write(fileStorage1, "keypoints_1", keypoints1);
 cv::write(fileStorage2, "keypoints_2", keypoints2);
 
 fileStorage1.release();
 fileStorage2.release();
 
 cv::imshow("keypointsvector_img1", keypointsvector_img1);
 cv::imshow("keypointsvector_img2", keypointsvector_img2);
 
 
// Q6) create SurfDescriptorExtractor && create feature discriptor for each keypoint

 Ptr<SURF> extractor = SURF::create();
 Mat feature_discriptor1, feature_discriptor2;
 extractor->compute(img1, keypoints1, feature_discriptor1);
 extractor->compute(img2, keypoints2, feature_discriptor2);

// Q7) 

 //detecting keypoint vectors of image1_crop and creating feature discriptor for image1_crop

 std::vector<KeyPoint> keypoints_image1_crop;
 detector->detect(image1_crop, keypoints_image1_crop);
 cv::Mat keypointsvector_image1_crop;
 drawKeypoints(image1_crop,keypoints_image1_crop, keypointsvector_image1_crop, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
 
 imshow("keypoints_image1_crop", keypointsvector_image1_crop);
 
 //creating feature discriptor for image1_crop
 cv::Mat feature_descriptor_image1_crop;
 extractor->compute(image1_crop,keypoints_image1_crop, feature_descriptor_image1_crop);
 
 //Matching descriptor vectors with a brute force matcher
 
 BFMatcher matcher(NORM_L2);
 std::vector< DMatch > matches;
 matcher.match(feature_descriptor_image1_crop,feature_discriptor1, matches);
 
 


// Q8) Draw matches and writing output to output_matching.jpg file
   
 cv::Mat img_matches;
 cv::drawMatches( image1_crop, keypoints_image1_crop,img1, keypoints1, matches, img_matches);
 
 //displaying matches
 imshow("Matches", img_matches);
 
 //writing output output_matching.jpg file
 imwrite("output/output_matching.jpg", img_matches);
 
   
//Q11) implementation of FLANN matching

  FlannBasedMatcher flannmatcher;
  std::vector< DMatch > flann_matches;
  matcher.match( feature_descriptor_image1_crop,feature_discriptor1, flann_matches );
  
  double max_dist = 0; double min_dist = 100;

  //Quick calculation of max and min distances between keypoints
  for( int i = 0; i < feature_descriptor_image1_crop.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }
  
  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );
  
  //Draw only "good" matches (i.e. whose distance is less than 2*min_dist,

  std::vector< DMatch > good_matches;
  for( int i = 0; i < feature_descriptor_image1_crop.rows; i++ )
  { if( matches[i].distance <= max(2*min_dist, 0.02) )
    { good_matches.push_back( matches[i]); }
  }

  cv::Mat img_matches_flann;
  drawMatches( image1_crop, keypoints_image1_crop,img1, keypoints1,
               good_matches, img_matches_flann, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );


  
  //writing good matches file to output_good_matching file

  imshow("Good Matches", img_matches_flann);
  imwrite("output/output_qood_matching.jpg", img_matches_flann);
  

  cv::waitKey(0);


}
 
 
