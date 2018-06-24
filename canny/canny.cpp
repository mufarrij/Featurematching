#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main(){

   //defining variables to store gray images and edge detections

    cv::Mat edges1;
    cv::Mat edges2;
    cv::Mat gray_image1;
    cv::Mat gray_image2;
   
    //setting threshold values

    int edgeThresh = 1;
    int lowThreshold=50;
    int const max_lowThreshold =500 ;
    int ratio = 3;
    int kernel_size = 3;


    
    //loading images 0001.jpg , 0199.jpg into cv::Mat objects image1 ,image2
    
    cv::Mat image1= cv::imread("../data/Fish/img/0001.jpg",CV_LOAD_IMAGE_COLOR);
    cv::Mat image2= cv::imread("../data/Fish/img/0199.jpg",CV_LOAD_IMAGE_COLOR);
    
    
   
   
    
    //Converting image1,image2  from RGB to gray 
    //Reduce noise with a kernel 3x3
    //calling cany detector on gray_image1 ,gray_image2 and saving into edges1 ,edges2
    
    cvtColor( image1, gray_image1, CV_RGB2GRAY);
    
    blur( gray_image1, edges1, cv::Size(3,3) );
    
    cv::Canny(gray_image1,edges1,lowThreshold, lowThreshold*ratio, kernel_size );


    cvtColor( image2, gray_image2, CV_RGB2GRAY );
    
    blur( gray_image2,edges2,cv::Size(3,3));
    
    cv::Canny(gray_image2,edges2,lowThreshold, lowThreshold*ratio, kernel_size );

   
    //displaying the detected edges 

    cv::namedWindow("edges_image1");
    cv::imshow("edge_image1",edges1);
    

    cv::namedWindow("edge_image2");
    cv::imshow("edge_image2",edges2);
   
    //writing detected edges to files output3.jpg , output4.jpg
    imwrite( "output2/output3.jpg", edges1);
    imwrite( "output2/output4.jpg", edges2);
    cv::waitKey(0);

}
