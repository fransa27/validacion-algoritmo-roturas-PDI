#include<iostream>
#include<map>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/core/core.hpp>

#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;
cv::Mat kme(cv::Mat img);

int main(int argc, char *argv[]) {
  //Load the video. 
  cv::VideoCapture vc("D:/U/PDI/videos/video1.mp4");  
  
  //If the video isn't open, terminate program.  
  if (!vc.isOpened()) {  
    std::cerr << "[-] Error: Unable to load video.\n" << std::endl;
    std::cerr << "Press ENTER to exit..." << std::endl;
    std::cin.ignore();  
  
    return EXIT_FAILURE; 
  }  
  //Create new window. 
  cv::namedWindow("Output", CV_WINDOW_AUTOSIZE);  
  //Get frames per seconds. 
  double fps = vc.get(CAP_PROP_FPS);  
  //Calculate the time between each frame to display.
  int delay = 1000 / (int)fps;  
  
  std::cout << "Press ESC to exit..." << std::endl;  
  for (;;) {  
    cv::Mat frame;   
    vc >> frame;   
    if (frame.empty())   
      break; 
    cv::imshow("Output", kme(frame));   
    if (cv::waitKey(delay) == 27)   
      break; 
    }
    vc.release(); 
    cv::destroyWindow("Output");  
    return EXIT_SUCCESS;
}



cv::Mat kme(cv::Mat img) {
    Mat samples(img.cols*img.rows, 1, CV_32FC3); 
    // Matriz de marca, conformación de 32 bits 
    Mat labels(img.cols*img.rows, 1, CV_32SC1);
    uchar* p; 
    int i, j, k=0; 
    for(i=0; i < img.rows; i++) 
    { 
    p = img.ptr<uchar>(i); 
    for(j=0; j< img.cols; j++) 
    { 
        samples.at<Vec3f>(k,0)[0] = float(p[j*3]); 
        samples.at<Vec3f>(k,0)[1] = float(p[j*3+1]); 
        samples.at<Vec3f>(k,0)[2] = float(p[j*3+2]); 
        k++; 
    } 
    }

    int clusterCount = 10; //Cantida de clusters
    Mat centers(clusterCount, 1, samples.type()); 
    kmeans(samples, clusterCount, labels, 
    TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);
    // Sabemos que hay 3 grupos, que están representados por diferentes niveles de gris. 
    Mat img1(img.rows, img.cols, CV_8UC1); 
    float step=255/(clusterCount - 1); 
    k=0; 
    for(i=0; i < img1.rows; i++) 
    { 
        p = img1.ptr<uchar>(i); 
        for(j=0; j< img1.cols; j++) 
        { 
            int tt = labels.at<int>(k, 0); 
            k++; 
            p[j] = 255 - tt*step; 
        } 
    }

    return img1;
}