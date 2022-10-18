#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat src_gray;
Mat dst, detected_edges;
int lowThreshold = 15; //You can change it
const int max_lowThreshold = 100;
const int radio = 3;
const int kernel_size = 3;
const char* window_name = "Edge Map";
int main( int argc, char** argv )
{
    VideoCapture cap("C:/Users/lesly.fuentes/Desktop/U/PDI/data_damage-detection-20221016T221004Z-001/data_damage-detection/videos/video1.mp4");
    if ( !cap.isOpened() )  // isOpened() returns true if capturing has been initialized.
    {
        cout << "Cannot open the video file. \n";
        return -1;
    }

    double fps = cap.get(CAP_PROP_FPS); //get the frames per seconds of the video
    namedWindow("Canny",WINDOW_AUTOSIZE); //create a window called "MyVideo"

     while(1)
    {
        Mat src;
       
        // Mat object is a basic image container. frame is an object of Mat.

        if (!cap.read(src)) // if not success, break loop
        // read() decodes and captures the next frame.
        {
            cout<<"\n Cannot read the video file. \n";
            break;
        }
        int cols = src.cols;
        int rows = src.rows;
        resize(src,src,Size(cols/4,rows/4));

        dst.create( src.size(), src.type() );
        cvtColor( src, src_gray, COLOR_BGR2GRAY );
        blur( src_gray, detected_edges, Size(3,3) ); 
        Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*radio, kernel_size );
        dst = Scalar::all(0);
        src.copyTo( dst, detected_edges);
        imshow("Canny", dst);
        // first argument: name of the window.
        // second argument: image to be shown(Mat object).

        if(waitKey(30) == 27) // Wait for 'esc' key press to exit
        { 
            break; 
        }
    }
    return 0;
}