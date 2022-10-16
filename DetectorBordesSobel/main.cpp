#include<iostream>
#include<map>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



using namespace std;
using namespace cv;
cv::Mat sobelGradient(cv::Mat src);

int main(int argc, char *argv[]) {

     /* if(argc < 4) {
        cerr << "Usage: ./OPENCV_FILTERS image smooth_size sigma" << endl;
        return 1;
    } */

    /*  int size = atoi(argv[2]);
    int sigma = atof(argv[3]); */
    cv::VideoCapture vid("C:/Users/pancholopez/Documents/1USM/2022-2/PDI-Real-ELO329/OPENCV_FILTERS_mio/SAMPLES/video1.mp4");

    //cv::VideoCapture vid;
    //vid.open(0);
    vid.set(cv::CAP_PROP_AUTO_EXPOSURE, 0);
    vid.set(cv::CAP_PROP_EXPOSURE, -7.0);
    if(!vid.isOpened()) {
        cerr << "Error opening video." << endl;
        return 1;
    } 

    int size= 3;
    int sigma=40;

    if(size % 2 == 0) {
        cerr << "Usage: ./OPENCV_FILTERS image smooth_size sigma." << endl;
        cerr << "Error: smooth_size must be an odd number." << endl;
        return 1;
    }

    if(sigma <= 0) {
        cerr << "Usage: ./OPENCV_FILTERS image smooth_size sigma." << endl;
        cerr << "Error: sigma must be positive." << endl;
        return 1;
    }

    //  cv::Mat M = cv::imread( "C:/Users/pancholopez/Documents/1USM/2022-2/PDI-Real-ELO329/OPENCV_BASICO_CUDA/SAMPLES/contrast2.jpg");
    //cv::Mat img = cv::imread( argv[1], 1 ), gblur;
        //cv::Mat img = cv::imread("C:/Users/pancholopez/Documents/1USM/2022-2/PDI-Real-ELO329/OPENCV_FILTERS_mio/SAMPLES/video1.mp4"), gblur;
    cv::Mat img , gblur,bg;
    
    bool first = true;
    int i=0,c;
    int esize = 25;
    vid >> img;
   

    while(1) {
        vid >> img;
        
        i++;
        if(i>20) {
            if(first) {
                first = false;
                img.copyTo(bg);
                
            } else {

                if(size > 0)
                    cv::GaussianBlur(img, gblur, cv::Size(size,size), sigma, sigma);
                else
                    gblur = img;
                cv::Mat final = sobelGradient(gblur);
                              
                //vid.read(final);


               
                int cols = img.cols, rows = img.rows; //obtener las filas y columnas
                resize(img,img,Size(cols/8,rows/8)); //le reasigno el tamaño

                int cols2 = final.cols, rows2 = final.rows; //obtener las filas y columnas
                resize(final,final,Size(cols2/4,rows2/4)); //le reasigno el tamaño

                imshow("Video Normal", img);
                imshow("Video Sobel", final);
                


            }
        }
        if( (c=cv::waitKey(10)) != -1)
            break;
    }

   

    vid.release();
    

    //cv::waitKey(0);
    //vid.release();
    return 0;
}



cv::Mat sobelGradient(cv::Mat src) {
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    /// Generate grad_x and grad_y
    cv::Mat grad, grad_x, grad_y, abs_grad_x, abs_grad_y;

    /// Gradiente X
    cv::Sobel(src, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_x, abs_grad_x);

    /// Gradiente Y
    cv::Sobel(src, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);

    /// Total Gradient (approximate)
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    std::vector<cv::Mat> canales, convertido;
    cv::split(grad, canales);

    double maxVal, factor;
    for(unsigned int i=0; i<canales.size(); i++) {
        cv::minMaxLoc(canales[i], NULL, &maxVal, NULL, NULL);
        factor = 255/maxVal;
        cv::Mat m8u;
        canales[i].convertTo(m8u, CV_8U, factor);
        convertido.push_back(m8u);
    }

    cv::Mat final;
    cv::merge(convertido, final);
    
    return final;
}
