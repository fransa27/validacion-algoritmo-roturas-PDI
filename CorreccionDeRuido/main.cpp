// C++ program to demonstrate the
// opening morphological transformation
#include <iostream>
#include <opencv2/core/core.hpp>

// Library include for drawing shapes
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;
using namespace std;

// Function to demonstrate the
// opening morphological operator

int lowThreshold = 20;
const int radio = 5;
const int kernel_size = 3;
const char* window_name = "Opening";
int openingMorphological()
{
    Mat canny,dst,detected_edges;
    // Reading the Image
    Mat image = imread("C:/Users/jorge/Pictures/OpenCV/capture1.jpg", IMREAD_GRAYSCALE); //Cambiar la ruta de la imagen yo usaba una captura de los videos de mallas
    int cols = image.cols, rows = image.rows;
    resize(image, image, Size(cols / 4, rows / 4));
    // Check if the image is
    // created successfully or not
    if (!image.data) {
        cout << "Could not open or"
            << " find the image\n";

        return 0;
    }
    // Create a structuring element
    int morph_size = 2;
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(4, 3)); //Elemento estucturante
    Mat output;


    ///Hasta aqui es lo de apertura lo otro es canny

    // Opening
    morphologyEx(image, output,
        MORPH_OPEN, element,
        Point(-1, -1), 2);
    //Lo que me faltaba hacer es buscar cual elemento estructurante es mejor y entender que es Point(-1, -1), 2
    imshow("Opening", output);

    canny.create(image.size(), image.type());
    blur(image, detected_edges, Size(2, 2)); //output es una imagen en GRAYSCALE
    Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * radio, kernel_size);
    canny = Scalar::all(0);
    image.copyTo(canny, detected_edges);
    imshow("Canny", canny);


    dst.create(output.size(), output.type());
    blur(output, detected_edges, Size(2, 2)); //output es una imagen en GRAYSCALE
    Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * radio, kernel_size);
    dst = Scalar::all(0);
    output.copyTo(dst, detected_edges);
    imshow("Canny + Opening", dst);


    namedWindow(window_name, WINDOW_AUTOSIZE);
    imshow("source", image);
    waitKey();

    return 0;
}

// Driver Code
int main()
{

    openingMorphological();

    return 0;
}
