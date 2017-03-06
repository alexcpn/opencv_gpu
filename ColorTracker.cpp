#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
    //VideoCapture cap("/home/alex/Videos/test_thermal.flv"); //capture the video from web cam
    VideoCapture cap(1); //capture the video from web cam
    if (!cap.isOpened())  // if not success, exit program
    {
        cout << "Cannot open the web cam" << endl;
        return -1;
    }

    namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

    int iLowH = 0;
    int iHighH = 179;

    int iLowS = 0;
    int iHighS = 255;

    int iLowV = 0;
    int iHighV = 255;


    int threshold_value = 0;
    int threshold_type = 3;;
    int const max_value = 255;
    int const max_type = 4;
    int const max_BINARY_value = 255;

    char *trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
    char *trackbar_value = "Value";

    //Create trackbars in "Control" window
    cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
    cvCreateTrackbar("HighH", "Control", &iHighH, 179);

    cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
    cvCreateTrackbar("HighS", "Control", &iHighS, 255);

    cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
    cvCreateTrackbar("HighV", "Control", &iHighV, 255);

    /// Create Trackbar to choose type of Threshold
    createTrackbar(trackbar_type,
                   "Control", &threshold_type,
                   max_type);

    createTrackbar(trackbar_value,
                   "Control", &threshold_value,
                   max_value);

    double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
    double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
    // double totalfps = cap.get(CV_CAP_PROP_FRAME_COUNT);
    Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
    VideoWriter oVideoWriter;
    oVideoWriter.open("/tmp/threshold_out.avi", CV_FOURCC('D', 'I', 'V', 'X'), 3, frameSize, true);
    if (!oVideoWriter.isOpened()) //if not initialize the VideoWriter successfully, exit the program
    {
        cout << "ERROR: Failed to write the video" << endl;
        return -1;
    }
    while (true) {
        Mat imgOriginal;

        bool bSuccess = cap.read(imgOriginal); // read a new frame from video

        if (!bSuccess) //if not success, break loop
        {
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }

        Mat imgHSV;

        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

        Mat imgThresholded;

        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV),
                imgThresholded); //Threshold the image

        //morphological opening (remove small objects from the foreground)
        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
        dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

        //morphological closing (fill small holes in the foreground)
        dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

        RNG rng(12345);
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        Mat canny_output, dst;
        threshold(imgThresholded, dst, threshold_value, max_BINARY_value, threshold_type);
        /// Detect edges using canny
        Canny(dst, canny_output, threshold_value, threshold_value * 2, 3);
        findContours(dst, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        vector<Rect> boundRect(contours.size());
        /// Draw contours
        Mat drawing = Mat::zeros(dst.size(), CV_8UC3);
        for (int i = 0; i < contours.size(); i++) {
            Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            drawContours(imgOriginal, contours, i, color, 2, 8, hierarchy, 0, Point());
            boundRect[i] = boundingRect(Mat(contours[i]));
            rectangle(imgOriginal, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
        }
        //imshow( "Bounding", drawing );
        imshow("Thresholded Image", imgThresholded); //show the thresholded image
        imshow("Original", imgOriginal); //show the original image
        oVideoWriter.write(imgOriginal);
        if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
        {
            cout << "esc key is pressed by user" << endl;
            break;
        }
    }
    oVideoWriter.release();

    return 0;
}