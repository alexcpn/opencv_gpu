//#include <stdafx.h>
#include <iostream>
#include <time.h> 

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "opencv2/cudaobjdetect.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"
#include "opencv2/core/cuda.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace cv;
using namespace cv::cuda;
using namespace boost;


/** Global variables */
String opencv_path = "~/opencv";
String videoFile = "";
String outFile = "/tmp/out.avi";
String useAlgo = "hog";
bool useGpu = false;
int counter_frames_processed = 0;
int counter_frames_skipped = 0;
int counter_frames_detected = 0;
Size  downFrameSize(640, 480);


/* 
These are the setting for HOG Person detector; There is no one setting that is good for all
Using daimlerpeopledetector ,see where the SVM is set
Default people detector   getDefaultPeopleDetector work only with win_width = 48, with GPU it works with
win_width = 64 as well; but detection rate is very poor
 -->OpenCV Error : Assertion failed(checkDetectorSize()) in cv::HOGDescriptor::setSVMDetector
 */
int win_width = 48; 
//48*96 rectangle is found for HOG 
int cell_width = 8; 

int win_stride_width = 8;
int block_width = win_stride_width*2;

int hogLevels =  HOGDescriptor::DEFAULT_NLEVELS;

/* From above these below are standard setting*/
int g_nbins = 9;
Size g_cell_size(cell_width, cell_width);
Size g_win_stride(win_stride_width, win_stride_width);
Size g_win_size(win_width, win_width * 2);
Size g_block_size(block_width, block_width);
Size g_block_stride(block_width / 2, block_width / 2);


cv::HOGDescriptor g_cpu_hog(g_win_size, g_block_size, g_block_stride, g_cell_size, g_nbins, 1, -1,
	HOGDescriptor::L2Hys, .2, false, hogLevels);
Ptr<cuda::CascadeClassifier> cascade_gpu_upperbody, cascade_gpu_lowerbody, cascade_gpu_fullbody;
Ptr<cv::cuda::HOG> gpu_hog = cv::cuda::HOG::create(g_win_size, g_block_size, g_block_stride, g_cell_size, g_nbins);

cv::CascadeClassifier upperbody_cascade;
cv::CascadeClassifier lowerbody_cascade;
cv::CascadeClassifier fullbody_cascade;

/**
Sclar - BGR value
**/
void drawMarker(Mat img, std::vector<cv::Rect>  found, Scalar sc, int size = 2) {
	
	for (int i = 0; i < (int)found.size(); i++)
	{
		cv::Rect r = found[i]; 
		cv::rectangle(img, r, sc, size);
	}
}

 std::string getPropertyVal(string file_name,string key){
            
    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(file_name, pt);
    std::string  temp = pt.get<std::string>(key);
    return temp;
}

/** @function detectAndDisplay using CPU  */
void detectAndDisplayHOG(Mat img, VideoWriter oVideoWriter, bool useGPU)
{
	Mat frame;
	std::vector<cv::Rect>  found;
	//The GroupThreshold and ScaleFactor are the two important parameters 
	//decrease will get more hits, with more false positives
	int _hitThreshold = 0;// //going mad tuning this for cuda// not to be adjusted
	//decrease will get more hits, with more false positives
	double _scaleFactor = lexical_cast<double>(getPropertyVal("config.ini","HOG.scaleFactor"));//1.05;// 20 sec --> huge impact on performance
        int cell_height =  lexical_cast<int>(getPropertyVal("config.ini","HOG.min_cell_size"));

	if (useGPU) {
		int hogGroupThreshold = lexical_cast<int>(getPropertyVal("config.ini","HOG.hogGroupThreshold"));//
		cv::cvtColor(img, frame, COLOR_BGR2BGRA);// COLOR_BGR2BGRA);
		GpuMat gpuFrame(frame);
		gpu_hog->setScaleFactor(_scaleFactor);
		gpu_hog->setNumLevels(hogLevels); 
		gpu_hog->setWinStride(g_win_stride);
		//gpu_hog->setHitThreshold(0); // play with this at your own risk :)
		gpu_hog->setGroupThreshold(hogGroupThreshold);// setting it to higher will reduce false positives// give all
		gpu_hog->detectMultiScale(gpuFrame, found);
		drawMarker(img, found, Scalar(255, 0, 0), 1);//BGR

	}
	else
	{
		cv::cvtColor(img, frame, COLOR_BGR2GRAY);//(img.type() == CV_8U || img.type() == CV_8UC3)
		g_cpu_hog.detectMultiScale(frame, found, _hitThreshold, g_win_stride, cv::Size(cell_height, cell_height), _scaleFactor);
		drawMarker(img, found, Scalar(255, 0, 0));//BGR
		
	}
	if (found.size() > 1) {
		counter_frames_detected += 1;
	}
	
	oVideoWriter.write(img);
        if(lexical_cast<bool>(getPropertyVal("config.ini","GENERAL.SHOW_DISPLAY"))){
            imshow("opencv", img);
        }
}

/** Helper funcitons**/
void setCudaClassifierProperties(Ptr<cuda::CascadeClassifier> classifier) {

        // The smaller it is the better, though tradeoff is processing (should be >1 )
	classifier->setScaleFactor(lexical_cast<double>(getPropertyVal("config.ini","HAAR.scaleFactor"))); 
        // the larger this is there would be less false positives;
        // However it will also start to miss ;best is 3 to 4, but there are misses wiht this
        classifier->setMinNeighbors(lexical_cast<int>(getPropertyVal("config.ini","HAAR.numberOfNeighbours"))); 

}
/** Helper funcitons**/
void run_classifier_detection(Ptr<cuda::CascadeClassifier> classifier, GpuMat gpuGreyFrame, std::vector<cv::Rect>  *found) {
	GpuMat facesBuf_gpu;
	//Now let the cascaders run
	setCudaClassifierProperties(classifier);
	classifier->detectMultiScale(gpuGreyFrame, facesBuf_gpu);
	classifier->convert(facesBuf_gpu, *found);
}

/** @function detectAndDisplay using CPU  */
void detectAndDisplayHAAR(Mat img, VideoWriter oVideoWriter, bool useGPU)
{

	Mat frame;
	cv::cvtColor(img, frame, COLOR_BGR2GRAY);
	std::vector<cv::Rect>  found;
	//-- Detect Upper body classifier
	// http://fewtutorials.bravesites.com/entries/emgu-cv-c/level-3c---how-to-improve-face-detection

	//Now let the cascaders run, we are running three cascades here
	// Running on GPU for HAAR is much faster than for CPU

	//Now let the cascaders run, we are running three cascades here
	// Running on GPU for HAAR is much faster than for CPU
	if (useGPU) {

		GpuMat gray_gpu(frame);//  gray_gpu, resized_gpu;
                run_classifier_detection(cascade_gpu_upperbody, gray_gpu, &found);
		drawMarker(img, found, Scalar(0, 255, 0));//Green .BGR

		run_classifier_detection(cascade_gpu_fullbody, gray_gpu, &found);
		drawMarker(img, found, Scalar(0, 0, 255));//BGR

		run_classifier_detection(cascade_gpu_lowerbody, gray_gpu, &found);
		drawMarker(img, found, Scalar(255, 0, 0));//BGR
		
	}else {
	

		double scalingFactor = lexical_cast<double>(getPropertyVal("config.ini","HAAR.scaleFactor"));// with 1.001,too much false positive
		int numberOfNeighbours =lexical_cast<int>(getPropertyVal("config.ini","HAAR.numberOfNeighbours"));
                int cell_height =  lexical_cast<int>(getPropertyVal("config.ini","HAAR.min_cell_size"));

		upperbody_cascade.detectMultiScale(frame, found, scalingFactor, numberOfNeighbours, 0, cv::Size(30, 30));
		drawMarker(img, found, Scalar(0, 255, 0));//Green .BGR

		lowerbody_cascade.detectMultiScale(frame, found, scalingFactor, numberOfNeighbours, 0, cv::Size(32, 32));
		drawMarker(img, found, Scalar(0, 0, 255));//BGR

		fullbody_cascade.detectMultiScale(frame, found, scalingFactor, numberOfNeighbours, 0, cv::Size(32, 32));
		drawMarker(img, found, Scalar(255, 0, 0));//BGR
	}

	if (found.size() > 1) {
		counter_frames_detected += 1;
	}
	oVideoWriter.write(img);
        if(lexical_cast<bool>(getPropertyVal("config.ini","GENERAL.SHOW_DISPLAY"))){
            imshow("opencv", img);
        }

}



// To run this you need OpenCV compiled with CUDA support (and a machine with CUDA compliant /NVDIA GPU card
// Based on the sample program from OpenCV - \opencv\samples\gpu\cascadeclassifier.cpp and other samples in net

int main(int argc, char* argv[])
{
	cout << "A Simple Object detection test from Video" <<endl;
	cout << "Set VIDEO_PATH, OPENCV_PATH, USE_GPU=<0/1> USE_ALGO=haar/hog OUT_PATH <output file *avi full path> in Config.ini " << endl;
	///assert((g_win_stride_.width % block_stride_.width == 0 && g_win_stride_.height % block_stride_.height == 0));
	        
        videoFile = getPropertyVal("config.ini","GENERAL.VIDEO_PATH") ;
        opencv_path = getPropertyVal("config.ini","GENERAL.OPENCV_PATH") ;
        outFile = getPropertyVal("config.ini","GENERAL.OUT_PATH") ;
        useAlgo = getPropertyVal("config.ini","GENERAL.USE_ALGO") ;
        useGpu = lexical_cast<bool>(getPropertyVal("config.ini","GENERAL.USE_GPU")) ;
        
	cout << "videoFile = " << videoFile << endl;
	cout << "opencvpath = " << opencv_path << endl;
	cout << "Algorithm Used = " << useAlgo << endl;
	cout << "run_on_gpu = " << useGpu << endl;
	cout << "outFile = " << outFile << endl;
        
    	/**
	Intialize the Algorithm Settings; The speed as well as false positives depended on these
	Unfortunately there is no one setting that is good for all
	**/
        string device_id =getPropertyVal("config.ini","VIDEO.VIDEO_DEVICE_ID");
	VideoCapture cap(lexical_cast<int>(device_id)); // open the video file for reading
        
        string frame_width =getPropertyVal("config.ini","VIDEO.CV_CAP_PROP_FRAME_WIDTH");
        string frame_height =getPropertyVal("config.ini","VIDEO.CV_CAP_PROP_FRAME_HEIGHT");
        
	cap.set(CV_CAP_PROP_FRAME_WIDTH,lexical_cast<int>(frame_width));//640
	cap.set(CV_CAP_PROP_FRAME_HEIGHT,lexical_cast<int>(frame_height));//480 for normal cams
	if (!cap.isOpened())  // if not success, exit program
	{
		cout << " Cannot open the video file" << videoFile << endl;
		return -1;
	}
	cout << " Opened the video file" << videoFile << endl;

	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video
	double totalfps = cap.get(CV_CAP_PROP_FRAME_COUNT);
	Size frameSize(static_cast<int>(dWidth), static_cast<int>(dHeight));
        if(! lexical_cast<bool>(getPropertyVal("config.ini","GENERAL.REDUCE_FRAME")) ){
            downFrameSize = frameSize; // If you dont want to re-size the frame you could uncomment this , it will take more CPU/GPU
        }
	cout << " Orginal Frame Size = " << dWidth << "x" << dHeight << endl;
	cout << " Reduced Frame Size = " << downFrameSize << endl;

	double fps = cap.get(CV_CAP_PROP_FPS); //get the frames per seconds of the video
	cout << "Frame per seconds : " << fps << endl;

	VideoWriter oVideoWriter(outFile, CV_FOURCC('D', 'I', 'V', 'X'), 3, downFrameSize, true);
	if (!oVideoWriter.isOpened()) //if not initialize the VideoWriter successfully, exit the program
	{
		cout << "ERROR: Failed to write the video" << endl;
		return -1;
	}

	if (useGpu) {
		
		if (cv::cuda::getCudaEnabledDeviceCount() == 0) {

			cout << "No GPU found or the library is compiled without CUDA support" << endl;
			return -1;
		}
		cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());

		if (useAlgo == "hog") {
			// If you need to detect other objects you need to train it
			// https://github.com/DaHoC/trainHOG
                        Mat detector = gpu_hog->getDefaultPeopleDetector(); //this will select 48*96 or 64*128 based on window size
			gpu_hog->setSVMDetector(detector);
			cout << "Created the CUDA HOG Classifuer" << endl;
			//cout << gpu_hog->getScaleFactor() << "---" <<  gpu_hog->getGroupThreshold() << endl;
		}
		else //use harr 
		{
			//The below are the path to the HAAR trained casrcades
			//The below taken from http://alereimondo.no-ip.org/OpenCV/34.version?id=60 ; not for commercial use
			String upperbody_cascade_name = opencv_path + "/data/HS22x20/HS.xml"; //head and sholders
			//The below are CUDA Classisfier does not work with older format Cascade xmls; the below are from OpenCV source
			String cuda_lowerbody_cascade_name = opencv_path + "/data/haarcascades_cuda/haarcascade_lowerbody.xml";
			String cuda_fullbody_cascade_name = opencv_path + "/data/haarcascades_cuda/haarcascade_fullbody.xml";

			cout << "head and Shoulder Cascade Name" << upperbody_cascade_name << "Colored GREEN Rectangle" << endl;
			cout << "lowerbody_cascade_name" << cuda_lowerbody_cascade_name << "Colored BLUE Rectangle" << endl;
			cout << "fullbody_cascade_name" << cuda_fullbody_cascade_name << "Colored RED Rectangle" << endl;

			//Load the GPU/CUdA Compliant  video cascaders
			cascade_gpu_upperbody = cuda::CascadeClassifier::create(upperbody_cascade_name);
			cascade_gpu_lowerbody = cuda::CascadeClassifier::create(cuda_lowerbody_cascade_name);
			cascade_gpu_fullbody = cuda::CascadeClassifier::create(cuda_fullbody_cascade_name);
			cout << "Created the CUDA HAAR Classifiers" << endl;
		}

	}
	else //use CPU
	{
		if (useAlgo == "haar") {

			//The below are the path to the HAAR trained casrcades
			//The below taken from http://alereimondo.no-ip.org/OpenCV/34.version?id=60 ; not for commercial use
			String upperbody_cascade_name = opencv_path + "/data/HS22x20/HS.xml"; //head and sholders
			//String lowerbody_cascade_name = opencv_path + "/data/haarcascades/haarcascade_lowerbody.xml";
			String lowerbody_cascade_name = opencv_path + "/data/haarcascades/haarcascade_frontalface_alt_tree.xml";
			String fullbody_cascade_name = opencv_path + "/data/haarcascades/haarcascade_fullbody.xml";

			cout << "head and Shoulder Cascade Name" << upperbody_cascade_name << "Colored GREEN Rectangle" << endl;
			cout << "lowerbody_cascade_name" << lowerbody_cascade_name << "Colored BLUE Rectangle" << endl;
			cout << "fullbody_cascade_name" << fullbody_cascade_name << "Colored RED Rectangle" << endl;

			//-- 1. Load the cascades
			if (!upperbody_cascade.load(upperbody_cascade_name)) {
				printf("--(!)Error loading UpperBody\n");
				return -1;
			};
			if (!lowerbody_cascade.load(lowerbody_cascade_name)) {
				printf("--(!)Error loading lowerbody \n");
				return -1;
			};

			if (!fullbody_cascade.load(fullbody_cascade_name)) {
				printf("--(!)Error loading fullbody\n");
				return -1;
			};

			cout << "Created the HAAR Classifiers" << endl;
		}
		else //use hog
		{
			g_cpu_hog.setSVMDetector(cv::HOGDescriptor::getDaimlerPeopleDetector());
			cout << "Set the HOG Classifiers" << endl;
		}
	}


	double delay = lexical_cast<double>(getPropertyVal("config.ini","GENERAL.DELAY"));
	cout << "Delay is " << delay << endl;
	clock_t startTimeG = clock();
	bool doLoop = true;
	while (doLoop)
	{
		Mat frame, resized;
		
		bool bSuccess = cap.read(frame); // read a new frame from video
		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read the frame from video file" << endl;
			doLoop = false;
			break;
		}
		counter_frames_processed += 1;
	
		cv::resize(frame, resized, downFrameSize);// resize the frame to something smaller- makes computatin faster
		if (useAlgo == "hog") {
			detectAndDisplayHOG(resized, oVideoWriter,useGpu);
		}	
		else //haar
		{
			detectAndDisplayHAAR(resized, oVideoWriter,useGpu);
		}
		
		clock_t endTime = clock() + (delay* CLOCKS_PER_SEC); 
		while (clock()  < endTime) { // This is the best my card supports
			if (cap.read(frame)) { 
				counter_frames_skipped += 1;
			}
			waitKey(1);
		}
		cout << "Frames processed = " << counter_frames_processed << " Frames found = "
		<< counter_frames_detected << " Frames skipped = " << counter_frames_skipped 
		<< " % Time taken =" << (clock() - startTimeG) / CLOCKS_PER_SEC << " seconds"
		<< " \r" ;
	}
	oVideoWriter.release();
	cout << "Total time taken = " << (clock() - startTimeG) / CLOCKS_PER_SEC << " seconds" << endl;
	cout << "counter_frames_processed = " << counter_frames_processed << endl;
	cout << "counter_frames_skipped = " << counter_frames_skipped << endl;
	cout << "counter_frames_detected = " << counter_frames_detected << endl;
	return 0;
}

