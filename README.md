# opencv_gpu

This demonstrated the use of OpenCV HOG and HAAR algorithms as well as understanding their main parameters for effective Object recongintion.Though this is wrtiten for webcam it is trivial to modify this for any video or image.This also has the OpenCV CUDA related parts for both HOG and HAAR. For tihs you need a NVIDIA GPU and also OpenCV compiled with CUDA.Please see the link http://alexpunnen.blogspot.in/2017/02/compiling-opencv-with-cuda-gpu-using.html
If you need to run it only for CPU you can change the configuraion file appropriately and if you dont want to compile opencv; you can try installing this via sudo apt-get installlibopencv-dev ; I am not sure if that works

The solution is trivial to modify in windows too. Use CMake in Windows to generate a Visual Studio solution

The dependencies for the project are boost libraries and opencv.

sudo apt-get install --no-install-recommends libboost-all-dev
sudo apt-get installlibopencv-dev  (may work else see above on how to compile and use opencv)
