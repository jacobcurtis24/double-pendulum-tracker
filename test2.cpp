// Standard include files
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
 
using namespace cv;
 
int main(int argc, char **argv)
{
    // Set up tracker. 
    // Instead of MIL, you can also use 
    // BOOSTING, KCF, TLD, MEDIANFLOW or GOTURN  
    Ptr<Tracker> tracker = Tracker::create( "GOTURN" );
 
    // Read video
    VideoCapture video("45deg_short.mov");
     
    // Check video is open
    if(!video.isOpened())
    {
        std::cout << "Could not read video file" << std::endl;
        return 1;
    }
 
    // Read first frame. 
    Mat frame;
    video.read(frame);
 
    // Define an initial bounding box
    Rect2d bbox(287, 23, 86, 320);
 
    // Uncomment the line below if you 
    // want to choose the bounding box
    // bbox = selectROI(frame, false);
     
    // Initialize tracker with first frame and bounding box
    tracker->init(frame, bbox);
 
    while(video.read(frame))
    {
        // Update tracking results
        tracker->update(frame, bbox);
 
        // Draw bounding box
        rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
 
        // Display result
        imshow("Tracking", frame);
        int k = waitKey(1);
        if(k == 27) break;
 
    }
 
    return 0; 
     
}