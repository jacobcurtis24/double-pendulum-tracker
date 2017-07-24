# Import the required modules
import cv2
import argparse as ap
import get_points
from data_processing import *
import numpy as np
import os.path

def run(source=0, dispLoc=False):
    # Create the VideoCapture object
    cam = cv2.VideoCapture(source)
    lower = np.array([5, 2, 116])
    upper = np.array([65, 110, 236])
    # If Camera Device is not opened, exit the program
    if not cam.isOpened():
        print "Video device or file couldn't be opened"
        exit()
    
    print "Press key `p` to pause the video to start tracking"
    while True:
        # Retrieve an image and Display it.
        retval, img = cam.read()
        #mask = cv2.inRange(img, lower, upper)
        #img = cv2.bitwise_and(img, img, mask=mask)
        if not retval:
            print "Cannot capture frame device"
            exit()
        if(cv2.waitKey(10) % 256 == ord('p')):
            break
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
    cv2.destroyWindow("Image")

    # Co-ordinates of objects to be tracked 
    # will be stored in a list named `points`
    points = get_points.run(img, multi=True) 
    # add a way to select origin
    if len(points) != 3:
        print "The first box must be the origin. Two points on the double pendulum must be selected"
        exit()

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", img)

    # Initial co-ordinates of the object to be tracked 
    # Create the tracker object
    origin = points[0]
    origin = (origin[0] + origin[2] / 2, origin[1] + origin[3] / 2)
    points2 = []
    for i in range(1, 3):
        points2.append(points[i])

    points = points2
    tracker = []
    for i in range(2):
        tracker.append(cv2.Tracker_create("MIL")) #use BOOSTING, but change samplerSearchFactor. seems to work well at 2.5f
    # Provide the tracker the initial position of the object
    [tracker[i].init(img, rect) for i, rect in enumerate(points)]

    bounding_boxes = []

    while True:
        curr_bound_box = []
        # Read frame from device or file
        retval, img = cam.read()
        #mask = cv2.inRange(img, lower, upper)
        #img = cv2.bitwise_and(img, img, mask=mask)
        if not retval:
            cam.release()
            break
        # Update the tracker  
        for i in xrange(len(tracker)):
            ok, bbox = tracker[i].update(img)
            curr_bound_box.append(bbox)
            # Get the bounding box of th object, draw a 
            # box around it and display it.
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(img, pt1, pt2, (255, 255, 255), 2)
            #print "Object {} tracked at [{}, {}] \r".format(i, pt1, pt2),
            if dispLoc:
                loc = (int(rect.left()), int(rect.top()-20))
	        #txt = "Object tracked at [{}, {}]".format(pt1, pt2)
	        #cv2.putText(img, txt, loc , cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.imshow("Image", img)
        # Continue until the user presses ESC key
        if cv2.waitKey(1) % 256 == 27:
            break
        bounding_boxes.append(curr_bound_box)
    
    # need to add time. check the analyze function for correct data form
    primary = np.zeros((len(bounding_boxes), 3), dtype=float)
    secondary = np.zeros((len(bounding_boxes), 3), dtype=float)
    for i, rect in enumerate(bounding_boxes):
        primary[i][0] = i / float(240)
        secondary[i][0] = i / float(240)
        primary[i][1] = rect[0][0] + rect[0][2] / 2 - origin[0]
        primary[i][2] = 720 - rect[0][1] + rect[0][3] / 2 - (720 - origin[1])
        secondary[i][1] = rect[1][0] + rect[1][2] / 2 - origin[0]
        secondary[i][2] = 720 - rect[1][1] + rect[1][3] / 2 - (720 - origin[1])
    return primary, secondary
    

if __name__ == "__main__":
    # Parse command line arguments
    parser = ap.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', "--deviceID", help="Device ID")
    group.add_argument('-v', "--videoFile", help="Path to Video File")
    parser.add_argument('-l', "--dispLoc", dest="dispLoc", action="store_true")
    args = vars(parser.parse_args())

    # Get the source of video
    if args["videoFile"]:
        source = args["videoFile"]
    else:
        source = int(args["deviceID"])


    primary_name = "_primary_pos.csv"
    secondary_name = "_secondary_pos.csv"
    
    # check if position CSV files have already been created
    video_name = args["videoFile"]
    video_name = (video_name.split('.', 1))[0]
    primary_name = video_name + primary_name
    secondary_name = video_name + secondary_name
    
    if os.path.isfile(primary_name) and os.path.isfile(secondary_name):
        primary = get_csv(primary_name)
        secondary = get_csv(secondary_name)
    else:
        primary, secondary = run(source, args["dispLoc"])
        cv2.destroyWindow("Image")
        make_csv(primary_name, primary)
        make_csv(secondary_name, secondary)

    analyze(primary, secondary, video_name)