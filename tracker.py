# Import the required modules
import argparse as ap
import get_points
from data_processing import *
import numpy as np
import os.path


# need to do perspective correction before selecting origin
# need to figure out what to use as reference to calculate perspective transform
# possible candidates: white background board
# can't use primary arm as any reference is already distorted and we need to calculate actual
# arm size. That is, we will scale incorrectly. What about using spacing between screws on white
# backboard? This could work, but a measurement will need to be made in person.
# use width of bottom of primary arm to set length/pixel scale in image.
# it passes relatively close to the center of the image so shouldn't be distorted too much there.
def run(source=0, dispLoc=False):
    vertical_size = 720
    frame_rate = 240
    try:
        import cv2
    except:
        print "openCV not installed"
        exit()
    # Create the VideoCapture object
    cam = cv2.VideoCapture(source)
    # If Camera Device is not opened, exit the program
    if not cam.isOpened():
        print "Video device or file couldn't be opened"
        exit()
    
    print "Press key `p` to pause the video to start tracking"
    while True:
        # Retrieve an image and Display it.
        retval, img = cam.read()
        vertical_size = len(img)
        if not retval:
            print "Cannot capture frame device"
            exit()
        if(cv2.waitKey(10) % 256 == ord('p')):
            break
        cv2.namedWindow("Video Preview", cv2.WINDOW_NORMAL)
        cv2.imshow("Video Preview", img)
    cv2.destroyWindow("Video Preview")

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
        if not retval:
            cam.release()
            break
        # Update the tracker  
        for i in xrange(len(tracker)):
            ok, bbox = tracker[i].update(img)
            curr_bound_box.append(bbox)
            # Get the bounding box of ith object, draw a 
            # box around it and display it.
            pt1 = (int(bbox[0]), int(bbox[1]))
            pt2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(img, pt1, pt2, (255, 255, 255), 2)
            if dispLoc:
                loc = (int(rect.left()), int(rect.top()-20))
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
        primary[i][0] = i / float(frame_rate)
        secondary[i][0] = i / float(frame_rate)
        primary[i][1] = rect[0][0] + rect[0][2] / 2 - origin[0]
        primary[i][2] = vertical_size - rect[0][1] + rect[0][3] / 2 - (vertical_size - origin[1])
        secondary[i][1] = rect[1][0] + rect[1][2] / 2 - origin[0]
        secondary[i][2] = vertical_size - rect[1][1] + rect[1][3] / 2 - (vertical_size - origin[1])
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