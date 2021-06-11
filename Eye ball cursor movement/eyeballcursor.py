# USAGE
# python eyeballcursor.py --shape-predictor shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
##import playsound
import argparse
import imutils
import time
import dlib
import cv2
import math
import pyautogui
def sound_alarm(path):
        # play an alarm sound
        playsound.playsound(path)

def eye_aspect_ratio(eye):
        # compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        kk1=eye[1];
        kk2=eye[5];
        x1=kk1[1];
        y1=kk2[0];
        h1=kk1[1]-kk2[1];
        w1=kk2[0]-kk1[0];
        

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
##      print eye[0],eye[3]
        vert1=eye[0];
        vert2=eye[3];
        x=vert1[1];
        y=vert1[0];
        h=vert1[1]-vert2[1];
        w=vert2[0]-vert1[0];
        if w==0:
                w=-1;
        if h==0:
                h=1;
##        print w

        # compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
##      print eye[0],eye[3]

        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # return the eye aspect ratio
        cnter=(eye[0]+eye[3])/2;
        return ear,h,w,x,y,cnter
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
##ap.add_argument("-a", "--alarm", type=str, default="",
##      help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
        help="index of webcam on system")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold for to set off the
# alarm
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 1

# initialize the frame counter as well as a boolean used to
# indicate if the alarm is going off
COUNTER = 0
ALARM_ON = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)
data=['q','w','e','r','t','y','u','i','o','p'];
# loop over frames from the video stream
cursdata=[[1,192],[193,384],[385,576]];
muldiffcentX=0;
muldiffcentY=0;
CentInitX=1920/2;
CentInitY=1080/2;
FrameCentX=960/2;
FrameCentY=540/2;
lp=0;
while True:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        
        frame = vs.read()
        frame = cv2.resize(frame,(960, 540), interpolation = cv2.INTER_CUBIC)
##      if lp==0:
##                dat=input('initialize the keyboard');
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # extract the left and right eye coordinates, then use the
                # coordinates to compute the eye aspect ratio for both eyes
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                ear,h,w,x,y,cnter = eye_aspect_ratio(leftEye)
                if lp>0:
                        c11=x;
                        c22=y;
                x=int(math.ceil(cnter[0]));
                y=int(math.ceil(cnter[1]));
                cv2.circle(frame,(x,y),5,255,-1);
                if lp>0:
                        diffcentX=FrameCentX-x;
                        diffcentY=FrameCentY-y;
                        muldiffcentX=diffcentX*6;
                        muldiffcentY=diffcentY*6;
                pyautogui.moveTo(CentInitX+muldiffcentX,CentInitY-muldiffcentY);
                lp=1;
                rightEAR,hrr,wr,xr,yr,cnterr = eye_aspect_ratio(rightEye)
##              rightEAR = eye_aspect_ratio(rightEye)

                # average the eye aspect ratio together for both eyes
##              ear = (leftEAR + rightEAR) / 2.0

                # compute the convex hull for the left and right eye, then
                # visualize each of the eyes
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                # check to see if the eye aspect ratio is below the blink
                # threshold, and if so, increment the blink frame counter
                if ear < EYE_AR_THRESH:
                        pyautogui.doubleClick() 
##                        pyautogui.click(CentInitX+muldiffcentX,CentInitY-muldiffcentY, button='left')
                        COUNTER += 1

                        # if the eyes were closed for a sufficient number of
                        # then sound the alarm
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                # if the alarm is not on, turn it on
                                if not ALARM_ON:
                                        ALARM_ON = True

                                        # check to see if an alarm file was supplied,
                                        # and if so, start a thread to have the alarm
                                        # sound played in the background
##                                      if args["alarm"] != "":
##                                              t = Thread(target=sound_alarm,
##                                                      args=(args["alarm"],))
##                                              t.deamon = True
##                                              t.start()

                                # draw an alarm on the frame
                                cv2.putText(frame, "BUTTON PRESSED!", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # otherwise, the eye aspect ratio is not below the blink
                # threshold, so reset the counter and alarm
                else:
                        COUNTER = 0
                        ALARM_ON = False
                if rightEAR < EYE_AR_THRESH:
##                        pyautogui.click(CentInitX+muldiffcentX,CentInitY-muldiffcentY, button='left')
                        COUNTER += 1
                        pyautogui.click(button='right')
                        # if the eyes were closed for a sufficient number of
                        # then sound the alarm
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                                # if the alarm is not on, turn it on
                                if not ALARM_ON:
                                        ALARM_ON = True

                                        # check to see if an alarm file was supplied,
                                        # and if so, start a thread to have the alarm
                                        # sound played in the background
##                                      if args["alarm"] != "":
##                                              t = Thread(target=sound_alarm,
##                                                      args=(args["alarm"],))
##                                              t.deamon = True
##                                              t.start()

                                # draw an alarm on the frame
                                cv2.putText(frame, "RIGHT CLICK PRESSED!", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # otherwise, the eye aspect ratio is not below the blink
                # threshold, so reset the counter and alarm
                else:
                        COUNTER = 0
                        ALARM_ON = False
                # draw the computed eye aspect ratio on the frame to help
                # with debugging and setting the correct eye aspect ratio
                # thresholds and frame counters
                cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
        # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
 
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
                break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
