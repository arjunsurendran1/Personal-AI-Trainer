#import libraries
import cv2
import time
import numpy as np
import poseModule as pm #module that created before for find positions 
import math


#input video
cap = cv2.VideoCapture('projects/personal AI trainer/1.mp4')

#use this for take video from webcam
#cap = cv2.VideoCapture('0')

#create object 
detector = pm.poseDetector()


#function for find angle between 3 points
def findAngle(img,p1,p2,p3,draw = True):
    #get the landmarks
    x1,y1 = lmList[p1][1:]
    x2,y2 = lmList[p2][1:]
    x3,y3 = lmList[p3][1:]
    
    #claculate angle
    angle = math.degrees(math.atan2(y3-y2,x3-x2)-math.atan2(y1-y2,x1-x2))
    #print(angle)
    if angle < 0 :
        angle+=360


    #draw
    if draw:
        
        cv2.circle(img,(x2,y2),3,(0,0,255),2)
        cv2.line(img,(x1,y1),(x2,y2),(255,255,255),3)
        cv2.line(img,(x3,y3),(x2,y2),(255,255,255),3)

        #10,15
        cv2.circle(img,(x1,y1),10,(0,0,255),cv2.FILLED)
        cv2.circle(img,(x1,y1),15,(0,0,255),3)

        cv2.circle(img,(x2,y2),10,(0,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),15,(0,0,255),3)

        cv2.circle(img,(x3,y3),10,(0,0,255),cv2.FILLED)
        cv2.circle(img,(x3,y3),15,(0,0,255),3)
        #cv2.putText(img,str(int(angle)),(x2 - 50,y2 + 50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),4)

    return angle


count = 0
dir = 0
pTime = 0

while True:
    succes , img = cap.read()
    img = cv2.resize(img,(1280,720))
    #img = cv2.imread('/home/arjun/Documents/opencv/Advanced cv/projects/personal AI trainer/1.jpg')

    img=detector.findPose(img,draw=False)
    lmList = detector.findPosition(img,draw=False)
    #print(lmList)
    if len(lmList) != 0:
        #right arm
        #findAngle(img,12,14,16)
        
        #left arm
        angle = findAngle(img,11,13,15)
        per = np.interp(angle,(210,310),(0,100))
        bar = np.interp(angle,(220,310),(650,100))
        #print(angle, per)

        #check for the dumbbell curls
        color=(255,0,255)
        if per == 100:
            color=(0,255,0)
            if dir == 0:
                count+=0.5
                dir = 1
        if per == 0:
            color=(0,255,0)
            if dir == 1:
                count+=0.5
                dir = 0

        #print(count)
        #draw bar

        cv2.rectangle(img,(1100,100),(1175,650),color,3)
        cv2.rectangle(img,(1100,int(bar)),(1175,650),color,cv2.FILLED)
        cv2.putText(img,f'{int(per)}%',(1100,75),cv2.FONT_HERSHEY_PLAIN,4,color,4)

        #draw curl count
        cv2.rectangle(img,(0,450),(250,720),(0,255,0),cv2.FILLED)
        cv2.putText(img,str(int(count)),(45,670),cv2.FONT_HERSHEY_PLAIN,15,(255,0,0),25)

        #show fps
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(50,100),cv2.FONT_HERSHEY_PLAIN,10,(255,0,0),5)


    #resize image
    #img = cv2.resize(img,(1280,720))

    cv2.imshow('image',img)
    cv2.waitKey(1)
