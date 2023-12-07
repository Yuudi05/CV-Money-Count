
import cv2 as cv
import cvzone as cz
import numpy as np
cap = cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
Total = 0

def empty(a):
    pass

cv.namedWindow("Settings")
cv.resizeWindow("Settings",640,240)
cv.createTrackbar("Tresh1","Settings",50,255,empty)
cv.createTrackbar("Tresh2","Settings",100,255,empty)

def PreProcessing(img):

    imgPre = cv.GaussianBlur(img,(5,5),3)
    th1 = cv.getTrackbarPos("Tresh1","Settings")
    th2 = cv.getTrackbarPos("Tresh2","Settings")
    imgPre = cv.Canny(imgPre,123,255)
    kernel = np.ones((3,3),np.uint8)
    imgPre = cv.dilate(imgPre,kernel,iterations=1)
    imgPre = cv.morphologyEx(imgPre,cv.MORPH_CLOSE,kernel)


    return imgPre

while True:
    success, img = cap.read()
    imgPre = PreProcessing(img)
    imgCount,counF = cz.findContours(img,imgPre,minArea=15)
    Total=0
    if counF:
        for contour in counF:
            peri = cv.arcLength(contour['cnt'], True)
            approx = cv.approxPolyDP(contour['cnt'], 0.02 * peri, True)

            if len(approx) > 5:
                area = contour["area"]
                print(area)
                if 3700>area>3500:
                    Total += 0.5
                elif 12000>area>12900:
                    Total += 1
                elif 5300>area>5000:
                    Total+=5
                elif 5800>area>5600:
                    Total+=2
                else:
                    Total=+10


    image = cz.stackImages([img,imgPre,imgCount],2,1)
    cz.putTextRect(image,f'DH{Total}',(50,50))
    cv.imshow("Image",image)
    cv.waitKey(1)
