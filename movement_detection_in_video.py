import cv2
import numpy as np
cap=cv2.VideoCapture('vtest.avi')
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output.avi',fourcc,20.0,(1280,720))
_,frame1=cap.read()

_,frame2=cap.read()
while cap.isOpened():
    diff=cv2.absdiff(frame1,frame2)
    diff_gray=cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(diff_gray,(5,5),0)
    _,thresh=cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated=cv2.dilate(thresh,None,iterations=3)
    countours,_=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame1,countours,-1,(0,255,0),2,cv2.LINE_AA)
    for contour in countours:
        (x,y,w,h)=cv2.boundingRect(contour)
        if cv2.contourArea(contour)<700 :
            continue
        else:
            cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame1,"Status : {}".format("movement"),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
    image=cv2.resize(frame1,(1280,720))
    out.write(image)
    cv2.imshow('Detection',frame1)
    frame1=frame2
    _,frame2=cap.read()


    k=cv2.waitKey(10)
    if k==27:
        break
cv2.destroyAllWindows()
cap.release()
out.release()