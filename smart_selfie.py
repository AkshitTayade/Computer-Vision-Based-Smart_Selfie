import cv2 
import numpy as np 
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/akshit/Downloads/Skyfi Labs Computer Vision/shape_predictor.dat') 

(mstart,mend) = (48,67)
smile_cont = 5
counter = 0 
selfie_no = 0

def rect_to_bb(rect) :
    x = rect.left()
    y = rect.top()
    w = rect.right()-x
    h = rect.bottom()-y
    return (x,y,w,h)

def shape_to_np(shape,dtype='int') :
    coords = np.zeros((68,2),dtype=dtype)

    for i in range(0,68) :
        coords[i] = (shape.part(i).x , shape.part(i).y)
    return coords

def smile(shape) :
    left = shape[48]
    right = shape[54]

    mid = (shape[51]+shape[62]+shape[66]+shape[57])/4

    dist = np.abs(np.cross(right-left,left-mid)) / np.linalg.norm(right-left)
    return dist

cam = cv2.VideoCapture(0)

while(cam.isOpened()) :
    ret,frame = cam.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    rects = detector(gray,1)

    for i in range(0,len(rects)) :
        (x,y,w,h) = rect_to_bb(rects[i])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
        shape = predictor(gray,rects[i])

        shape = shape_to_np(shape)

        mouth = shape[mstart:]
       

        for (x,y) in mouth:
            cv2.circle(frame , (x,y), 1 , (255,0,0),-1)
        
        smile_param = smile(shape)
        cv2.putText(frame,"SP: {:.2f}".format(smile_param),(300,30),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)

        if smile_param > smile_cont :
            cv2.putText(frame,"Smile Detected",(300,60),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),2)
            counter = counter +1

            if counter > 15 :
                selfie_no = selfie_no+1
                ret,frame1 = cam.read()
                img_name = "/Users/akshit/Downloads/Skyfi Labs Computer Vision/Extensions/smart_selfie_{}.png".format(selfie_no)
                cv2.imwrite(img_name,frame1)
                print('taken !')
                counter = 0

        else :
            counter = 0

    cv2.imshow('frames',frame)
    key = cv2.waitKey(10)

    if key == 27 :
        break 


cam.release()
cv2.destroyAllWindows()