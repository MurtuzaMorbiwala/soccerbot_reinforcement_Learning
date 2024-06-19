
#Import OpenCV
import cv2
#Import Numpy
import numpy as np
import math



import tensorflow as tf
import ScoccerWorld as sc
env = sc.Soccerworld()
observations = env.countstate
actions = 4
e = 0.1



inputs1 = tf.placeholder(shape=[1,observations],dtype=tf.float32)



def soccerpredict(xa,ya,xb,yb):
    s = env.getstate(xa,ya,xb,yb)
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('C:/Users/mmorbiwala/Desktop/HackathonProject/PratikUpdatedFiles/Savemodel/model.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('C:/Users/mmorbiwala/Desktop/HackathonProject/PratikUpdatedFiles/Savemodel/'))
        graph = tf.get_default_graph()
        W = graph.get_tensor_by_name("W:0")
        Qout = tf.matmul(inputs1,W)
        predict = tf.argmax(Qout,1)
        a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(observations)[s:s+1]})
        return a[0]



def BoundBox(frame,Xlower,Xupper):
    foundbox = False
    #Convert the current frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #Define the threshold for finding a blue object with hsv [0,120,120],[20,255,255] for blue
    lower_bound = np.array(Xlower)
    upper_bound = np.array(Xupper)

    #Create a binary image, where anything blue appears white and everything else is black
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    cv2.imshow('mask',mask)
    #Get rid of background noise using erosion and fill in the holes using dilation and erode the final image on last time
    element = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    mask = cv2.erode(mask,element, iterations=2)
    mask = cv2.dilate(mask,element,iterations=2)
    mask = cv2.erode(mask,element)
    
    #Create Contours for all blue objects
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maximumArea = 0
    bestContour = None
    for contour in contours:
        currentArea = cv2.contourArea(contour)
        if currentArea > maximumArea:
            bestContour = contour
            maximumArea = currentArea
     #Create a bounding box around the biggest blue object
    if bestContour is not None:
        x,y,w,h = cv2.boundingRect(bestContour)
        #cv2.rectangle(frame, (x,y),(x+w,y+h), (0,0,255), 3)       
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(frame,str(x) + ',' + str(y),(x,y), font, 1,(255,255,255),2)
        foundbox = True
    elif  bestContour is None:
        x = 1 
        y = 1
        w = 0
        h = 0        
    return [int(x+w/2),int(y+h/2)],foundbox
    






def calibrate(frame,pointxy):
    mindis = 10000
    corner1 = []
    corner2 = []
    center = []
    state = []
    for x in range(1,9):
        for y in range(1,9):        
            dist = math.hypot(pointxy[0] - (x*50+95), pointxy[1] - (y*50+15))
            if dist <= mindis:
                mindis = dist
                corner1 = [x*50+70,y*50-10]
                corner2 = [x*50+120,y*50+40]
                center = [x*50+95,y*50+15]
                state = [x,y]
    cv2.rectangle(frame, (corner1[0],corner1[1]),(corner2[0],corner2[1]), (0,0,255), 1)
    #mindis,corner1,corner2,center,state
    return frame,state


while True:
    
    import urllib.request as urls
    url='http://192.168.200.46:8080/shot.jpg'
    imgResp=urls.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)
    frame = img

    pointxy,greenfoundbox = BoundBox(frame,[40,100,100],[80,255,255])
    if greenfoundbox == True:
        frame,stategreen = calibrate(frame,pointxy)
        print('Green' + str(stategreen))


    pointxy,bluefoundbox = BoundBox(frame,[70,100,100],[110,255,255])
    if bluefoundbox == True:
        frame,stateblue = calibrate(frame,pointxy)
        print('Blue' + str(stateblue))


    if bluefoundbox == True and greenfoundbox == True:
        rew = soccerpredict(stategreen[0],stategreen[1],stateblue[0],stateblue[1])
        if rew == 0 :
            action = 'up'
        if rew == 1 :
            action = 'down'
        if rew == 2 :
            action = 'right'
        if rew == 3 :
            action = 'left'
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,'action is :' + str(action),(320,20), font, 1,(255,255,255),2)
    
    
    cv2.rectangle(frame, (120,40),(520,440), (0,0,255), 1)    
    cv2.imshow('four',frame)
    
    
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    



#Testing


"""
#From Phone

import urllib.request as urls
url='http://192.168.0.12:8080/shot.jpg'
imgResp=urls.urlopen(url)
imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
img=cv2.imdecode(imgNp,-1)
frame = img

frame = BoundBox(frame,[90,100,100],[110,255,255])
frame = BoundBox(frame,[40,100,60],[80,255,255])

cv2.imshow('frame',frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


#LapTopCam

camera_feed = cv2.VideoCapture(0)
_,frame = camera_feed.read()
camera_feed.release()
frame = BoundBox(frame,[90,100,100],[110,255,255])
frame = BoundBox(frame,[40,100,60],[80,255,255])
cv2.imshow('frame',frame)"""

#draw box


