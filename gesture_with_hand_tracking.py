# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:27:05 2020

@author: anupa
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 12:38:28 2019

@author: Anupam Anand
"""

import cv2
from tensorflow.keras.models import load_model, Sequential, Model
import numpy as np
import threading
import pyttsx3
import sqlite3
from collections import deque
from imutils.video import WebcamVideoStream
#import pyttsx3

engine = pyttsx3.init()
conn =sqlite3.connect('gesturedb')
c = conn.cursor()
row = c.execute('Select * from ges_dy')
    
frame_count=0
direction_list=''
pts = deque(maxlen=15)
(dX,dY) = (0,0)
direction = ""
word_dict ={}

for i in row:
    
    print(i)
    if(i[0] not in word_dict.keys()):
        word_dict[i[0]]= {}
    
    word_dict[i[0]][i[2]] = i[1]
        
print(word_dict)
    
def letter(t):
    
    return(chr(65+t))
    
def speak(word):
    
    global speak
    engine.say(word)
    engine.runAndWait()
      
    
print("Use default webcam? \n y/n : ")
cam= 0

if(input()=='n'):
    
    print("Enter video stream output link: ")
    cam = input()

video = WebcamVideoStream(cam).start()

print("Video Capture Started")
voice = "Video Capture Started"
th = threading.Thread(target = speak, args=(voice,))
th.start()

print("Loading Brain")
model = load_model('best_cust_model.h5')
print("Loaded")

tracker = cv2.TrackerCSRT_create()


#model = pickle.load(open('svchand.sav', 'rb'))
hand = False
initBB = None  #initial Bounding Box

while(True):
    
    frame = video.read()
    frame = cv2.flip(frame,flipCode=1)
    word=''
    key = cv2.waitKey(20)
    
    ret= True
    if (ret==True):
        
        if not hand:
            cv2.rectangle(frame, (412,278),(608,82),(0,100,0),2 )
        
        if (key == ord("s") or key==ord("S")):
                
            initBB = (412,82,196,196)
            #initBB = cv2.selectROI("Frame", frame, fromCenter=False)
            tracker.init(frame, initBB)
            hand = True
            
        elif(key==ord('r') or key==ord("R")):
            
            initBB= None
            hand = False
            tracker.clear()

            
        
        if initBB is not None:
            
            success, box = tracker.update(frame)
            
            if success:
                 
                (x,y,w,h) = [int(v) for v in box]
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                
                
                try:
                    image = frame[y:y+h, x:x+w]
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image = cv2.resize(image, dsize=(28,28))
                    
                except:
                    print("You are out of bounds")
                    cv2.putText(frame,"You are out of bounds",org= (210,320),fontFace= cv2.FONT_HERSHEY_PLAIN,fontScale=1, color=(0,0,255),thickness=1)
                    continue
                
                
                if(key == ord('c')):
            
                    cv2.imwrite('image.jpg', image)
                
                image = cv2.flip(image,flipCode=1)
                image = image.reshape(1,28,28,1)
                
                if(frame_count==0):
                    text = model.predict(image.astype(float))
                
                
                if(max(text[0])>=0.8):
                    
                    m = np.argmax(text[0])
            
                    if(frame_count<30):
                        
                        frame_count+=1
                        centroid = (x+0.5*w, y+0.5*h)
                        pts.appendleft(centroid)
                        cv2.putText(frame,str(m)+"--"+str(frame_count),org= (10,10),fontFace= cv2.FONT_HERSHEY_PLAIN,fontScale=1, color=(0,0,255),thickness=1)
                        
                        for i in range(1,len(pts)):
                        
                            if(pts[i-1] is None or pts[i] is None):
                                continue
                            
                            if frame_count >= 15 and i==1 and pts[-10] is not None:
                                
                                dX = pts[-10][0] - pts[i][0]
                                
                                dY = pts[-10][1] - pts[i][1]
                                direction = ""
                                if(np.abs(dX)>20):
                                    direction = "left" if np.sign(dX)==1 else "right"
                                    
                                    if(len(direction_list)==0):
                                        direction_list += direction[0]
                                    elif(direction_list[len(direction_list)-1]!=direction[0]):
                                        direction_list += direction[0] 
                                    print(direction)
                                    
                                    
                                elif(np.abs(dY)>20):
                                    direction = "up" if np.sign(dY)==1 else "down"
                                    
                                    if(len(direction_list)==0):
                                        direction_list += direction[0]
                                    elif(direction_list[len(direction_list)-1]!=direction[0]):
                                        direction_list += direction[0] 
                                    print(direction)
                                    
                    elif(frame_count==30):
                        
                        print(m,direction_list)
                        
                        if(direction_list in word_dict[m].keys()):
                            word = word_dict[m][direction_list]
                            
                        frame_count=0
                        direction_list= ''
                        
                        
                    if(word!=''):
                        cv2.putText(frame, word , (300,340), cv2.FONT_HERSHEY_COMPLEX, 1, (0,100,0),3)
                        t1 = threading.Thread(target = speak, args=(word,))
                        t1.start()
                
            else:
                frame_count=0
                    
        if (key == ord('q') or key==ord("Q")):
            
            break
            
    else:
        break
        
    cv2.imshow('Video Feed', frame);
    
video.stop()
cv2.destroyAllWindows()
conn.close()