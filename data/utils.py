import cv2
import os
import numpy as np
from datetime import datetime

def anotate_clip(clip,label):
    out_clip = []
    for i,frame in enumerate(clip):
        out_clip.append(write_label(frame,label))
    return  out_clip        

def write_label(frame, prediction):
    
    height,width = frame.shape[0], frame.shape[1]
    midX = width//2
    
    borderColor = (69,209,14)
    
    if prediction['label'].lower() == 'violence':
        borderColor =  (36,28,236)  

    #### Bottom Border  ####
    frame = cv2.copyMakeBorder(frame,
        top=0,
        bottom=10,
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=[*borderColor])
    
    ####   polygon   ####
    vrx = np.array((
                    [(midX-120),height],
                    [(midX-100),(height-20)],
                    [(midX+100),(height-20)],
                    [(midX+120),height]),
                   np.int32)
    vrx = vrx.reshape((-1,1,2))
    frame = cv2.polylines(frame, [vrx], False, (0,0,0),2)
    cv2.fillPoly(frame, pts = [vrx], color =borderColor)
    ###############################

    ####   TEXT    ####
    text_location = (midX-80,height)
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.5
    font_color =(255, 255, 255)
    lineType= 1
    outText = prediction['label']+' '+str(prediction['score'])+'%'
    cv2.putText(frame,
                outText, 
                text_location, 
                font, 
                font_scale,
                font_color,
                lineType,)
    ##############################
    
    return frame


class PreformanceTimer():
    def __init__(self) -> None:
        self.previous = 0
        self.current = 0
        self.lastRecord = 0
        self.records = []
        
    def record(self):
        self.current = cv2.getTickCount()
        if self.previous:  
            self.lastRecord = (self.current - self.previous)/ cv2.getTickFrequency()  
            self.records.append(self.lastRecord) 
        self.previous = self.current
    
    def averageTime(self):
        return sum(self.records)/len(self.records)
    def timePerFrame(self,numOfFrames):
        return ( self.averageTime() / numOfFrames )
    def framerate(self,numOfFrames):
        if not len(self.records):
            return 0
        return ( numOfFrames/self.averageTime() )
    
    def calculateDelay(self,outputSPF ,numOfFrames):
        #average time to read and classify clip - average time to output/stream clip
        delayPerClip = self.averageTime() - outputSPF*numOfFrames
        timePassed = self.timeFromStarting()  
        delay = (delayPerClip-timePassed) if (delayPerClip > timePassed ) else 0
        return delay 
    
    def setStartingTime(self):
        self.startingTime = cv2.getTickCount()
        
    def timeFromStarting(self):
        return (cv2.getTickCount() - self.startingTime) / cv2.getTickFrequency() 
    
    def hasRecords(self, n=2):
        return len(self.records) == n    



def store_clip(clip, storeDirectory = './recorded_clips/'):

    if not os.path.exists(storeDirectory):
        os.makedirs(storeDirectory)
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    clip_name = 'PV_'+dt_string+'.mp4'
    out = cv2.VideoWriter(storeDirectory+'/'+clip_name,cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640,480))# 30.0 fps
    for frame in clip:
        frame = cv2.resize(frame,(640,480))
        out.write(frame) 
    out.release()
         