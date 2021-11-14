import cv2
import os
from datetime import datetime

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

def anotate_clip(clip,label):
    out_clip = []
    for i,frame in enumerate(clip):
        out_clip.append(write_label(frame,label))
    return  out_clip        

def write_label(frame, prediction):
    

    text_location = (500,740)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color =(0, 192, 7)

    lineType               = 2
    bordersize             = 30
    borderColor            = (0, 0, 0)


    label = prediction

    
    frame = cv2.copyMakeBorder(frame,
        top=0,
        bottom=bordersize,
        left=0,
        right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=[*borderColor])
    
            
    cv2.putText(frame,label['label']+' '+str(label['score']), 
        text_location, 
        font, 
        font_scale,
        font_color,
        lineType,)
    
    # timestamp = datetime.datetime.now()
    # cv2.putText(frame, timestamp.strftime(
	# 		"%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
	# 		cv2.FONT_HERSHEY_SIMPLEX, 0.65, (32, 64, 255),1)

    # frame = cv2.copyMakeBorder(frame,
    #         top=bordersize,
    #         bottom=bordersize,
    #         left=bordersize,
    #         right=bordersize,
    #         borderType=cv2.BORDER_CONSTANT,
    #         value=[*borderColor])
            
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
    
    def framerate(self,numOfFrames):
        if not len(self.records):
            return 0
        return ( numOfFrames/self.averageTime() )
        
        
         