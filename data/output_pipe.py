import cv2
import numpy as np
from collections import deque
from .utils import anotate_clip
import threading

class OutputPipe():
    def __init__(self):
        
        self.buffer = deque([])
        self.labels = []
        
        
        self.stop_flag = threading.Event()
        self.buffer_lock = threading.Lock()
  
        self.blank_frame = self.encode_frame(np.zeros((640,480)))
    
    def read_output(self, clip,label):
        clip = anotate_clip(clip,label)
        with self.buffer_lock:
            self.buffer.extend(clip)
        self.labels.extend([label]*len(clip))
        
    
    
    def output_stream(self):
        last  = cv2.getTickCount()
        current  = cv2.getTickCount()    
        while(not self.stop_flag.is_set()):
            current = cv2.getTickCount()
            frame_time = (current - last)/ cv2.getTickFrequency()
            if len(self.buffer) and frame_time > 0.03:# 30 fps
                last= cv2.getTickCount()
                with self.buffer_lock:                
                    frame = self.encode_frame(self.buffer.popleft())
                yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n') 
        # flush remaining frames
        while(len(self.buffer)):
            current = cv2.getTickCount()
            frame_time = (current - last)/ cv2.getTickFrequency()  
            if frame_time > 0.03:# 30 fps
                last= cv2.getTickCount()     
                frame = self.encode_frame(self.buffer.popleft())
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')        

        
    def encode_frame(self,frame):
        frame = cv2.resize(frame,(640,480))      
        ret, buf = cv2.imencode('.jpg', frame)
        frame = buf.tobytes()
        return frame
    
    def start_stream(self):
        self.stream.daemon = True
        self.stream.start()
        
    def end(self):
        self.stop_flag.set()