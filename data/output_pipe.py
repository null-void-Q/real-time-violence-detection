import cv2
import numpy as np
import time
from collections import deque
from .utils import anotate_clip
import threading

class OutputPipe():
    def __init__(self, fps = 30):
        
        self.buffer = deque([])
        self.labels = []
        
        self.fps = fps
        self.spf = 1/fps
        
        self.stop_flag = threading.Event()
        self.start_flag = threading.Event()
        self.buffer_lock = threading.Lock()
    
    def read_output(self, clip,label):
        clip = anotate_clip(clip,label)
        with self.buffer_lock:
            self.buffer.extend(clip)
        self.labels.extend([label]*len(clip))
        
    
    
    def output_stream(self):
        
        #timing for output fps
        prev  = 0
        current  = 0
        
        #wait until there is frames to stream
        while(not self.start_flag.is_set()): pass
        
        #main output loop   
        while(not self.stop_flag.is_set() or len(self.buffer)):
            while(len(self.buffer)):
                current = cv2.getTickCount()
                time_since_last_frame = (current - prev)/ cv2.getTickFrequency()  
                if time_since_last_frame >= self.spf:
                    prev = cv2.getTickCount()
                    with self.buffer_lock:     
                        frame = self.encode_frame(self.buffer.popleft())
                        
                        yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')      
  
        # return black frame when done
        yield (b'--frame\r\n'
                 b'Content-Type: image/jpeg\r\n\r\n' + self.encode_frame(np.zeros((1280,720)))  + b'\r\n') 
        
    def encode_frame(self,frame):
        frame = cv2.resize(frame,(1280,720))      
        ret, buf = cv2.imencode('.jpg', frame)
        frame = buf.tobytes()
        return frame
    
    
    def start(self):
        self.start_flag.set()  
        
    def end(self):
        self.stop_flag.set()
    
    def start_after_delay(self,delay):
        time.sleep(delay)
        self.start()    