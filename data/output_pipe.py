import cv2
import numpy as np
import time
from collections import deque
from .utils import write_label
import threading

class OutputPipe():
    def __init__(self, fps = 30):
        
        self.buffer = deque([])
        self.labels = deque([])
        
        self.fps = fps
        self.spf = 1/fps
        
        self.stop_flag = threading.Event()
        self.start_flag = threading.Event()
        self.terminate_flag = threading.Event()
        
        self.buffer_lock = threading.Lock()
    
    def read_output(self, clip,label):
        with self.buffer_lock:
            self.buffer.extend(clip)
            self.labels.extend([label]*len(clip))
        
    
    
    def output_stream(self):
        
        #timing for output fps
        prev  = 0
        current  = 0
        
        #wait until there is frames to stream
        while(not self.start_flag.is_set() and not self.terminate_flag.is_set()): pass
        
        #main output loop   
        while((not self.stop_flag.is_set() or len(self.buffer)) and not self.terminate_flag.is_set()):
            while(len(self.buffer) and not self.terminate_flag.is_set()):
                current = cv2.getTickCount()
                time_since_last_frame = (current - prev)/ cv2.getTickFrequency()  
                if time_since_last_frame >= self.spf:
                    prev = cv2.getTickCount()
                    self.buffer_lock.acquire(blocking=False)
                    try:
                        frame = self.buffer.popleft()
                        label = self.labels.popleft()
                    finally:
                        self.buffer_lock.release()    
                        
                    frame = self.prepare_frame(frame,label)
                    frame = self.encode_frame(frame)
                    yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')      
  
        # return black frame when done
        yield (b'--frame\r\n'
                 b'Content-Type: image/jpeg\r\n\r\n' + self.encode_frame(np.zeros((720,1280)))  + b'\r\n') 
    
    def prepare_frame(self,frame,label):
        frame = cv2.resize(frame,(1280,720))
        frame = write_label(frame,label)
        return frame   
            
    def encode_frame(self,frame):   
        ret, buf = cv2.imencode('.jpg', frame)
        frame = buf.tobytes()
        return frame
    
    
    def start(self):
        self.start_flag.set()  
        
    def end(self):
        self.stop_flag.set()
        
    def terminate(self):
        self.terminate_flag.set()
        
    def start_after_delay(self,delay):
        
        def delayed_start(delay,self):
            time.sleep(delay)
            self.start()   
        
        delay_thread = threading.Thread(target=delayed_start,args=(delay,self))          
        delay_thread.daemon = True
        delay_thread.start()
 