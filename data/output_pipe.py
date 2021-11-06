import cv2
from collections import deque
import os
from datetime import datetime
from .utils import anotate_clip
import threading

class OutputPipe():
    def __init__(self):
        
        self.buffer = deque([])
        self.labels = []
        
        
        self.stop_flag = threading.Event()
        self.buffer_lock = threading.Lock()
        self.stream = threading.Thread(target=OutputPipe.output_stream, args=(self.buffer,self.buffer_lock,self.stop_flag))
  

    
    def read_output(self, clip,label):
        clip = anotate_clip(clip,label)
        with self.buffer_lock:
            self.buffer.extend(clip)
        self.labels.extend([label]*len(clip))
        
    
    @staticmethod
    def output_stream(buffer, buffer_lock, stop_flag):

        out = cv2.VideoWriter(OutputPipe.get_record_path(),cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (640,480))# 30.0 fps
        while(not stop_flag.is_set()):
            if len(buffer):
                with buffer_lock:                
                    frame = buffer.popleft()
                frame = cv2.resize(frame,(640,480))
                out.write(frame)
        # flush latest remaining frames
        while(len(buffer)):       
            frame = buffer.popleft()
            frame = cv2.resize(frame,(640,480))
            out.write(frame)         
        out.release()
    
    def start_stream(self):
        self.stream.daemon = True
        self.stream.start()
        
    def end(self):
        self.stop_flag.set()
        self.stream.join(5.0)
        exit(0)
    
    @staticmethod    
    def get_record_path(storeDirectory = './recorded_clips/'):
        if not os.path.exists(storeDirectory):
            os.makedirs(storeDirectory)
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        clip_name = 'PV_'+dt_string+'.mp4'
        return storeDirectory+'/'+clip_name        