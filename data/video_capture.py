import cv2
from collections import deque
import threading

class VideoCapture():
    
    def __init__(self, video_src):
        
        self.cap = VideoCapture.start_cap(video_src or 0)
        self.buffer = deque([])
        
        self.stop_flag = threading.Event()
        self.buffer_lock = threading.Lock()
        self.capture_thread = threading.Thread(target=self.capture)

        
    def isPlaying(self):
        return self.cap.isOpened()
    

    def capture(self):
        
        while(self.isPlaying() and not self.stop_flag.is_set()):
            
            ret, frame = self.cap.read()
            if ret == True:
                self.buffer_lock.acquire(blocking=False)
                try:
                    self.buffer.append(frame)  
                finally:
                    self.buffer_lock.release()                                   
            else:
                break
        self.cap.release()          
    
    def read_clip(self,clip_size):
        clip = []
        
        # if reading is lagging wait for buffer to fill up. unless capture has ended
        while(len(self.buffer) < clip_size and self.isPlaying()):pass
        
            
        while(len(clip) < clip_size and len(self.buffer)):
            with self.buffer_lock:
                clip.append(self.buffer.popleft())                                     

                 
        #loop clip if not enough frames    
        while len(clip) < clip_size:
            clip.extend(clip[: clip_size - len(clip) ])
                    
        return clip
            
    def start_capture_thread(self):
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
    def end_capture_thread(self):
        self.stop_flag.set()
        self.capture_thread.join()
    
    def isFlowing(self):
        return self.isPlaying() or len(self.buffer)
        
    @staticmethod
    def start_cap(video_src):
        cap = cv2.VideoCapture(video_src) 
        
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
            exit(1)
        
        return cap    