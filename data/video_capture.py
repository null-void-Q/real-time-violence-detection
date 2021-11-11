import cv2

class VideoCapture():
    
    def __init__(self, video_src):
        
        self.cap = VideoCapture.start_cap(video_src or 0)
    
    def isPlaying(self):
        return self.cap.isOpened()
    
    def read_clip(self,clip_size):
        clip = []
        
        while(self.isPlaying() and len(clip) < clip_size):
            ret, frame = self.cap.read()
            if ret == True:
                clip.append(frame)                                     
            else:
                self.cap.release()
                break
        
        #loop clip if not enough frames    
        if len(clip) < clip_size:
            clip.extend(clip[: clip_size - len(clip) ])
                 
        return clip        
    
    @staticmethod
    def start_cap(video_src):
        cap = cv2.VideoCapture(video_src) 
        
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
            exit(1)
        
        return cap    