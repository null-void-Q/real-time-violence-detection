from model.model import ViolenceModel
from data.video_capture import VideoCapture
from data.output_pipe import OutputPipe
from data.utils import PreformanceTimer
import threading
from pydantic import BaseModel
import os

class Controller():
    
    
    def __init__(self):
        
        self.model = ViolenceModel(clip_size=64, memory=3, threshold=70)
        self.output_pipe = None
        self.video_capture = None
        
        self.preformanceTimer = None
        self.streaming_delay = -1
        self.frame_rate = 0
        
        self.processing_thread = None
        self.stop_flag = None
        

        
  


    def start(self, config):
        
        #wait for previous process to end
        self.end()
        
        self.model.update(config.modelConfig)
        
        self.output_pipe = OutputPipe()
        
        self.preformanceTimer = PreformanceTimer()
        self.streaming_delay = -1
        self.frame_rate = 0
        
        self.video_capture = VideoCapture(video_src=config.source)
        self.video_capture.start_capture_thread()
        
        self.stop_flag = threading.Event()
        self.processing_thread = threading.Thread(target=self.processing_loop)
          
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        
     
    def processing_loop(self):
        
        self.preformanceTimer.setStartingTime()
        while(self.video_capture.isFlowing() and not self.stop_flag.is_set()):
            
            clip = self.video_capture.read_clip(self.model.clip_size)
            
            label = self.model.classify(clip)
            
            self.output_pipe.read_output(clip,label)
            
            self.preformanceTimer.record()
            self.frame_rate = self.preformanceTimer.framerate(self.model.clip_size)
            
            #calculate required delay before streaming / depends on machine preformance
            if(self.preformanceTimer.hasRecords()):# record 2 classifications before calculating delay
                self.streaming_delay = self.preformanceTimer.calculateDelay(self.output_pipe.spf, self.model.clip_size)
                
                # start streaming classified frames
                self.output_pipe.start_after_delay(self.streaming_delay)    
                     
        self.output_pipe.end()
        self.video_capture.end_capture_thread()
        self.delTmpVideo()
        
    
    def end(self):
        if(self.stop_flag):
            self.stop_flag.set()
            self.output_pipe.terminate()
            self.processing_thread.join()
            
            
    def getModelConfig(self):
        return {
            "threshold":self.model.threshold,
            "clip_size":self.model.clip_size,
            "memory":self.model.memory
        } 
           
    def stream(self):
        return self.output_pipe.output_stream()     
    
    def delTmpVideo(self):
        if os.path.exists("tmp.mp4"):
            os.remove("tmp.mp4")
            
class ModelConfig(BaseModel):
    clip_size:int
    memory:int
    threshold:int
        
class StartUpConfig(BaseModel):
    source: str
    modelConfig: ModelConfig   