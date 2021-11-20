from model.model import ViolenceModel
from data.video_capture import VideoCapture
from data.output_pipe import OutputPipe
from data.utils import PreformanceTimer
import threading
from pydantic import BaseModel


class Controller():
    
    
    def __init__(self, clip_size=64, confidence_threshold=70, memory=3):
        
        self.model = ViolenceModel(clip_size=clip_size, memory=memory, threshold=confidence_threshold)
        self.output_pipe = None
        self.video_capture = None
        
        self.preformanceTimer = None
        self.streaming_delay = -1
        
        self.processing_thread = None
        self.stop_flag = None
        

        
  


    def start(self, config):
        
        self.model.update(config.modelConfig)
        self.output_pipe = OutputPipe()
        self.preformanceTimer = PreformanceTimer()
        
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
            
            self.preformanceTimer.record()
            
            self.output_pipe.read_output(clip,label)
            
            #calculate required delay before streaming / depends on machine preformance
            if(self.preformanceTimer.hasRecords()):# record 2 classifications before calculating delay
                delay = self.preformanceTimer.calculateDelay(self.output_pipe.spf, self.model.clip_size)
                
                # start streaming classified frames
                self.output_pipe.start_after_delay(delay)    
        
        self.video_capture.end_capture_thread()
        self.output_pipe.end()
    
    def end(self):
        self.stop_flag.set()
        
    def stream(self):
        return self.output_pipe.output_stream()
    
class ModelConfig(BaseModel):
    clip_size:int
    memory:int
    threshold:int
        
class StartUpConfig(BaseModel):
    source: str
    modelConfig: ModelConfig   