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
        
        self.processing_loop = None
        self.stop_flag = None
        

        
  


    def start(self, config):
        
        self.model.update(config.modelConfig)
        self.output_pipe = OutputPipe()
        self.preformanceTimer = PreformanceTimer()
        
        self.video_capture = VideoCapture(video_src=config.source)
        self.video_capture.start_capture_thread()
        
        self.stop_flag = threading.Event()
        self.processing_loop = threading.Thread(target=Controller.processing_loop, args=(self.model,
                                                                                         self.video_capture,
                                                                                         self.output_pipe,
                                                                                         self.preformanceTimer,
                                                                                       self.stop_flag))
          
        self.processing_loop.daemon = True
        self.processing_loop.start()
        
        
    @staticmethod            
    def processing_loop(model, video_capture, output_pipe, preformanceTimer,stop_flag):
        
        preformanceTimer.setStartingTime()
        while(video_capture.isFlowing() and not stop_flag.is_set()):
            
            clip = video_capture.read_clip(model.clip_size)
            
            label = model.classify(clip)
            
            preformanceTimer.record()
            
            output_pipe.read_output(clip,label)
            
            #calculate required delay before streaming / depends on machine preformance
            if(preformanceTimer.hasRecords()):# record 2 classifications before calculating delay
                delay = preformanceTimer.calculateDelay(output_pipe.spf, model.clip_size)
                
                # start streaming classified frames
                output_pipe.start_after_delay(delay)    
        
        video_capture.end_capture_thread()
        output_pipe.end()
    
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