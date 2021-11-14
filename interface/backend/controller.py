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
        
        

        
  


    def start(self, source):
        
        self.video_capture = VideoCapture(video_src=source)
        self.output_pipe = OutputPipe()
        self.preformanceTimer = PreformanceTimer()
        
        self.processing_loop = threading.Thread(target=Controller.processing_loop, args=(self.model,
                                                                                         self.video_capture,
                                                                                         self.output_pipe,
                                                                                         self.preformanceTimer))
        self.processing_loop.daemon = True
        self.processing_loop.start()
        
    @staticmethod            
    def processing_loop(model, video_capture, output_pipe, preformanceTimer):
        
        preformanceTimer.setStartingTime()
        while(video_capture.isPlaying()):

            clip = video_capture.read_clip(model.clip_size)
    
            label = model.classify(clip)
            
            preformanceTimer.record()
            
            output_pipe.read_output(clip,label)
            
            #calculate required delay before streaming / depends on machine preformance
            if(preformanceTimer.hasRecords()):# record 2 classifications before calculating delay
                delay = preformanceTimer.calculateDelay(output_pipe.spf, model.clip_size)
                
                # start streaming classified frames
                output_pipe.start_after_delay(delay)    

        output_pipe.end()
        
    def stream(self):
        return self.output_pipe.output_stream()
    
class Source(BaseModel):
    source: str    