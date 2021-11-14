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
        
        self.processing_loop = None
        
        self.preformanceTimer = PreformanceTimer()

        
  


    def start(self, source):
        
        self.video_capture = VideoCapture(video_src=source)
        self.output_pipe = OutputPipe()
        
        self.processing_loop = threading.Thread(target=Controller.processing_loop, args=(self.model,
                                                                                         self.video_capture,
                                                                                         self.output_pipe,
                                                                                         self.preformanceTimer))
        self.processing_loop.daemon = True
        self.processing_loop.start()
        
    @staticmethod            
    def processing_loop(model, video_capture, output_pipe, preformanceTimer):
        while(video_capture.isPlaying()):

            clip = video_capture.read_clip(model.clip_size)
    
            label = model.classify(clip)
            
            preformanceTimer.record()
            
            output_pipe.read_output(clip,label)

        output_pipe.end()
        
    def stream(self):
        return self.output_pipe.output_stream()
    
class Source(BaseModel):
    source: str    