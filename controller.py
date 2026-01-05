from model.model import ViolenceModel
from video_capture import VideoCapture
from output_pipe import OutputPipe
from utils import PerformanceTimer
import threading
import cv2
from pydantic import BaseModel
import os

class Controller():


    def __init__(self):

        self.model = ViolenceModel(clip_size=32, memory=3, threshold=70)
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

        self.preformanceTimer = PerformanceTimer()
        self.streaming_delay = -1
        self.frame_rate = 0

        self.video_capture = VideoCapture(video_src=config.source,trigger_mode=True)
        if not config.source == "Webcam Streaming":
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
            self.frame_rate = self.preformanceTimer.getFramerate(self.model.clip_size)

            #calculate required delay before streaming / depends on machine preformance
            if(self.preformanceTimer.hasRecords(n=1)):# record n classifications before calculating delay
                self.streaming_delay = self.preformanceTimer.calculateDelay(self.output_pipe.spf, self.model.clip_size)
                # start streaming labeled frames
                self.output_pipe.start_after_delay(self.streaming_delay)
            #if video is shorter than n records just start
            elif not self.video_capture.isFlowing():
                self.output_pipe.start()

        self.output_pipe.end()
        self.video_capture.stop()
        self.delTmpVideo()

    def end(self):
        if(self.stop_flag):
            self.stop_flag.set()
            self.output_pipe.terminate()
            self.video_capture.stop()
            self.processing_thread.join()
        self.output_pipe = None
        self.video_capture = None

        self.preformanceTimer = None
        self.streaming_delay = -1
        self.frame_rate = 0

        self.processing_thread = None
        self.stop_flag = None

    def getModelConfig(self):
        return {
            "threshold":self.model.threshold,
            "clip_size":self.model.clip_size,
            "memory":self.model.memory
        }


    def frame_stream(self):
        return self.output_pipe.frame_stream()

    def labeled_stream(self):
        return self.output_pipe.labeled_frame_stream()

    def trigger_capture(self, frame):
        self.video_capture.trigger_capture(frame)

    def get_playback_FPS(self):
        if not self.output_pipe:
            return -1
        return self.output_pipe.current_fps

    def get_capture_FPS(self):
        if not self.video_capture:
            return -1
        return self.video_capture.getFPS()

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
