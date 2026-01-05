import cv2
from collections import deque
import threading
from utils import PerformanceTimer

class VideoCapture():

    def __init__(self, video_src, trigger_mode=False):
        self.trigger_mode = trigger_mode

        self.cap = self.open_cap(video_src or 0)
        self.buffer = deque([])

        self.stop_flag = threading.Event()
        self.buffer_lock = threading.Lock()
        self.buffer_cv = threading.Condition(self.buffer_lock)

        self.capture_thread = threading.Thread(target=self.capture)

        self.fpsRecord = PerformanceTimer()

    def isPlaying(self):
        if self.trigger_mode:
            return True
        return self.cap.isOpened()


    def capture(self):

        while(self.isPlaying() and not self.stop_flag.is_set()):

            ret, frame = self.cap.read()
            if ret == True:
                with self.buffer_cv:
                        self.buffer.append(frame)
                        self.fpsRecord.record()
                        self.buffer_cv.notify_all()
            else:
                break
        self.cap.release()
        with self.buffer_cv:
            self.buffer_cv.notify_all()

    def trigger_capture(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)#TODO move this to model code
        with self.buffer_cv:
            self.buffer.append(frame)
            self.fpsRecord.record()
            self.buffer_cv.notify_all()

    def read_clip(self,clip_size):
        clip = []

        with self.buffer_cv:
            # if reading is lagging wait for buffer to fill up. unless capture has ended
            while(len(self.buffer) < clip_size and self.isPlaying() and not self.stop_flag.is_set()):
                self.buffer_cv.wait(timeout=0.01)


            while(len(clip) < clip_size and len(self.buffer)):
                clip.append(self.buffer.popleft())


        #loop clip if not enough frames
        while len(clip) < clip_size:
            clip.extend(clip[: clip_size - len(clip) ])

        return clip

    def start_capture_thread(self):
        self.capture_thread.daemon = True
        self.capture_thread.start()

    def end_capture_thread(self):
        if self.capture_thread.is_alive():
            self.capture_thread.join()
    
    def stop(self):
        self.stop_flag.set()
        self.end_capture_thread()

    def isFlowing(self):
        return self.isPlaying() or len(self.buffer)

    def getFPS(self):
        return self.fpsRecord.getFramerate()

    def open_cap(self, video_src):
        if self.trigger_mode:
            return None

        cap = cv2.VideoCapture(video_src)

        if (cap.isOpened()== False):
            print("Error opening video stream or file")
            exit(1)

        return cap
