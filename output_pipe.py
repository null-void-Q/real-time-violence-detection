import cv2
import numpy as np
import time
from collections import deque
from utils import write_label
import threading

class OutputPipe():
    def __init__(self, fps = 30):

        self.buffer = deque([])
        self.labels = deque([])

        self.fps = fps
        self.spf = 1/fps
        self.current_fps = -1

        self.stop_flag = threading.Event()
        self.start_flag = threading.Event()
        self.terminate_flag = threading.Event()

        self.buffer_lock = threading.Lock()
        self.buffer_cv = threading.Condition(self.buffer_lock)

    def read_output(self, clip,label):
        with self.buffer_cv:
            self.buffer.extend(clip)
            self.labels.extend([label]*len(clip))
            self.buffer_cv.notify_all()

    def frame_stream(self):
        
        #wait until there is frames to stream
        with self.buffer_cv:
            self.buffer_cv.wait_for(lambda: self.start_flag.is_set() or self.terminate_flag.is_set())
        if self.terminate_flag.is_set():
            return

        #main output loop
        next_emit = time.perf_counter()
        last_emit_time = None

        while True:
            #wait for data or terminate_flag
            with self.buffer_cv:
                self.buffer_cv.wait_for(
                    lambda: len(self.buffer) > 0 or self.stop_flag.is_set() or self.terminate_flag.is_set()
                )

                if self.terminate_flag.is_set():
                    break
                if len(self.buffer) == 0 and self.stop_flag.is_set():
                    #stop only after all frames are out
                    break

                #pop frame
                frame = self.buffer.popleft()
                label = self.labels.popleft()

            #pace fps
            now = time.perf_counter()
            sleep_time = next_emit - now
            if sleep_time > 0:
                time.sleep(sleep_time)
                now = time.perf_counter()

            #update next emit time - avoid drift on long delays
            if now > next_emit + self.spf:
                #late -> reset schedule to now
                next_emit = now + self.spf
            else:
                next_emit += self.spf

            #update current_fps
            if last_emit_time is not None:
                dt = max(now - last_emit_time, 1e-6)
                self.current_fps = 1.0 / dt
            last_emit_time = now

            #format and send frame and label
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame, label

        # black frame at the end
        black = np.zeros((720, 1280, 3), dtype=np.uint8)
        yield black, label

    def prepare_frame(self,frame,label):
        frame = cv2.resize(frame,(1280,720))
        frame = write_label(frame,label)
        return frame

    def encode_frame(self,frame):
        ret, buf = cv2.imencode('.jpg', frame)
        frame = buf.tobytes()
        return frame


    def start(self):
        self.start_flag.set()
        with self.buffer_cv:
            self.buffer_cv.notify_all()

    def end(self):
        self.stop_flag.set()
        with self.buffer_cv:
            self.buffer_cv.notify_all()

    def terminate(self):
        self.terminate_flag.set()
        with self.buffer_cv:
            self.buffer_cv.notify_all()

    def start_after_delay(self, delay):
        timer = threading.Timer(delay, self.start)
        timer.daemon = True
        timer.start()


    def labeled_frame_stream(self):

        #timing for output fps
        prev  = 0
        current  = 0

        #wait until there is frames to stream
        while not self.terminate_flag.is_set():
            if self.start_flag.wait(timeout=0.05):
                break

        #main output loop
        while((not self.stop_flag.is_set() or len(self.buffer)) and not self.terminate_flag.is_set()):
            while(len(self.buffer) and not self.terminate_flag.is_set()):
                current = cv2.getTickCount()
                time_since_last_frame = (current - prev)/ cv2.getTickFrequency()
                if time_since_last_frame >= self.spf:
                    self.buffer_lock.acquire(blocking=False)
                    prev = cv2.getTickCount()
                    try:
                        frame = self.buffer.popleft()
                        label = self.labels.popleft()
                    finally:
                        self.buffer_lock.release()

                    frame = self.prepare_frame(frame,label)
                    frame = self.encode_frame(frame)
                    yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # return black frame when done
        yield (b'--frame\r\n'
                 b'Content-Type: image/jpeg\r\n\r\n' + self.encode_frame(np.zeros((720,1280)))  + b'\r\n')
