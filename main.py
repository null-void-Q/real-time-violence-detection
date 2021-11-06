from model.model import ViolenceModel
from data.video_capture import VideoCapture
from data.output_pipe import OutputPipe

VID_SRC = './test_video.mp4'

clip_size = 64

save_result = True

confidence_threshold = 70
memory = 3

def main():
    
    model = ViolenceModel(clip_size=clip_size, memory=memory, threshold=confidence_threshold)
    video_capture = VideoCapture(video_src=VID_SRC)
    output_pipe = OutputPipe()
    output_pipe.start_stream()
    
    while(video_capture.isPlaying()):

        clip = video_capture.read_clip(clip_size)
   
        label = model.classify(clip)
        
        print(label)
         
        output_pipe.read_output(clip,label)
        

    output_pipe.end()
    
        
    
if __name__ == '__main__':
    main()