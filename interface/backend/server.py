from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi import Request
from interface.backend.controller import Controller, ModelConfig, StartUpConfig
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

url = 'http://localhost:5006'

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost",
    "http://localhost:4200",
    url
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/app", StaticFiles(directory="interface/frontend/"), name="frontend")
t = Jinja2Templates(directory="interface/frontend")

controller = Controller()


@app.get('/')
def index(request: Request):
    return t.TemplateResponse("index.html",{"request":request})

@app.post('/start')
def start(config:StartUpConfig):
    
        controller.start(config)
        return {
                    "stream":url+'/video_feed'
                  }
@app.post("/fstart")
def fstart(video: UploadFile = File(...),clip_size: int= Form(),threshold: int = Form(),memory: int = Form()):
    try:
        contents = video.file.read()
        with open("tmp.mp4", 'wb') as f:
            f.write(contents)
    except Exception:
        return {"message": f"There was an error opening {video.filename}"}
    finally:
        video.file.close()
    
    config = StartUpConfig(source='tmp.mp4',
                           modelConfig=ModelConfig(clip_size=clip_size,memory=memory,threshold=threshold))   
    controller.start(config)
    
    return { "stream":url+'/video_feed' }
        
@app.get('/end')
def end():
        controller.end()
        return {
            "STATUS":True
                }

@app.get('/config')
def modelConfig():
        return controller.getModelConfig()
        
@app.get('/delay')
def delay():
        return {
                "delay":controller.streaming_delay
                }
@app.get('/fps')
def fps():
        return {
                "frame_rate":controller.frame_rate
                }
            
@app.get('/video_feed')
def video_feed():
    return StreamingResponse(controller.stream(), media_type="multipart/x-mixed-replace;boundary=frame")
