from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from interface.backend.controller import Controller, StartUpConfig
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost",
    "http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


templates = Jinja2Templates(directory="templates")

controller = Controller()



@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})

@app.post('/start')
def start(config:StartUpConfig):
    
        controller.start(config)
        return {
                    "stream":'http://localhost:5000/video_feed'
                  }
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
            
@app.get('/video_feed')
def video_feed():
    return StreamingResponse(controller.stream(), media_type="multipart/x-mixed-replace;boundary=frame")
