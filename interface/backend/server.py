import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from interface.backend.controller import Controller

app = FastAPI()
templates = Jinja2Templates(directory="templates")

controller = Controller()


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})

@app.post('/start')
def start(source):
    return {'result':'yay'}

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(None, media_type="multipart/x-mixed-replace;boundary=frame")
