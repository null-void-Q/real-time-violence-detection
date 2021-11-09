import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})

@app.post('/start')
def start(source):
    return {'result':'yay'}

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(None, media_type="multipart/x-mixed-replace;boundary=frame")

if __name__ == '__main__':
    uvicorn.run("fastapiserver:app", host="0.0.0.0", port=5000, access_log=False)