import uvicorn


def serve():
    uvicorn.run("interface.backend.server:app", host="0.0.0.0", port=5000, access_log=False)

    
        
    
if __name__ == '__main__':
    serve()