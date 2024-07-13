from fastapi import FastAPI, HTTPException
from inference2 import infer
from pydantic import BaseModel
from fastapi.responses import FileResponse
import os

app = FastAPI()

class RequestModel(BaseModel):
    text: str
    outname: str

@app.post("/synthesize/")
async def synthesize_audio(request: RequestModel):
    try:
        # Assuming your infer function returns the filename of the synthesized audio
        filename, generation_time = infer(request.text, request.outname)

        # Ensure the file exists
        if not os.path.exists(request.outname):
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(filename, media_type='audio/wav', filename=request.outname)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
