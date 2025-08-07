from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from yolov8_deepsort import process_video

app = FastAPI()

# Setup folders
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mount static directories
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_name = "output_" + file.filename
    output_path = os.path.join(UPLOAD_FOLDER, output_name)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    process_video(input_path, output_path)

    return RedirectResponse(url=f"/view/{output_name}", status_code=302)

@app.get("/view/{filename}", response_class=HTMLResponse)
async def view_result(request: Request, filename: str):
    return templates.TemplateResponse("view.html", {
        "request": request,
        "video_name": filename
    })
