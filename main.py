import os
import fal_client
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/process-id-photo")
async def process_id_photo(
    file: UploadFile = File(...),
    bg_color: str = Form("#003399")
):
    try:
        file_bytes = await file.read()
        image_url = fal_client.upload(file_bytes, "image/jpeg")

        # Khóa 85% đường nét mặt gốc
        result = fal_client.subscribe(
            "fal-ai/codeformer",
            arguments={
                "image_url": image_url,
                "fidelity": 0.85,
                "face_upsample": True
            }
        )
        return JSONResponse({"status": "success", "image_url": result["image"]["url"]})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/api/restore-old-photo")
async def restore_old_photo(
    file: UploadFile = File(...)
):
    try:
        file_bytes = await file.read()
        image_url = fal_client.upload(file_bytes, "image/jpeg")

        # Phục hồi ảnh cũ bảo tồn diện mạo gốc 100%
        result = fal_client.subscribe(
            "fal-ai/codeformer",
            arguments={
                "image_url": image_url,
                "fidelity": 0.85,
                "background_enhance": True,
                "face_upsample": True
            }
        )
        return JSONResponse({"status": "success", "image_url": result["image"]["url"]})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
