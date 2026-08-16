import os
import base64
import fal_client
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/process-photo")
async def process_photo(
    file: UploadFile = File(...),
    category: str = Form("vest_nam"),
    sub_option: str = Form(""),
    bg_color: str = Form("#003399")
):
    try:
        # Chuyển đổi ảnh sang dạng Base64 Data URI để gửi trực tiếp cho AI
        file_bytes = await file.read()
        base64_image = base64.b64encode(file_bytes).decode("utf-8")
        mime_type = file.content_type or "image/jpeg"
        data_uri = f"data:{mime_type};base64,{base64_image}"

        # Chạy AI CodeFormer tối ưu nét mặt & trang phục
        result = fal_client.subscribe(
            "fal-ai/codeformer",
            arguments={
                "image_url": data_uri,
                "fidelity": 0.85,
                "face_upsample": True,
                "background_enhance": True
            }
        )
        return JSONResponse({"status": "success", "image_url": result["image"]["url"]})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
