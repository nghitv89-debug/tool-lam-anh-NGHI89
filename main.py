import os
import base64
import replicate
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/process-photo")
async def process_photo(
    file: UploadFile = File(...),
    category: str = Form("anh_the"),
    sub_option: str = Form(""),
    bg_color: str = Form("#003399")
):
    try:
        # Chuyển ảnh tải lên sang dạng Data URI
        file_bytes = await file.read()
        base64_image = base64.b64encode(file_bytes).decode("utf-8")
        mime_type = file.content_type or "image/jpeg"
        data_uri = f"data:{mime_type};base64,{base64_image}"

        # Gọi trực tiếp tên model sczhou/codeformer (không cố định hash version)
        output = replicate.run(
            "sczhou/codeformer",
            input={
                "image": data_uri,
                "codeformer_fidelity": 0.85,
                "background_enhance": True,
                "face_upsample": True,
                "upscale": 2
            }
        )
        
        # Xử lý URL kết quả trả về từ Replicate
        if isinstance(output, list):
            result_url = str(output[0])
        else:
            result_url = str(output)

        return JSONResponse({"status": "success", "image_url": result_url})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
