import os
import base64
import time
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
    prompt: str = Form(...),
    prompt_strength: float = Form(0.65)
):
    try:
        # 1. Chuyển ảnh tải lên thành Data URI
        file_bytes = await file.read()
        mime_type = file.content_type or "image/jpeg"
        
        # Tạo bản base64 cho ảnh gốc để trả về và hiển thị ngay trên web
        base64_original = base64.b64encode(file_bytes).decode("utf-8")
        original_data_uri = f"data:{mime_type};base64,{base64_original}"

        # 2. Chuẩn hóa Prompt
        full_prompt = f"high quality photorealistic photo, {prompt}, sharp focus, preserve original facial structure and identity"
        negative_prompt = "ugly, distorted, blurry, low quality, bad anatomy, altered face identity"

        # 3. Tối ưu mô hình Stable Diffusion XL (SDXL) - Tự động lấy version mới nhất
        max_retries = 3
        output = None
        
        # Sử dụng phiên bản SDXL ổn định nhất, Replicate sẽ tự chọn hash ổn định.
        model_version = "stability-ai/sdxl:39ed52f2a78e9323042180715c915d6f4b0d718fe50720fe0e5a7a0105755924"
        
        # Nếu phiên bản cố định bị lỗi 422, hãy thử gọi trực tiếp "stability-ai/sdxl"
        # model_version = "stability-ai/sdxl" # Bỏ comment dòng này và comment dòng trên nếu lỗi 422 lặp lại

        for attempt in range(max_retries):
            try:
                output = replicate.run(
                    model_version,
                    input={
                        "image": original_data_uri,
                        "prompt": full_prompt,
                        "negative_prompt": negative_prompt,
                        "prompt_strength": prompt_strength,
                        "num_inference_steps": 30
                    }
                )
                break
            except Exception as req_err:
                if "429" in str(req_err) and attempt < max_retries - 1:
                    time.sleep(6)
                    continue
                else:
                    raise req_err

        # Lấy URL ảnh kết quả
        result_url = str(output[0]) if isinstance(output, list) else str(output)
        
        # 4. Trả về cả URL ảnh gốc (base64) và URL ảnh AI
        return JSONResponse({
            "status": "success", 
            "image_url": result_url, 
            "original_url": original_data_uri # Trả về ảnh gốc để so sánh
        })

    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)
