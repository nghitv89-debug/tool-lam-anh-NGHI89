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
        # 1. Chuyển ảnh tải lên thành Data URI (Dùng để gửi cho AI)
        file_bytes = await file.read()
        mime_type = file.content_type or "image/jpeg"
        
        # Tạo base64 cho ảnh gốc để hiển thị xem trước và gửi cho AI
        base64_original = base64.b64encode(file_bytes).decode("utf-8")
        original_data_uri = f"data:{mime_type};base64,{base64_original}"

        # 2. Tối ưu Prompt (Tự động thêm từ khóa chất lượng)
        full_prompt = f"high quality photorealistic studio photo, {prompt}, sharp focus, preserve original facial structure and identity"
        negative_prompt = "ugly, distorted, blurry, low quality, bad anatomy, altered face identity, cartoon, illustration"

        # 3. Gọi mô hình Stable Diffusion XL (SDXL) - CÁCH GỌI MỚI, ỔN ĐỊNH DỨT ĐIỂM
        max_retries = 3
        output = None
        
        # --- CẢI TIẾN QUAN TRỌNG: Gọi trực tiếp bằng tên mô hình, không dùng hash ---
        # Điều này giúp Replicate tự động chọn phiên bản tốt nhất hiện có.
        model_name = "stability-ai/sdxl"
        # ----------------------------------------------------------------------

        for attempt in range(max_retries):
            try:
                # Gọi mô hình với tham số "input" chuẩn của SDXL
                output = replicate.run(
                    model_name,
                    input={
                        "image": original_data_uri,
                        "prompt": full_prompt,
                        "negative_prompt": negative_prompt,
                        "prompt_strength": prompt_strength,
                        "num_inference_steps": 30
                    }
                )
                break  # Thành công thì thoát vòng lặp retries
            except Exception as req_err:
                # Nếu dính lỗi Rate Limit (429), đợi 6 giây rồi thử lại
                if "429" in str(req_err) and attempt < max_retries - 1:
                    time.sleep(6)
                    continue
                else:
                    # Nếu là lỗi khác (như API key), ném lỗi ra ngoài luôn
                    raise req_err

        # 4. Lấy URL ảnh kết quả (Xử lý cả dạng list và dạng string)
        if isinstance(output, list) and len(output) > 0:
            result_url = str(output[0])
        elif output:
            result_url = str(output)
        else:
            raise Exception("AI không trả về kết quả.")

        # 5. Trả về cả URL ảnh gốc (base64) và URL ảnh AI render
        return JSONResponse({
            "status": "success", 
            "image_url": result_url, 
            "original_url": original_data_uri
        })

    except Exception as e:
        # Nếu có lỗi (kể cả lỗi API), trả về thông báo lỗi chi tiết cho client
        return JSONResponse({
            "status": "error", 
            "message": f"AI Studio Error: {str(e)}"
        }, status_code=500)
