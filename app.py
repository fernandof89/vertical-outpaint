from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from gradio_client import Client, handle_file
import tempfile

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")  # ← and this

hf = Client("fernandofurundarena/diffusers-image-outpaint")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r") as f:
        return HTMLResponse(f.read())

@app.post("/api/infer/")
async def infer(
    image: UploadFile = File(...),
    width: int = Form(720),
    height: int = Form(1280),
    overlap_percentage: int = Form(10),
    num_inference_steps: int = Form(8),
    resize_option: str = Form("Full"),
    custom_resize_percentage: int = Form(50),
    prompt_input: str = Form(""),
    alignment: str = Form("Middle"),
    overlap_left: bool = Form(True),
    overlap_right: bool = Form(True),
    overlap_top: bool = Form(True),
    overlap_bottom: bool = Form(True),
):
    # write upload to temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(await image.read())
    tmp.flush()

    # call the HF /infer endpoint
    out = hf.predict(
        image=handle_file(tmp.name),
        width=width,
        height=height,
        overlap_percentage=overlap_percentage,
        num_inference_steps=num_inference_steps,
        resize_option=resize_option,
        custom_resize_percentage=custom_resize_percentage,
        prompt_input=prompt_input,
        alignment=alignment,
        overlap_left=overlap_left,
        overlap_right=overlap_right,
        overlap_top=overlap_top,
        overlap_bottom=overlap_bottom,
        api_name="/infer"
    )
    # out is a tuple: (masked_background, full_outpaint)
    _, full_outpaint = out

    # return the actual outpainted image
    return FileResponse(full_outpaint, media_type="image/png")