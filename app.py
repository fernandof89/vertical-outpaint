import os
import uuid
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Don’t import Client at top-level to avoid blocking on startup
# from gradio_client import Client, handle_file

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def get_hf_client():
    """
    Lazily import and return the Gradio Client.
    If initialization fails (e.g. offline), returns None as a stub.
    """
    if not hasattr(get_hf_client, "client"):
        try:
            from gradio_client import Client, handle_file
            # you can add request_timeout or max_retries here if needed:
            get_hf_client.client = Client("fernandofurundarena/diffusers-image-outpaint", hf_token=os.environ["HUGGINGFACE_HUB_TOKEN"])
            get_hf_client.handle_file = handle_file
        except Exception as e:
            # failed to reach HF space — stub it
            print("⚠️  Warning: could not init HF client:", e)
            get_hf_client.client = None
            get_hf_client.handle_file = None
    return get_hf_client.client, get_hf_client.handle_file

@app.get("/", response_class=HTMLResponse)
async def read_index():
    return HTMLResponse(open("index.html", "r", encoding="utf-8").read())


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
    # 1) Save the upload to a temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(await image.read())
    tmp.flush()

    # 2) Get (or stub) the HF client
    hf, handle_file = get_hf_client()

    if hf is None:
        # LOCAL-DEV FALLBACK: Just copy the input to results so you can test UI
        os.makedirs("static/results", exist_ok=True)
        out_name = f"{uuid.uuid4().hex}.png"
        out_path = os.path.join("static", "results", out_name)
        shutil.copy(tmp.name, out_path)
        return {"url": f"/static/results/{out_name}"}

    # 3) Otherwise, call the real /infer endpoint on the HF Space
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
        api_name="/infer",
    )
    _, full_outpaint = out  # (masked, full)

    # 4) Persist into static/results and return its URL
    os.makedirs("static/results", exist_ok=True)
    out_name = f"{uuid.uuid4().hex}.png"
    out_path = os.path.join("static", "results", out_name)
    shutil.copy(full_outpaint, out_path)

    return {"url": f"/static/results/{out_name}"}