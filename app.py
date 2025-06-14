import os, uuid, shutil, tempfile, asyncio

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# ──────────────────────────────────────────────────────────────────────────────
# Lazy HF client (unchanged)
def get_hf_client():
    if not hasattr(get_hf_client, "client"):
        try:
            from gradio_client import Client, handle_file

            get_hf_client.client = Client(
                "fernandofurundarena/diffusers-image-outpaint",
                hf_token=os.environ["HUGGINGFACE_HUB_TOKEN"],
            )
            get_hf_client.handle_file = handle_file
        except Exception as e:
            print("⚠️  Warning: could not init HF client:", e)
            get_hf_client.client = None
            get_hf_client.handle_file = None
    return get_hf_client.client, get_hf_client.handle_file


# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_index():
    return HTMLResponse(open("index.html", encoding="utf-8").read())


# ──────────────────────────────────────────────────────────────────────────────
# /api/infer/ – generates 3 variants and returns their URLs
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
    overlap_left: bool = Form(False),
    overlap_right: bool = Form(False),
    overlap_top: bool = Form(True),
    overlap_bottom: bool = Form(True),
):
    # 1) save upload
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(await image.read())
    tmp.flush()

    # 2) get HF client
    hf, handle_file = get_hf_client()
    if hf is None:  # offline dev → just echo input 3×
        os.makedirs("static/results", exist_ok=True)
        urls = []
        for _ in range(3):
            name = f"{uuid.uuid4().hex}.png"
            dest = os.path.join("static", "results", name)
            shutil.copy(tmp.name, dest)
            urls.append(f"/static/results/{name}")
        return {"urls": urls}

    # helper: ONE blocking predict() -> path
    async def run_one():
        loop = asyncio.get_running_loop()

        def _call():
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
                api_name="/infer",  # <-- keep this!
            )
            return out[1]  # full outpaint
        return await loop.run_in_executor(None, _call)

    # 3) fire three in parallel
    outpaint_paths = await asyncio.gather(run_one(), run_one(), run_one())

    # 4) persist + return URLs
    os.makedirs("static/results", exist_ok=True)
    urls = []
    for src in outpaint_paths:
        name = f"{uuid.uuid4().hex}.png"
        dest = os.path.join("static", "results", name)
        shutil.copy(src, dest)
        urls.append(f"/static/results/{name}")

    return {"urls": urls}