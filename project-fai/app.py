"""
================================================================================
AI Crop Advisor – Modern Web Application
================================================================================
A high-performance, responsive AI web application for crop disease classification.
Featuring a custom responsive frontend with nature-inspired agronomic styling,
interactive SVG confidence gauges, and real-time inference.

Backend prediction and advisory logic in utils.py remains 100% untouched.

Usage:
    python app.py
================================================================================
"""

import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Optimize cache and silence non-critical warnings
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from PIL import Image
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Add project directory to Python path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Import the original, UNCHANGED prediction utility
from utils import predict_disease

# Directories
WORKSPACE_ROOT = CURRENT_DIR.parent
DATA_DIR = WORKSPACE_ROOT / "data"
TEMPLATES_DIR = CURRENT_DIR / "templates"
STATIC_DIR = CURRENT_DIR / "static"

# FastAPI App
app = FastAPI(
    title="AI Crop Advisor",
    description="Intelligent Crop Disease Classification & IPM Advisory",
    version="2.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Verified sample images for quick testing (checks static/samples, falls back to data/images)
def _resolve_sample(filename: str, fallback_path: Path) -> Path:
    local_sample = STATIC_DIR / "samples" / filename
    return local_sample if local_sample.exists() else fallback_path

SAMPLE_IMAGES = [
    {
        "id": 0,
        "path": _resolve_sample("cotton_bacterial_blight.jpg", DATA_DIR / "images" / "cotton" / "cotton" / "bacterial_blight" / "bact1.jpg"),
        "label": "Cotton – Bacterial Blight",
        "crop": "Cotton"
    },
    {
        "id": 1,
        "path": _resolve_sample("paddy_blast.jpg", DATA_DIR / "images" / "paddy" / "train_images" / "blast" / "100074.jpg"),
        "label": "Paddy – Blast",
        "crop": "Paddy"
    },
    {
        "id": 2,
        "path": _resolve_sample("chilli_cercospora.jpg", DATA_DIR / "images" / "chilli" / "cropped" / "cropped" / "cercospora" / "IMG_3420.JPG"),
        "label": "Chilli – Cercospora",
        "crop": "Chilli"
    },
    {
        "id": 3,
        "path": _resolve_sample("chilli_healthy.jpg", DATA_DIR / "images" / "chilli" / "cropped" / "cropped" / "healthy" / "IMG_2978_1.jpg"),
        "label": "Chilli – Healthy",
        "crop": "Chilli"
    },
]


def handle_prediction(image, soil_type="Unknown", growth_stage="Unknown", severity="Unknown", api_key=None):
    """
    Standard callback for image processing and output formatting.
    Preserved for full test-suite and backwards compatibility.
    """
    soil_type_str = str(soil_type) if soil_type else "Unknown"
    growth_stage_str = str(growth_stage) if growth_stage else "Unknown"
    severity_str = str(severity) if severity else "Unknown"
    api_key_clean = str(api_key).strip() if (api_key and str(api_key).strip()) else None

    if image is None:
        return (
            "—",
            "—",
            "—",
            "—",
            "⚠️ **Please upload a crop leaf image.**",
            "*No predictions available. Please upload an image.*",
            "Please upload a crop leaf image."
        )

    try:
        result = predict_disease(
            image_input=image,
            soil_type=soil_type_str,
            growth_stage=growth_stage_str,
            severity=severity_str,
            api_key=api_key_clean
        )
    except Exception as e:
        return (
            "⚠️ Processing Error",
            "—",
            "—",
            "—",
            f"❌ An error occurred during prediction: {type(e).__name__}",
            "N/A",
            "Unable to generate advisory."
        )

    if not result.get("success", False):
        error_msg = result.get("error", "An error occurred during prediction.")
        return (
            "⚠️ Invalid Input",
            "—",
            "—",
            "—",
            f"❌ {error_msg}",
            "N/A",
            "Unable to generate advisory."
        )

    validation_text = result.get("validation_status", "Valid crop-leaf image")

    if not result.get("is_valid", True):
        validation_display = f"⚠️ {validation_text}" if not validation_text.startswith("⚠️") else validation_text
        status_md = (
            "⚠️ **Invalid / uncertain input**\n\n"
            "*The uploaded image does not appear to contain a clear crop leaf. "
            "Please upload a close-up image of a Paddy, Cotton, or Chilli leaf.*"
        )
        return (
            validation_display,
            "—",
            "—",
            "—",
            status_md,
            "N/A",
            "No pesticide or disease-specific advisory is provided because the uploaded image does not appear to contain a clear crop leaf."
        )

    crop = result.get("crop", "—")
    disease = result.get("disease", "—")
    conf_pct = result.get("confidence_pct", "—")
    is_confident = result.get("is_confident", False)

    top_3_list = result.get("top_3", [])
    if top_3_list:
        top_3_text = ""
        for item in top_3_list:
            top_3_text += f"{item['rank']}. **{item['display_name']}** — `{item['percentage']}`\n\n"
    else:
        top_3_text = "N/A"

    if is_confident:
        display_crop = crop
        display_disease = disease
        status_md = "✅ **Reasonable confidence**"
        
        adv = result.get("advisory")
        if adv and adv.get("is_ai"):
            advisory_md = adv.get("ai_text", "")
        elif adv and adv.get("ai_text"):
            notice = "> *Notice: Gemini AI advisory is currently unavailable. Displaying standard reference advisory below.*\n\n" if api_key_clean else ""
            advisory_md = notice + adv.get("ai_text", "")
        else:
            advisory_md = "No advisory available."
    else:
        display_crop = "Uncertain / Low Confidence"
        display_disease = f"{disease} (Tentative)"
        status_md = (
            "⚠️ **Low confidence / uncertain input**\n\n"
            "*Input image could not be identified with sufficient confidence.*"
        )
        advisory_md = (
            "No disease-specific pesticide recommendation is provided because the model confidence is below the 60% threshold.\n\n"
            "Please upload a clear close-up image of the crop leaf."
        )

    return (
        validation_text,
        display_crop,
        display_disease,
        conf_pct,
        status_md,
        top_3_text.strip(),
        advisory_md.strip()
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Serve the modern interactive web application."""
    available_examples = []
    for sample in SAMPLE_IMAGES:
        if sample["path"].exists():
            available_examples.append({
                "id": sample["id"],
                "label": sample["label"],
                "crop": sample["crop"]
            })

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"examples": available_examples}
    )


@app.get("/sample-image/{image_id}")
def get_sample_image(image_id: int):
    """Serve verified sample images for one-click testing."""
    for sample in SAMPLE_IMAGES:
        if sample["id"] == image_id and sample["path"].exists():
            return FileResponse(str(sample["path"]))
    return JSONResponse(status_code=404, content={"error": "Sample image not found"})


@app.post("/predict")
def handle_predict(
    image: UploadFile = File(...),
    soil_type: str = Form("Unknown"),
    growth_stage: str = Form("Unknown"),
    severity: str = Form("Unknown"),
    api_key: Optional[str] = Form(None)
):
    """
    Handle crop leaf disease prediction.
    Feeds the uploaded image to utils.predict_disease() without changing any backend logic.
    """
    try:
        contents = image.file.read()
        if not contents:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Uploaded file is empty. Please select a valid crop leaf image."}
            )

        try:
            pil_image = Image.open(io.BytesIO(contents))
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Unable to read image. Please upload a valid JPG, PNG, or WEBP file."}
            )

        soil_type_str = str(soil_type).strip() if soil_type else "Unknown"
        growth_stage_str = str(growth_stage).strip() if growth_stage else "Unknown"
        severity_str = str(severity).strip() if severity else "Unknown"
        api_key_clean = str(api_key).strip() if (api_key and str(api_key).strip()) else None

        result = predict_disease(
            image_input=pil_image,
            soil_type=soil_type_str,
            growth_stage=growth_stage_str,
            severity=severity_str,
            api_key=api_key_clean
        )

        return JSONResponse(content=result)

    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"An unexpected error occurred during prediction: {type(exc).__name__}: {str(exc)}"
            }
        )


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  AI CROP ADVISOR - MODERN WEB APPLICATION")
    print("  Server running at: http://127.0.0.1:7860")
    print("=" * 65 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=7860)
