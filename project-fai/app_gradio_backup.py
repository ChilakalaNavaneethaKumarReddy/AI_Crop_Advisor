"""
================================================================================
AI Crop Advisor – Gradio Web Application (Phase 4)
================================================================================
A simple, interactive AI web application for crop disease classification.
Built with Gradio as required by the FAI (Fundamentals of AI) syllabus.

Usage:
    python app.py
================================================================================
"""

import os
from pathlib import Path
import gradio as gr
from utils import predict_disease

# Get paths for sample examples
CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent
DATA_DIR = WORKSPACE_ROOT / "data"

# Identify real sample images for interactive Gradio examples
EXAMPLES = []
sample_cotton_bacterial_blight = DATA_DIR / "images" / "cotton" / "cotton" / "bacterial_blight" / "bact1.jpg"
sample_paddy_blast = DATA_DIR / "images" / "paddy" / "train_images" / "blast" / "100074.jpg"
sample_chilli_cercospora = DATA_DIR / "images" / "chilli" / "cropped" / "cropped" / "cercospora" / "IMG_3420.JPG"
sample_chilli_healthy = DATA_DIR / "images" / "chilli" / "cropped" / "cropped" / "healthy" / "IMG_2978_1.jpg"

for s_path in [sample_cotton_bacterial_blight, sample_paddy_blast, sample_chilli_cercospora, sample_chilli_healthy]:
    if s_path.exists():
        EXAMPLES.append([str(s_path)])


def handle_prediction(image, soil_type="Unknown", growth_stage="Unknown", severity="Unknown", api_key=None):
    """
    Callback function invoked when the user clicks 'Predict Disease'.
    Processes image via utils.py and formats Gradio output fields.
    """
    # Safe string conversion for all optional inputs
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

    # Perform prediction
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

    # Handle errors gracefully
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

    # If the image was rejected early (e.g. not a leaf)
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

    # Extract results
    crop = result.get("crop", "—")
    disease = result.get("disease", "—")
    conf_pct = result.get("confidence_pct", "—")
    is_confident = result.get("is_confident", False)

    # Top 3 formatting
    top_3_list = result.get("top_3", [])
    if top_3_list:
        top_3_text = ""
        for item in top_3_list:
            top_3_text += f"{item['rank']}. **{item['display_name']}** — `{item['percentage']}`\n\n"
    else:
        top_3_text = "N/A"

    # Format according to confidence threshold
    if is_confident:
        display_crop = crop
        display_disease = disease
        status_md = "✅ **Reasonable confidence**"
        
        # Format advisory
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


def create_gradio_app():
    """
    Builds the Gradio user interface layout.
    """
    with gr.Blocks(title="AI Crop Advisor") as demo:
        # Header Section
        gr.Markdown(
            """
            # 🌱 AI Crop Advisor
            ### Simple AI-Based Crop Disease Detection System (FAI Project)
            Upload a **Paddy**, **Cotton**, or **Chilli** leaf image to detect potential diseases using a trained **MobileNetV2 CNN** model.
            
            *Supported crops: Paddy (10 classes), Cotton (4 classes), and Chilli (5 classes).*
            ---
            """
        )

        with gr.Row():
            # Left Column: Inputs
            with gr.Column(scale=1):
                gr.Markdown("### 📸 1. Input Leaf Image")
                input_image = gr.Image(
                    type="pil",
                    label="Upload Crop Leaf Image",
                    height=320
                )
                
                gr.Markdown("### 🌾 2. Field Conditions (Optional for AI Advisory)")
                soil_type = gr.Dropdown(choices=["Unknown", "Red Soil", "Black Cotton Soil", "Alluvial Soil", "Laterite Soil", "Sandy Soil"], value="Unknown", label="Soil Type")
                growth_stage = gr.Dropdown(choices=["Unknown", "Seedling", "Vegetative", "Flowering", "Fruiting/Boll Formation", "Harvesting"], value="Unknown", label="Growth Stage")
                severity = gr.Dropdown(choices=["Unknown", "Low (1-10%)", "Moderate (10-30%)", "High (>30%)"], value="Unknown", label="Disease Severity")
                
                with gr.Accordion("Advanced Settings", open=False):
                    api_key_input = gr.Textbox(label="Gemini API Key (Required for dynamic AI Advisory)", type="password", placeholder="Enter your Gemini API key here or set GEMINI_API_KEY env var")

                predict_btn = gr.Button("🔍 Predict Disease", variant="primary", size="lg")

                # Sample test images
                if EXAMPLES:
                    gr.Markdown("#### 💡 Quick Test Examples:")
                    gr.Examples(
                        examples=EXAMPLES,
                        inputs=input_image,
                        label="Click any example to load:"
                    )

            # Right Column: Prediction Results & Advisory
            with gr.Column(scale=1):
                gr.Markdown("### 🛡️ 1. Image Validation")
                out_validation = gr.Textbox(label="Validation Check", interactive=False)

                gr.Markdown("### 🎯 2. Prediction Result")
                
                with gr.Row():
                    out_crop = gr.Textbox(label="Crop", interactive=False)
                    out_disease = gr.Textbox(label="Condition", interactive=False)
                    out_conf = gr.Textbox(label="Confidence", interactive=False)

                gr.Markdown("#### Status:")
                out_status = gr.Markdown()

                gr.Markdown("### 📊 3. Top 3 Predictions")
                out_top3 = gr.Markdown()

                gr.Markdown("### 🌾 4. AI Advisory")
                out_advisory = gr.Markdown()

        # Connect button click
        predict_btn.click(
            fn=handle_prediction,
            inputs=[input_image, soil_type, growth_stage, severity, api_key_input],
            outputs=[
                out_validation,
                out_crop,
                out_disease,
                out_conf,
                out_status,
                out_top3,
                out_advisory
            ]
        )

        # Footer / Educational Disclaimer
        gr.Markdown(
            """
            ---
            **⚠️ Educational Disclaimer:**  
            *This system is an AI-assisted educational project developed for the Fundamentals of Artificial Intelligence (FAI) course. It provides initial screening based on leaf image patterns and should not replace certified agricultural extension guidance.*
            """
        )

    return demo


if __name__ == "__main__":
    app = create_gradio_app()
    # Launch Gradio app locally
    print("\n" + "=" * 60)
    print("STARTING AI CROP ADVISOR GRADIO WEB APPLICATION")
    print("=" * 60)
    app.launch(server_name="127.0.0.1", server_port=7860, share=False)
