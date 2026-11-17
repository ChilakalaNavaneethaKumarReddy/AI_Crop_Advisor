"""
================================================================================
AI Crop Advisor – Verification Test Suite (19 Classes: Paddy, Cotton, Chilli)
================================================================================
Test Cases:
Case A: Cotton Bacterial Blight (bact1.jpg)
Case B: Paddy Blast (100074.jpg)
Case C: Chilli disease image (Cercospora - IMG_3420.JPG)
Case D: Chilli healthy image (IMG_2978_1.jpg)
Case E: Human / Person Photograph (safely rejected before inference)
Case F: Dark / Unclear Image (safely rejected by image quality check)
Case G: Random / Non-Crop Image (safely rejected by vegetation check)
Case H: No Image Uploaded (graceful prompt)
================================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
import json
import numpy as np
from PIL import Image
from utils import predict_disease, load_model_and_classes, is_likely_leaf_image
from app import handle_prediction

def run_tests():
    workspace_root = Path(__file__).resolve().parent.parent
    data_dir = workspace_root / "data"

    cotton_bacterial_blight = data_dir / "images" / "cotton" / "cotton" / "bacterial_blight" / "bact1.jpg"
    paddy_blast = data_dir / "images" / "paddy" / "train_images" / "blast" / "100074.jpg"
    chilli_disease = data_dir / "images" / "chilli" / "cropped" / "cropped" / "cercospora" / "IMG_3420.JPG"
    chilli_healthy = data_dir / "images" / "chilli" / "cropped" / "cropped" / "healthy" / "IMG_2978_1.jpg"

    # Human / person photograph proxy (centered human face/skin tone with background)
    human_photo_arr = np.zeros((224, 224, 3), dtype=np.uint8)
    human_photo_arr[:, :] = [90, 140, 70] # Greenish background (e.g. garden/room background)
    human_photo_arr[30:190, 45:175] = [215, 165, 135] # Central human face / skin tone
    human_photo_img = Image.fromarray(human_photo_arr)

    # Dark / unclear image (too little variance / underexposed)
    dark_unclear_arr = np.zeros((224, 224, 3), dtype=np.uint8)
    dark_unclear_img = Image.fromarray(dark_unclear_arr)

    # Non-crop / random image (e.g. solid red object)
    non_leaf_arr = np.zeros((224, 224, 3), dtype=np.uint8)
    non_leaf_arr[:, :, 0] = np.random.randint(160, 255, (224, 224)) # High red
    non_leaf_arr[:, :, 1] = np.random.randint(0, 30, (224, 224))    # Low green
    non_leaf_arr[:, :, 2] = np.random.randint(0, 30, (224, 224))    # Low blue
    non_leaf_img = Image.fromarray(non_leaf_arr)

    test_cases = [
        ("Case A: Cotton Bacterial Blight", cotton_bacterial_blight, "Cotton prediction + disease advisory"),
        ("Case B: Paddy Blast", paddy_blast, "Paddy prediction + disease advisory"),
        ("Case C: Chilli Disease Image (Cercospora)", chilli_disease, "Chilli prediction + disease advisory"),
        ("Case D: Chilli Healthy Image", chilli_healthy, "Chilli healthy prediction + advisory"),
        ("Case E: Human / Person Photograph", human_photo_img, "Validation rejection (Image may not contain a crop leaf)"),
        ("Case F: Dark / Unclear Image", dark_unclear_img, "Validation rejection (Image may not contain a crop leaf)"),
        ("Case G: Random / Non-Crop Image", non_leaf_img, "Validation rejection (Image may not contain a crop leaf)"),
        ("Case H: No Image Uploaded", None, "Prompt to upload image")
    ]

    print("=" * 75)
    print("AI CROP ADVISOR - COMPREHENSIVE VERIFICATION SUITE (19 CLASSES)")
    print("=" * 75)

    # Load 19 classes list for validation
    _, class_names = load_model_and_classes()
    print(f"Total Registered Classes in Model: {len(class_names)}")
    assert len(class_names) == 19, f"Expected 19 classes, got {len(class_names)}"

    passed_count = 0
    failed_count = 0

    for case_id, img_input, expected in test_cases:
        print(f"\n[*] Running {case_id}")
        print(f"   Expected: {expected}")

        # 1. Test direct utils.predict_disease
        res = predict_disease(img_input)

        # 2. Test Gradio handle_prediction
        val_status, crop, disease, conf, status, top3, advisory = handle_prediction(img_input)

        print(f"   [Gradio Display Outputs]")
        print(f"   - Validation: {val_status}")
        print(f"   - Crop      : {crop}")
        print(f"   - Condition : {disease}")
        print(f"   - Confidence: {conf}")
        print(f"   - Status    : {status.splitlines()[0] if status else 'None'}")
        print(f"   - Advisory  : {advisory.splitlines()[0] if advisory else 'None'}")

        try:
            # Verification per test case
            if case_id.startswith("Case A"):
                assert res["success"] is True
                assert res["is_valid"] is True
                assert res["raw_class"] == "Cotton___bacterial_blight"
                assert 0.0 <= res["confidence_val"] <= 1.0
                assert res["is_confident"] is True
                assert crop == "Cotton"
                assert "Reasonable confidence" in status
                assert "Pesticide / Chemical Management" in advisory
                assert "Safety & Regulatory Note" in advisory
                print("   [PASS]: Case A Cotton successfully verified with Chemical Management.")

            elif case_id.startswith("Case B"):
                assert res["success"] is True
                assert res["is_valid"] is True
                assert res["raw_class"] == "Paddy___blast"
                assert 0.0 <= res["confidence_val"] <= 1.0
                assert res["is_confident"] is True
                assert crop == "Paddy"
                assert "Reasonable confidence" in status
                assert "Pesticide / Chemical Management" in advisory
                assert "Safety & Regulatory Note" in advisory
                print("   [PASS]: Case B Paddy successfully verified with Chemical Management.")

            elif case_id.startswith("Case C"):
                assert res["success"] is True
                assert res["is_valid"] is True
                assert res["raw_class"].startswith("Chilli___")
                assert 0.0 <= res["confidence_val"] <= 1.0
                assert crop == "Chilli"
                assert "Reasonable confidence" in status
                assert "Pesticide / Chemical Management" in advisory
                assert "Safety & Regulatory Note" in advisory
                print("   [PASS]: Case C Chilli disease successfully verified with Chemical Management.")

            elif case_id.startswith("Case D"):
                assert res["success"] is True
                assert res["is_valid"] is True
                assert res["raw_class"].startswith("Chilli___")
                assert 0.0 <= res["confidence_val"] <= 1.0
                assert crop == "Chilli"
                assert "Reasonable confidence" in status
                assert "Healthy Crop Management" in advisory
                assert "Safety & Regulatory Note" in advisory
                print("   [PASS]: Case D Chilli healthy successfully verified without chemical spray.")

            elif case_id.startswith("Case E"):
                assert res["is_valid"] is False
                assert "Image may not contain a crop leaf" in val_status
                assert crop == "—"
                assert disease == "—"
                assert conf == "—"
                assert "Invalid / uncertain input" in status
                assert "No pesticide or disease-specific advisory is provided" in advisory
                print("   [PASS]: Case E Human photograph successfully blocked.")

            elif case_id.startswith("Case F"):
                assert res["is_valid"] is False
                assert "Image may not contain a crop leaf" in val_status or "Invalid image" in val_status
                assert crop == "—"
                assert disease == "—"
                assert conf == "—"
                assert "No pesticide or disease-specific advisory is provided" in advisory
                print("   [PASS]: Case F Dark/unclear image successfully blocked.")

            elif case_id.startswith("Case G"):
                assert res["is_valid"] is False
                assert "Image may not contain a crop leaf" in val_status
                assert crop == "—"
                assert disease == "—"
                assert conf == "—"
                assert "No pesticide or disease-specific advisory is provided" in advisory
                print("   [PASS]: Case G Non-crop image successfully blocked.")

            elif case_id.startswith("Case H"):
                assert res["success"] is False
                assert "upload a crop leaf image" in res["error"].lower()
                assert "upload a crop leaf image" in status.lower()
                print("   [PASS]: Case H No image uploaded handled gracefully.")

            passed_count += 1

        except Exception as e:
            print(f"   [FAIL]: {case_id} failed: {e}")
            failed_count += 1

    # Additional Gemini Integration & Reliability Test Cases
    print("\n" + "=" * 75)
    print("RUNNING GEMINI INTEGRATION & RELIABILITY TEST CASES (CASES I - M)")
    print("=" * 75)

    # Case I: Valid disease image + no Gemini API key
    print("\n[*] Running Case I: Valid disease image + No Gemini API key")
    val_status, crop, disease, conf, status, top3, advisory = handle_prediction(cotton_bacterial_blight, "Red Soil", "Vegetative", "Moderate (10-30%)", None)
    try:
        assert crop == "Cotton"
        assert disease == "Bacterial Blight"
        assert "Reasonable confidence" in status
        assert "Pesticide / Chemical Management" in advisory
        assert "Safety & Regulatory Note" in advisory
        print("   [PASS]: Case I Fallback advisory returned safely with Chemical Management when no API key provided.")
        passed_count += 1
    except Exception as e:
        print(f"   [FAIL]: Case I failed: {e}")
        failed_count += 1

    # Case J: Valid disease image + Invalid Gemini API key
    print("\n[*] Running Case J: Valid disease image + Invalid Gemini API key")
    val_status, crop, disease, conf, status, top3, advisory = handle_prediction(cotton_bacterial_blight, "Black Cotton Soil", "Flowering", "High (>30%)", "INVALID_KEY_12345")
    try:
        assert crop == "Cotton"
        assert disease == "Bacterial Blight"
        assert "Reasonable confidence" in status
        assert "Pesticide / Chemical Management" in advisory
        assert "Safety & Regulatory Note" in advisory
        print("   [PASS]: Case J Invalid API key handled gracefully with fallback Chemical Management advisory.")
        passed_count += 1
    except Exception as e:
        print(f"   [FAIL]: Case J failed: {e}")
        failed_count += 1

    # Case K: Valid disease image + simulated/mocked Gemini API
    print("\n[*] Running Case K: Valid disease image + Gemini advisory verification")
    from unittest.mock import patch, MagicMock
    mock_response = MagicMock()
    mock_response.text = (
        "### 🌱 Crop & Disease\n- **Crop**: Cotton\n- **Detected Disease**: Bacterial Blight\n\n"
        "### 🦠 Disease Overview\nBacterial blight details...\n\n"
        "### 🌾 Immediate Management\nSanitation details...\n\n"
        "### 🧪 Pesticide / Chemical Management\n- **Chemical Group / Active Ingredient**: Copper Oxychloride\n\n"
        "### 🌿 Biological / Organic Management\nPseudomonas...\n\n"
        "### 🌱 Soil-Specific Management\nBlack cotton soil drainage...\n\n"
        "### 🧑🌾 Growth-Stage Advice\nVegetative stage...\n\n"
        "### ⚠️ Severity-Based Action\nModerate severity...\n\n"
        "### 🛡️ Safety & Regulatory Note\nUse registered products..."
    )
    with patch("google.generativeai.GenerativeModel.generate_content", return_value=mock_response):
        with patch("google.generativeai.configure"):
            val_status, crop, disease, conf, status, top3, advisory = handle_prediction(
                cotton_bacterial_blight, "Black Cotton Soil", "Vegetative", "Moderate (10-30%)", "mock_valid_api_key"
            )
            try:
                assert crop == "Cotton"
                assert disease == "Bacterial Blight"
                assert "Pesticide / Chemical Management" in advisory
                assert "Safety & Regulatory Note" in advisory
                print("   [PASS]: Case K Gemini AI advisory successfully generated with all required sections.")
                passed_count += 1
            except Exception as e:
                print(f"   [FAIL]: Case K failed: {e}")
                failed_count += 1

    # Case L: Low-confidence image + Gemini API key (Gemini must NOT be called)
    print("\n[*] Running Case L: Low-confidence / uncertain image + Gemini API key")
    with patch("google.generativeai.GenerativeModel.generate_content") as mock_gemini:
        low_conf_arr = np.zeros((224, 224, 3), dtype=np.uint8)
        low_conf_arr[:, :] = [100, 140, 70] # Greenish
        low_conf_img = Image.fromarray(low_conf_arr)
        with patch("utils.predict_disease", return_value={
            "success": True, "is_valid": True, "validation_status": "Valid crop-leaf image",
            "raw_class": "Paddy___blast", "crop": "Paddy", "disease": "Blast",
            "confidence_val": 0.45, "confidence_pct": "45.00%", "is_confident": False,
            "status_text": "Low confidence / uncertain input",
            "confidence_status": "⚠️ Low confidence / uncertain input",
            "confidence_note": "Below threshold", "top_3": [], "advisory": None
        }):
            val_status, crop, disease, conf, status, top3, advisory = handle_prediction(
                low_conf_img, "Alluvial Soil", "Seedling", "Low (1-10%)", "mock_valid_key"
            )
            try:
                assert crop == "Uncertain / Low Confidence"
                assert "Tentative" in disease
                assert "No disease-specific pesticide recommendation is provided" in advisory
                mock_gemini.assert_not_called()
                print("   [PASS]: Case L Gemini was NOT called for low-confidence prediction.")
                passed_count += 1
            except Exception as e:
                print(f"   [FAIL]: Case L failed: {e}")
                failed_count += 1

    # Case M: Healthy crop image + Gemini API key (Gemini must NOT be called)
    print("\n[*] Running Case M: Healthy crop image + Gemini API key")
    with patch("google.generativeai.GenerativeModel.generate_content") as mock_gemini:
        val_status, crop, disease, conf, status, top3, advisory = handle_prediction(
            chilli_healthy, "Red Soil", "Vegetative", "Low (1-10%)", "mock_valid_key"
        )
        try:
            assert crop == "Chilli"
            assert "Healthy" in disease
            assert "Healthy Crop Management" in advisory
            assert "No chemical pesticides are recommended" in advisory
            mock_gemini.assert_not_called()
            print("   [PASS]: Case M Gemini was NOT called for healthy crop leaf.")
            passed_count += 1
        except Exception as e:
            print(f"   [FAIL]: Case M failed: {e}")
            failed_count += 1

    print("\n" + "=" * 75)
    print(f"FINAL TEST RESULTS: {passed_count} Passed | {failed_count} Failed")
    print("=" * 75)
    assert failed_count == 0, f"{failed_count} tests failed!"

if __name__ == "__main__":
    run_tests()
