# 🌱 AI Crop Advisor – Simple AI-Based Crop Disease Detection System

**Course Project:** Fundamentals of Artificial Intelligence (FAI)  
**Frameworks Used:** TensorFlow / Keras, Gradio, Scikit-Learn, Pillow, NumPy, Matplotlib, Seaborn  

---

## 1. Project Title & Overview
**AI Crop Advisor** is an end-to-end Computer Vision and Machine Learning application designed to assist farmers and agricultural students in identifying crop diseases from leaf photographs.

Using a Convolutional Neural Network (**MobileNetV2**) pretrained on ImageNet and fine-tuned for crop pathology, the system classifies leaf images into **19 disease and healthy conditions across Paddy (Rice), Cotton, and Chilli**, outputs prediction confidence, provides the top-3 most likely diagnoses, and delivers actionable agronomic guidance through an interactive **Gradio web interface**.

---

## 2. Problem Statement
Crop diseases cause severe yield losses and economic hardship for farmers across staple and commercial crops like rice, cotton, and chilli. Manual visual inspection by agricultural experts is often slow, expensive, and inaccessible in remote farming regions. An automated, accessible AI screening tool helps provide rapid initial assessments, flags low-confidence or non-leaf images safely, and provides timely educational crop care guidance.

---

## 3. Project Objectives
1. Build a simple, lightweight, and fast image classification pipeline for crop disease diagnosis.
2. Implement **Transfer Learning** using MobileNetV2 to achieve stable feature extraction without training deep models from scratch.
3. Support **19 disease and healthy classes** across **Paddy**, **Cotton**, and **Chilli**.
4. Implement input validation heuristics to reject non-leaf images and a **60.0% confidence gate** to withhold disease advice on ambiguous inputs.
5. Deliver a clean, interactive **Gradio UI** (as covered in the FAI syllabus) for easy image upload and real-time inference.
6. Provide structured, educational crop advisory messages without making unrealistic medical or chemical dosage claims.
7. Create an easily explainable AI architecture suitable for a 3-member student team college presentation and viva.

---

## 4. Dataset & Supported Crops

The system is trained on a verified dataset of **12,648 high-resolution leaf images** across 19 categories:

| Crop | Classes Count | Total Images | Supported Status |
| :--- | :--- | :--- | :--- |
| **Paddy (Rice)** | 10 Classes | 10,407 Images | ✅ Fully Supported |
| **Cotton** | 4 Classes | 1,709 Images | ✅ Fully Supported |
| **Chilli** | 5 Classes | 532 Images | ✅ Fully Supported |
| **TOTAL** | **19 Classes** | **12,648 Images** | ✅ Fully Supported |

### **All 19 Supported Classes & Image Distribution:**

```
--- PADDY (RICE) ---
1.  Paddy___bacterial_leaf_blight     (479 images)
2.  Paddy___bacterial_leaf_streak     (380 images)
3.  Paddy___bacterial_panicle_blight  (337 images)
4.  Paddy___blast                     (1,738 images)
5.  Paddy___brown_spot                (965 images)
6.  Paddy___dead_heart                (1,442 images)
7.  Paddy___downy_mildew              (620 images)
8.  Paddy___hispa                     (1,594 images)
9.  Paddy___normal                    (1,764 images)
10. Paddy___tungro                    (1,088 images)

--- COTTON ---
11. Cotton___bacterial_blight         (448 images)
12. Cotton___curl_virus               (417 images)
13. Cotton___fussarium_wilt           (419 images)
14. Cotton___healthy                  (425 images)

--- CHILLI ---
15. Chilli___cercospora               (152 images)
16. Chilli___healthy                  (69 images)
17. Chilli___murda_complex            (107 images)
18. Chilli___nutritional              (102 images)
19. Chilli___powdery_mildew           (102 images)
```

---

## 5. Model Architecture & Pipeline

### **End-to-End Workflow:**
```
User uploads leaf image
         ↓
Lightweight Image Validation (Variance & Green-ratio heuristic)
         ↓
Resize to 224 × 224 × 3
         ↓
MobileNetV2 Preprocessing Layer (Scales pixel values to [-1, 1])
         ↓
Pretrained MobileNetV2 Base (Feature Extractor - Frozen ImageNet weights)
         ↓
GlobalAveragePooling2D (Reduces 7×7×1280 feature maps to 1280-dim vector)
         ↓
Dense Layer (128 units, ReLU activation)
         ↓
Dropout Layer (Rate = 0.2 to prevent overfitting)
         ↓
Final Dense Layer (19 units, Softmax activation)
         ↓
Softmax Probabilities [p₁, p₂, ..., p₁₉]
         ↓
Argmax → Predicted Disease Class & Confidence %
         ↓
60.0% Confidence Gate
         ↓
Top-3 Predictions & Educational Advisory Display
```

---

## 6. Training & Evaluation Results

- **Data Split:** 80% Training (10,118 images) / 20% Validation (2,530 images) using stratified sampling (`random_state = 42`).
- **Class Imbalance Handling:** Balanced class weights computed via Scikit-Learn to ensure smaller classes (such as Chilli) receive proportional gradient influence.
- **Optimizer:** Adam (`learning_rate = 0.001`)
- **Loss Function:** Sparse Categorical Cross-Entropy
- **Epochs:** Up to 8 with `EarlyStopping` (patience = 2, best weights restored)

### **Key Metrics Summary:**
- **Total Images:** 12,648
- **Training Images:** 10,118
- **Validation Images:** 2,530
- **Final Training Accuracy:** **66.58%**
- **Final Validation Accuracy:** **60.55%**
- **Macro F1-Score:** **64.18%**
- **Weighted F1-Score:** **60.99%**

### **Chilli Class Performance:**
- `Chilli___powdery_mildew`: **89.47% F1-score** (Precision: 94.4%, Recall: 85.0%)
- `Chilli___cercospora`: **79.37% F1-score** (Precision: 75.8%, Recall: 83.3%)
- `Chilli___nutritional`: **64.86% F1-score** (Precision: 70.6%, Recall: 60.0%)
- `Chilli___healthy`: **44.44% F1-score** (Precision: 30.0%, Recall: 85.7%)
- `Chilli___murda_complex`: **32.00% F1-score** (Precision: 100.0%, Recall: 19.1%)

### **Cotton Class Performance:**
- `Cotton___curl_virus`: **99.39% F1-score**
- `Cotton___healthy`: **93.17% F1-score**
- `Cotton___bacterial_blight`: **92.78% F1-score**
- `Cotton___fussarium_wilt`: **89.87% F1-score**

All generated artifacts are saved in `project-fai/models/`:
- `models/crop_disease_model.keras`
- `models/class_names.json`
- `models/confusion_matrix.png`
- `models/training_history.png`

---

## 7. Input Validation & Confidence Gate Mechanisms

### A. Lightweight Input Validation (Before Neural Network)
Before feeding an image to the neural network, `is_likely_leaf_image()` checks basic visual attributes:
1. **Variance & Brightness Check:** Rejects pitch-black, washed-out, or flat/blank images.
2. **Vegetation Ratio Check:** Converts RGB to HSV space and verifies that a minimum fraction of pixels fall in the broad green/yellow-green hue range ($0.08 < \text{Hue} < 0.60$).
3. **Safety Benefit:** Catches non-leaf images (human faces, plain documents, random objects) before the closed-set neural network can assign a false high-confidence disease label.

### B. 60.0% Confidence Gate (Application-Level Safety)
After the neural network produces softmax probabilities:
* **Confidence $\ge$ 60%:** Displays predicted crop, disease/condition, confidence percentage, Top 3 predictions, and actionable disease management advisory.
* **Confidence $<$ 60% (Uncertain / Low Confidence):** 
  - Displays crop status as: `⚠️ Low confidence / uncertain input`
  - Shows top 3 softmax probabilities
  - **Withholds disease-specific chemical/crop advice** and prompts the user for a clearer close-up photograph.

---

## 8. FAI Syllabus Concepts Mapping (22 Concepts)

| # | Concept | Explanation in the Context of AI Crop Advisor |
| :--- | :--- | :--- |
| 1 | **Artificial Intelligence** | Creating systems capable of performing tasks requiring visual pattern recognition. |
| 2 | **Supervised Learning** | Training the model on labeled pairs of leaf images and corresponding disease ground-truth classes. |
| 3 | **Dataset and Labels** | 12,648 images organized into 19 categorical targets across Paddy, Cotton, and Chilli. |
| 4 | **Image Preprocessing** | Resizing raw images to $224 \times 224$ and scaling RGB values to $[-1, 1]$ for MobileNetV2. |
| 5 | **CNN (Convolutional Neural Network)** | Deep architecture using spatial convolution kernels to detect edges, textures, and lesion patterns. |
| 6 | **Transfer Learning** | Leveraging features learned on 1.4M ImageNet images to train a custom classifier on a smaller agricultural dataset. |
| 7 | **MobileNetV2** | A lightweight CNN architecture using depthwise separable convolutions for efficient CPU/edge inference. |
| 8 | **Activation Functions** | Mathematical functions introducing non-linearity into neurons so the network can learn complex boundaries. |
| 9 | **ReLU (Rectified Linear Unit)** | $f(x) = \max(0, x)$, used in the Dense(128) layer to prevent vanishing gradients and speed up convergence. |
| 10 | **Softmax** | Converts raw logits into 19 normalized probabilities summing to 1.0 ($100\%$). |
| 11 | **Loss Function** | Sparse Categorical Cross-Entropy, quantifying the divergence between predicted probability distribution and true class index. |
| 12 | **Adam Optimizer** | Adaptive Moment Estimation optimizer combining momentum and RMSprop for stable gradient descent. |
| 13 | **Dropout** | Regularization technique randomly deactivating 20% of dense features during training to prevent overfitting. |
| 14 | **Training & Validation** | 80/20 stratified split to train parameters on training data and measure unbiased generalization on unseen validation data. |
| 15 | **Accuracy** | Fraction of total predictions correctly classified: $\frac{TP + TN}{\text{Total}}$. |
| 16 | **Precision** | Fraction of predicted disease instances that were true: $\frac{TP}{TP + FP}$. |
| 17 | **Recall** | Fraction of actual disease instances successfully detected: $\frac{TP}{TP + FN}$. |
| 18 | **F1-Score** | Harmonic mean of Precision and Recall: $2 \times \frac{P \times R}{P + R}$. |
| 19 | **Confusion Matrix** | $19 \times 19$ matrix visualizing correct classifications along the diagonal and inter-class confusion. |
| 20 | **Confidence Threshold** | The 60.0% cutoff deciding whether to issue advisory actions or flag uncertainty. |
| 21 | **Human-AI Interaction** | Designing transparent AI feedback (confidence percentages, Top-3 alternatives, uncertainty warnings). |
| 22 | **Gradio** | Python web framework providing an interactive graphical interface for user leaf upload and real-time diagnostic output. |

---

## 9. Project Directory Structure

```
AI_Crop_Advisor/
│
├── data/
│   ├── images/
│   │   ├── paddy/              # 10 classes (10,407 images)
│   │   ├── cotton/             # 4 classes (1,709 images)
│   │   └── chilli/             # 5 classes (532 images)
│   ├── crop_data.csv           # Regional agronomic reference
│   └── train (1).csv
│
└── project-fai/
    ├── app.py                  # Gradio Web Application
    ├── utils.py                # Preprocessing, inference, validation & advisory logic
    ├── train_model.py          # Complete 19-class training & evaluation script
    ├── inspect_dataset.py      # Dataset scanning & validation script
    ├── test_prediction.py      # Comprehensive 7-case test suite (Cases A to G)
    ├── requirements.txt        # Minimal Python dependencies
    ├── README.md               # Complete project documentation & FAI guide
    │
    └── models/
        ├── crop_disease_model.keras  # Saved 19-class trained model
        ├── class_names.json          # 19 class labels in exact output index order
        ├── confusion_matrix.png      # Scikit-learn confusion matrix plot
        └── training_history.png      # Accuracy and loss curves plot
```

---

## 10. How to Run

### **Step 1: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 2: Inspect Dataset**
```bash
python inspect_dataset.py
```

### **Step 3: Run Verification Test Suite**
```bash
python test_prediction.py
```

### **Step 4: Launch Gradio Web Application**
```bash
python app.py
```
Open your web browser at: **`http://127.0.0.1:7860`**

---

## 11. 3-Member Student Team Explanation Guide for College Viva

### 👨‍🎓 **Student 1: Dataset, Imbalance & Preprocessing Pipeline**
* **Key Points:**
  1. **Dataset Overview:** 12,648 images across 19 classes (10 Paddy, 4 Cotton, 5 Chilli).
  2. **Class Imbalance:** Paddy has ~10.4k images while Chilli has 532 images. We solved this without synthetic oversampling by computing **balanced class weights** in `train_model.py`.
  3. **Folder Structure & Label Normalization:** Raw nested folders are normalized to unique identifiers like `Chilli___powdery_mildew` to avoid naming conflicts.
  4. **Data Preprocessing & Split:** $224 \times 224$ resizing, MobileNetV2 scaling to $[-1, 1]$, and an 80/20 stratified split (`seed=42`).

### 👩‍🎓 **Student 2: AI Model Architecture, Transfer Learning & Training**
* **Key Points:**
  1. **Why Transfer Learning:** Training deep networks from scratch on limited images causes severe overfitting; MobileNetV2 provides rich pretrained visual feature representations.
  2. **Architecture Components:** MobileNetV2 Base (Frozen) $\rightarrow$ `GlobalAveragePooling2D` $\rightarrow$ `Dense(128, ReLU)` $\rightarrow$ `Dropout(0.2)` $\rightarrow$ `Dense(19, Softmax)`.
  3. **Optimization:** Adam optimizer (`lr=0.001`), Sparse Categorical Cross-Entropy, and `EarlyStopping` to restore the best epoch weights.
  4. **Evaluation Metrics:** Training accuracy (66.58%), validation accuracy (60.55%), macro F1-score (64.18%), weighted F1-score (60.99%), and confusion matrix analysis.

### 👨‍🎓 **Student 3: Input Validation, Confidence Gate & Gradio UI**
* **Key Points:**
  1. **Lightweight Input Validation:** `is_likely_leaf_image()` checks variance and HSV green pixel ratios to block non-crop images (human faces, blank screens) before inference.
  2. **60.0% Confidence Gate:** Prevents misleading disease advice on ambiguous leaf photos by displaying "Uncertain / Low Confidence" and withholding chemical/management practices.
  3. **Advisory System:** Educational disease descriptions and best practices mapped dynamically to the 19 classes.
  4. **Gradio UI:** Interactive interface with quick-test examples (Cotton, Paddy, Chilli, Healthy crop) and Top-3 probability breakdown.

---

## 12. Viva Questions & Model Answers for the Lecturer

**Q1: Why did you use MobileNetV2 instead of building a CNN from scratch?**  
> *Answer:* Training a deep CNN from scratch on 12,648 images is computationally expensive and prone to overfitting on smaller classes like Chilli. MobileNetV2 has already learned robust visual filters (edges, textures, shapes) from ImageNet. Freezing the base and training only the classification head allows fast, stable convergence on standard hardware.

**Q2: How did you handle class imbalance between Paddy (10,407) and Chilli (532)?**  
> *Answer:* We applied balanced class weights during training using Scikit-Learn's `compute_class_weight`. This penalizes errors on underrepresented classes (like Chilli) more heavily during backpropagation, preventing the model from ignoring minority classes.

**Q3: What is the purpose of Softmax in the final layer?**  
> *Answer:* Softmax transforms the 19 unnormalized logit values into a probability distribution where every value is in $[0, 1]$ and all 19 probabilities sum to $1.0$ ($100\%$).

**Q4: Why is there an input validation check before the neural network?**  
> *Answer:* The neural network is a closed-set 19-class classifier. If a user uploads a human face or car photo, the model would still be forced to assign one of the 19 crop classes. The lightweight HSV vegetation check filters out non-leaf images beforehand.

**Q5: Why is the 60% confidence threshold important?**  
> *Answer:* In ethical and practical AI deployment, an AI tool should express uncertainty when probabilities are ambiguous. If top confidence is under 60%, the system flags the result as uncertain, presents the Top-3 possibilities, and withholds specific disease advisories to avoid giving misleading advice.
