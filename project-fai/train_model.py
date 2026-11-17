"""
================================================================================
AI Crop Advisor – Model Training Pipeline (MobileNetV2 Transfer Learning)
================================================================================
FAI Project: Simple AI-Based Crop Disease Detection System
Supported Crops: Paddy (10 classes), Cotton (4 classes), Chilli (5 classes) -> Total 19 classes

Key FAI Concepts Implemented:
1. Supervised Learning & Image Classification
2. Transfer Learning using Pretrained CNN (MobileNetV2)
3. Data Augmentation & Normalization
4. Stratified Train/Validation Split (80% / 20%)
5. Softmax Activation & Multi-Class Cross-Entropy Loss
6. Evaluation: Loss, Accuracy, Confusion Matrix, Classification Report
================================================================================
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility across runs
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Hyperparameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 0.001


def load_and_verify_dataset(base_dir):
    """
    Step 1: Scan dataset folders for Paddy, Cotton, and Chilli, verify images,
    and build a dataset of (image_path, class_label).
    Supported classes: 10 Paddy + 4 Cotton + 5 Chilli = 19 classes total.
    """
    print("\n" + "=" * 60)
    print("STEP 1: LOADING & VERIFYING DATASET (PADDY + COTTON + CHILLI)")
    print("=" * 60)

    base_path = Path(base_dir)
    paddy_dir = base_path / "images" / "paddy" / "train_images"
    cotton_dir = base_path / "images" / "cotton" / "cotton"
    chilli_dir = base_path / "images" / "chilli" / "cropped" / "cropped"

    # Verify directory existence
    if not paddy_dir.exists():
        raise FileNotFoundError(f"Paddy dataset directory not found at: {paddy_dir}")
    if not cotton_dir.exists():
        raise FileNotFoundError(f"Cotton dataset directory not found at: {cotton_dir}")
    if not chilli_dir.exists():
        raise FileNotFoundError(f"Chilli dataset directory not found at: {chilli_dir}")

    image_paths = []
    labels = []
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".mpo"]

    # 1. Load Paddy classes (10 classes)
    print("Scanning Paddy classes...")
    for class_folder in sorted(paddy_dir.iterdir()):
        if class_folder.is_dir():
            class_name = f"Paddy___{class_folder.name}"
            count = 0
            for img_file in class_folder.glob("*.*"):
                if img_file.suffix.lower() in valid_extensions:
                    image_paths.append(str(img_file))
                    labels.append(class_name)
                    count += 1
            print(f"  - {class_name:<35}: {count} images")

    # 2. Load Cotton classes (4 classes)
    print("\nScanning Cotton classes...")
    for class_folder in sorted(cotton_dir.iterdir()):
        if class_folder.is_dir():
            class_name = f"Cotton___{class_folder.name}"
            count = 0
            for img_file in class_folder.glob("*.*"):
                if img_file.suffix.lower() in valid_extensions:
                    image_paths.append(str(img_file))
                    labels.append(class_name)
                    count += 1
            print(f"  - {class_name:<35}: {count} images")

    # 3. Load Chilli classes (5 classes)
    chilli_class_map = {
        "cercospora": "Chilli___cercospora",
        "healthy": "Chilli___healthy",
        "murda complex(mites,trips)": "Chilli___murda_complex",
        "nutritional": "Chilli___nutritional",
        "powdery mildew": "Chilli___powdery_mildew"
    }

    print("\nScanning Chilli classes...")
    for class_folder in sorted(chilli_dir.iterdir()):
        if class_folder.is_dir():
            raw_folder_name = class_folder.name
            if raw_folder_name in chilli_class_map:
                class_name = chilli_class_map[raw_folder_name]
                count = 0
                for img_file in class_folder.glob("*.*"):
                    if img_file.suffix.lower() in valid_extensions:
                        image_paths.append(str(img_file))
                        labels.append(class_name)
                        count += 1
                print(f"  - {class_name:<35}: {count} images (from '{raw_folder_name}')")
            else:
                print(f"  [Skipping non-class folder: {raw_folder_name}]")

    total_images = len(image_paths)
    print(f"\nTotal raw images found: {total_images}")
    if total_images == 0:
        raise ValueError("No valid images found in the dataset directories!")

    # Unique sorted class names (19 classes)
    unique_classes = sorted(list(set(labels)))
    class_to_idx = {cls_name: i for i, cls_name in enumerate(unique_classes)}

    label_indices = [class_to_idx[lbl] for lbl in labels]

    print(f"Total Unique Classes: {len(unique_classes)}")
    return image_paths, label_indices, unique_classes


def create_tf_dataset(image_paths, labels, is_training=True):
    """
    Step 2: Create a high-performance tf.data pipeline.
    Reads images from disk, decodes, resizes to 224x224, and batches.
    """
    def parse_image_and_label(path, label):
        # Read raw byte string from disk
        image_bytes = tf.io.read_file(path)
        # Decode image (3 channels RGB)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        # Resize to fixed input size (224, 224)
        image = tf.image.resize(image, IMG_SIZE)
        image.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
        return image, label

    path_ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = path_ds.map(parse_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)

    if is_training:
        ds = ds.shuffle(buffer_size=1024, seed=SEED)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def build_crop_disease_model(num_classes):
    """
    Step 3: Build Transfer Learning CNN Model with MobileNetV2.
    Architecture:
    Input (224x224x3) -> Augmentation -> Preprocessing -> MobileNetV2 Base (Frozen)
    -> GlobalAveragePooling2D -> Dense(128, ReLU) -> Dropout(0.2) -> Dense(num_classes, Softmax)
    """
    print("\n" + "=" * 60)
    print("STEP 3: BUILDING MODEL ARCHITECTURE")
    print("=" * 60)

    # 1. Input Layer
    inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="input_image")

    # 2. Data Augmentation Layers (Applied during training only)
    x = layers.RandomFlip("horizontal", seed=SEED)(inputs)
    x = layers.RandomRotation(0.05, seed=SEED)(x)
    x = layers.RandomZoom(0.1, seed=SEED)(x)

    # 3. MobileNetV2 Preprocessing (scales pixel values between -1 and 1)
    x = preprocess_input(x)

    # 4. Pretrained MobileNetV2 Base Model (Feature Extractor)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze pretrained weights

    x = base_model(x, training=False)

    # 5. Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.2, seed=SEED, name="dropout_regularization")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_prediction")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="AI_Crop_Advisor_MobileNetV2")

    # 6. Compile Model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


def plot_and_save_history(history, save_path):
    """
    Plot and save training & validation accuracy and loss curves.
    """
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs_range = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    # Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy", marker="o")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy", marker="s")
    plt.title("Training & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")

    # Loss subplot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss", marker="o", color="red")
    plt.plot(epochs_range, val_loss, label="Validation Loss", marker="s", color="orange")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved training history plot: {save_path}")


def plot_and_save_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Compute, plot, and save the confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title("Confusion Matrix – AI Crop Advisor", fontsize=14, pad=15)
    plt.xlabel("Predicted Disease Class", fontsize=12)
    plt.ylabel("Actual Disease Class", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix: {save_path}")


def main():
    print("=" * 60)
    print("AI CROP ADVISOR: MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Define paths
    current_dir = Path(__file__).resolve().parent
    workspace_root = current_dir.parent
    data_dir = workspace_root / "data"
    models_dir = current_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = models_dir / "crop_disease_model.keras"
    class_names_path = models_dir / "class_names.json"
    history_plot_path = models_dir / "training_history.png"
    cm_plot_path = models_dir / "confusion_matrix.png"

    # Step 1: Load and verify dataset
    image_paths, labels, class_names = load_and_verify_dataset(data_dir)

    # Save class names JSON
    with open(class_names_path, "w") as f:
        json.dump(class_names, f, indent=4)
    print(f"\nSaved class names to: {class_names_path}")

    # Step 2: Stratified Train / Validation Split (80% / 20%)
    print("\n" + "=" * 60)
    print("STEP 2: STRATIFIED TRAIN / VALIDATION SPLIT (80% / 20%)")
    print("=" * 60)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels
    )

    print(f"Training images count   : {len(train_paths)}")
    print(f"Validation images count : {len(val_paths)}")
    print(f"Number of classes       : {len(class_names)}")

    # Compute balanced class weights to handle slight class imbalance
    cls_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weight_dict = {i: float(w) for i, w in enumerate(cls_weights)}
    print("\nClass Weights (to handle class imbalance):")
    for i, cls_name in enumerate(class_names):
        print(f"  - {cls_name:<35}: weight = {class_weight_dict[i]:.3f}")

    # Create tf.data datasets
    train_ds = create_tf_dataset(train_paths, train_labels, is_training=True)
    val_ds = create_tf_dataset(val_paths, val_labels, is_training=False)

    # Step 3: Build Model
    model = build_crop_disease_model(num_classes=len(class_names))

    # Step 4: Setup Callbacks
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = callbacks.ModelCheckpoint(
        filepath=str(model_save_path),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    # Step 5: Train Model
    print("\n" + "=" * 60)
    print(f"STEP 4: TRAINING MODEL ({EPOCHS} EPOCHS)")
    print("=" * 60)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, checkpoint]
    )

    # Ensure best model is saved
    model.save(str(model_save_path))
    print(f"\nTrained model successfully saved to: {model_save_path}")

    # Plot training history
    plot_and_save_history(history, history_plot_path)

    # Step 6: Comprehensive Model Evaluation on Validation Set
    print("\n" + "=" * 60)
    print("STEP 5: EVALUATION ON VALIDATION DATASET")
    print("=" * 60)

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    train_loss = history.history["loss"][-1]
    train_acc = history.history["accuracy"][-1]

    print(f"Final Training Loss        : {train_loss:.4f}")
    print(f"Final Training Accuracy    : {train_acc * 100:.2f}%")
    print(f"Final Validation Loss      : {val_loss:.4f}")
    print(f"Final Validation Accuracy  : {val_acc * 100:.2f}%")

    # Generate predictions for confusion matrix & classification report
    print("\nGenerating validation predictions for detailed evaluation...")
    y_true_all = []
    y_pred_all = []

    for images, batch_labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_pred_all.extend(np.argmax(preds, axis=1))
        y_true_all.extend(batch_labels.numpy())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Classification report
    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(y_true_all, y_pred_all, target_names=class_names, digits=4))

    # Confusion matrix
    plot_and_save_confusion_matrix(y_true_all, y_pred_all, class_names, cm_plot_path)

    # Step 7: Sample Predictions Test
    print("\n" + "=" * 60)
    print("STEP 6: SAMPLE PREDICTIONS TEST (VALIDATION SAMPLES)")
    print("=" * 60)

    # Pick 5 diverse validation samples
    sample_indices = random.sample(range(len(val_paths)), min(5, len(val_paths)))
    for idx in sample_indices:
        sample_path = val_paths[idx]
        actual_class = class_names[val_labels[idx]]

        # Preprocess single image
        raw_img = Image.open(sample_path).convert("RGB").resize(IMG_SIZE)
        img_arr = np.array(raw_img, dtype=np.float32)
        img_batch = np.expand_dims(img_arr, axis=0)

        # Model prediction
        pred_probs = model.predict(img_batch, verbose=0)[0]
        pred_idx = np.argmax(pred_probs)
        pred_class = class_names[pred_idx]
        confidence = pred_probs[pred_idx] * 100.0

        print(f"Image      : {Path(sample_path).name}")
        print(f"Actual     : {actual_class}")
        print(f"Predicted  : {pred_class}")
        print(f"Confidence : {confidence:.2f}%")
        print("-" * 40)

    # Summary Output
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Classes: {len(class_names)}")
    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"\nSaved model:\n  {model_save_path}")
    print(f"\nSaved class names:\n  {class_names_path}")
    print(f"\nSaved confusion matrix:\n  {cm_plot_path}")
    print(f"\nSaved training history:\n  {history_plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
