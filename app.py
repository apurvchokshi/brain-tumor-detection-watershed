import json
import os

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras

IMG_SIZE = 224
MODEL_PATH = "best_model.keras"
LABEL_MAP_PATH = "label_map.json"


def anisotropic_diffusion(img, niter=3, k=15.0, lam=0.2, option=1):
    """Simple Perona-Malik anisotropic diffusion for grayscale images."""
    I = img.astype(np.float32)
    if I.max() > 1.5:
        I /= 255.0

    for _ in range(niter):
        north = np.zeros_like(I); north[1:, :] = I[:-1, :]
        south = np.zeros_like(I); south[:-1, :] = I[1:, :]
        east = np.zeros_like(I); east[:, :-1] = I[:, 1:]
        west = np.zeros_like(I); west[:, 1:] = I[:, :-1]

        dN = north - I
        dS = south - I
        dE = east - I
        dW = west - I

        if option == 1:
            cN = np.exp(-(dN / k) ** 2)
            cS = np.exp(-(dS / k) ** 2)
            cE = np.exp(-(dE / k) ** 2)
            cW = np.exp(-(dW / k) ** 2)
        else:
            cN = 1.0 / (1.0 + (dN / k) ** 2)
            cS = 1.0 / (1.0 + (dS / k) ** 2)
            cE = 1.0 / (1.0 + (dE / k) ** 2)
            cW = 1.0 / (1.0 + (dW / k) ** 2)

        I += lam * (cN * dN + cS * dS + cE * dE + cW * dW)

    I = np.clip(I, 0.0, 1.0)
    return (I * 255).astype(np.uint8)


def skull_strip(gray):
    """Simple skull stripping using Otsu thresholding and largest contour mask."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if np.sum(th == 255) < np.sum(th == 0):
        th = cv2.bitwise_not(th)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray

    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [c], -1, 255, thickness=cv2.FILLED)
    return cv2.bitwise_and(gray, mask)


def tophat_enhance(gray, kernel_size=15):
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)


def hist_eq(gray):
    return cv2.equalizeHist(gray)


def full_preprocess(bgr):
    if bgr is None:
        raise ValueError("Image read failure")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    adf = anisotropic_diffusion(gray, niter=3, k=20, lam=0.2, option=1)
    brain = skull_strip(adf)
    top = tophat_enhance(brain, kernel_size=15)
    he = hist_eq(top)
    he = cv2.resize(he, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return np.stack([he, he, he], axis=-1)


def watershed_segment(gray):
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.4 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.watershed(color, markers)

    seg = np.zeros_like(gray)
    seg[markers > 1] = 255
    seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, kernel, iterations=2)
    seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, kernel, iterations=1)
    return seg


def load_assets():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Train the notebook first, then upload best_model.keras to this Space."
        )
    if not os.path.exists(LABEL_MAP_PATH):
        raise FileNotFoundError(
            f"Missing {LABEL_MAP_PATH}. Train the notebook first, then upload label_map.json to this Space."
        )

    model = keras.models.load_model(MODEL_PATH)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    return model, idx_to_class


loaded_model, IDX2CLASS = load_assets()


def preprocess_for_model(bgr):
    rgb = full_preprocess(bgr)
    x = rgb.astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)
    return x, rgb


def predict_image(img):
    if img is None:
        raise gr.Error("Please upload a brain MRI image first.")

    np_img = np.array(img)
    if np_img.ndim == 2:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2BGR)
    else:
        np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    x, pp_rgb = preprocess_for_model(np_img)
    probs = loaded_model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = IDX2CLASS[str(pred_idx)]
    conf = float(probs[pred_idx])

    ws = watershed_segment(cv2.cvtColor(pp_rgb, cv2.COLOR_RGB2GRAY))
    contours, _ = cv2.findContours(ws, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis = pp_rgb.copy()
    cv2.drawContours(vis, contours, -1, (255, 0, 0), 2)

    topk = sorted(
        [(IDX2CLASS[str(i)], float(p)) for i, p in enumerate(probs)],
        key=lambda item: item[1],
        reverse=True,
    )
    top_text = "\n".join([f"{class_name}: {prob:.3f}" for class_name, prob in topk])

    return pred_class, f"confidence: {conf:.3f}\n\nclass probabilities:\n{top_text}", vis


demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload brain MRI"),
    outputs=[
        gr.Textbox(label="Predicted class"),
        gr.Textbox(label="Details", lines=6, max_lines=20),
        gr.Image(label="Preprocessed + watershed outline"),
    ],
    title="Brain Tumor Detection using CNN and Watershed Segmentation",
    description=(
        "Upload a brain MRI image. The app applies classical preprocessing, watershed segmentation, "
        "and a trained CNN model to predict the tumor class. This demo is for academic use only and "
        "is not a medical diagnostic tool."
    ),
)

if __name__ == "__main__":
    demo.launch()
