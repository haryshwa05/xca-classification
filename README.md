# XCA Training — YOLOv8 Coronary Vessel Segmentation Model

This repo contains the training pipeline for the YOLOv8 instance-segmentation model (`best.pt`) used by the [XCA UI](../xca-ui) app to detect and label coronary artery vessel segments in X-ray coronary angiogram (XCA) images. It trains a model on the [ARCADE](https://arcade.grand-challenge.org/) dataset to predict standard ARCADE vessel-segment IDs from angiogram images.

> **Scope note:** This repo covers **only the YOLOv8 segmentation model (`best.pt`)**. A second, separate model (`segmentation.pt`, a U-Net++ built with `segmentation_models_pytorch`) is also used by the XCA UI app for a general vessel/lesion mask, but its training code is not part of this repo — it hasn't been located/reconstructed yet. See [Known Gaps](#known-gaps--todo).

---

## Table of Contents

- [1. What This Notebook Does](#1-what-this-notebook-does)
- [2. Prerequisites](#2-prerequisites)
- [3. Dataset](#3-dataset)
- [4. Preprocessing ("Paper-Exact" Pipeline)](#4-preprocessing-paper-exact-pipeline)
- [5. Label Conversion (COCO → YOLOv8-seg)](#5-label-conversion-coco--yolov8-seg)
- [6. Training Configuration](#6-training-configuration)
- [7. Evaluation](#7-evaluation)
- [8. Running This Notebook Yourself](#8-running-this-notebook-yourself)
- [9. Output: Getting `best.pt` Into the App](#9-output-getting-bestpt-into-the-app)
- [10. Known Gaps / TODO](#10-known-gaps--todo)

---

## 1. What This Notebook Does

`angio-xca-yolov8-training.ipynb` is a Kaggle notebook that:

1. Loads the ARCADE dataset (COCO-format polygon annotations) from Kaggle.
2. Filters it down to a specific set of 15 vessel-segment classes.
3. Applies a deterministic, paper-derived image preprocessing pipeline to every image.
4. Converts COCO polygon annotations into YOLOv8-seg label format and builds a standard YOLO dataset directory.
5. Fine-tunes a pretrained `yolov8m-seg.pt` checkpoint on that dataset.
6. Runs qualitative sanity checks (ground-truth overlay, prediction overlay) and quantitative evaluation (`model.val()`).
7. Produces `best.pt`, the checkpoint deployed by the downstream XCA UI app's backend.

---

## 2. Prerequisites

This notebook was built and run on **Kaggle's GPU notebook environment**, and references Kaggle-specific paths (`/kaggle/input`, `/kaggle/working`) directly. To run it elsewhere (Colab, a local machine, another cloud notebook), update the path variables noted in Section 3.

| Requirement | Notes |
|---|---|
| Python | 3.10+ |
| GPU | Strongly recommended — training config assumes `device=0` (a single CUDA GPU). Kaggle's free tier (P100/T4) was used for the original run. |
| Packages | `ultralytics`, `opencv-python-headless`, `numpy`, `matplotlib`, `tqdm` — installed via the notebook's first cell (`!pip -q install ultralytics opencv-python-headless`) |
| Dataset access | A Kaggle account with access to the `xca-classification-haryshwa/arcade` dataset (see Section 3) |

---

## 3. Dataset

- **Source:** ARCADE coronary angiography dataset, loaded from the Kaggle dataset `xca-classification-haryshwa/arcade`.
- **Task subset used:** `syntax` — the vessel-segment labeling task, with images and COCO-format polygon annotations under:
  ```
  arcade/syntax/train/images/, arcade/syntax/train/annotations/*.json
  arcade/syntax/val/images/,   arcade/syntax/val/annotations/*.json
  arcade/syntax/test/images/,  arcade/syntax/test/annotations/*.json
  ```
- **Path variables to edit if you run this elsewhere:**
  ```python
  RAW_ROOT = "/kaggle/input/xca-classification-haryshwa/arcade/syntax"
  ANN_ROOT = "/kaggle/input/xca-classification-haryshwa/arcade"
  DATASET_ROOT = "/kaggle/input/xca-classification-haryshwa/arcade"
  WORK_ROOT = "/kaggle/working"
  ```
- **Classes used:** Only 15 of the full ARCADE category set are kept:
  ```python
  ALLOWED_CLASSES = [6, 5, 2, 1, 3, 7, 13, 8, 16, 4, 20, 9, 15, 25, 17]
  ```
  These ARCADE category IDs are remapped to contiguous YOLO class indices `0–14`:
  ```python
  CLASS_ID_MAP = {cid: idx for idx, cid in enumerate(ALLOWED_CLASSES)}
  ```
  **This exact list, in this exact order, is what the downstream serving app's class mapping is built from.** If you retrain with a different `ALLOWED_CLASSES` (different classes, different order, or additional classes), the serving app's mapping must be updated to match, or predictions will be silently mislabeled.
- **Filtering rules:**
  - Only images containing at least one instance of an allowed class are kept.
  - Images with zero valid polygons after filtering (e.g., degenerate polygons with fewer than 3 points) are dropped entirely, to avoid empty-label/image mismatches.

---

## 4. Preprocessing ("Paper-Exact" Pipeline)

Every image — at both training and inference time — goes through the same deterministic preprocessing function, `preprocess_xca_paper()`, before being fed to the model:

1. Invert the grayscale image.
2. Apply a white top-hat morphological operation (50×50 kernel) to the inverted image.
3. Subtract the top-hat result from the *original* grayscale image; clip any negative values to 0.
4. Apply CLAHE contrast enhancement (`clipLimit=2.0`, `tileGridSize=(8, 8)`).
5. Convert the resulting single-channel image to 3 channels (grayscale replicated across BGR), since YOLO expects 3-channel input.

```python
def preprocess_xca_paper(img_gray: np.ndarray) -> np.ndarray:
    img_not = cv2.bitwise_not(img_gray)
    kernel = np.ones((50, 50), np.uint8)
    tophat = cv2.morphologyEx(img_not, cv2.MORPH_TOPHAT, kernel)

    raw_minus = img_gray.astype(np.int32) - tophat.astype(np.int32)
    raw_minus = np.maximum(raw_minus, 0).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(raw_minus)
```

**This function is duplicated in the serving app's backend (`backend/app.py` in the `xca-ui` repo).** The two copies must stay identical — any drift between training-time and inference-time preprocessing will silently degrade model accuracy without raising an error. If you change this function here, port the change to the serving app too (or, better, refactor both to import a shared module instead of maintaining two hand-synced copies).

---

## 5. Label Conversion (COCO → YOLOv8-seg)

The notebook converts each COCO polygon annotation into a YOLOv8-seg label line (`class x1 y1 x2 y2 ... xN yN`, coordinates normalized to `[0, 1]` per image), and writes out a standard YOLO segmentation dataset layout:

```
arcade_yolo_paperprep/
├── arcade.yaml          # dataset config: image paths + class names, in ALLOWED_CLASSES order
├── train/
│   ├── images/           # preprocessed (paper-exact) training images
│   └── labels/           # YOLOv8-seg polygon label .txt files
└── val/
    ├── images/
    └── labels/
```

`arcade.yaml` is auto-generated with class names pulled from the COCO categories, in the same order as `ALLOWED_CLASSES`:

```yaml
path: /kaggle/working/arcade_yolo_paperprep
train: train/images
val: val/images

names:
  0: <name for ARCADE id 6>
  1: <name for ARCADE id 5>
  ...
```

---

## 6. Training Configuration

Training fine-tunes the pretrained `yolov8m-seg.pt` checkpoint using the `ultralytics` Python API:

```python
model = YOLO("yolov8m-seg.pt")

model.train(
    data=yaml_path,        # arcade.yaml from Section 5
    task="segment",
    imgsz=896,
    epochs=50,
    batch=8,
    device=0,               # single GPU
    workers=2,

    project="/kaggle/working/runs_arcade",
    name="yolov8m_seg_sanity",
    exist_ok=True,

    save=True,
    save_period=1,           # checkpoint every epoch, so partial progress survives a crash

    cache=True,
    optimizer="AdamW",
    lr0=1e-3,
    cos_lr=True,
    close_mosaic=2,
)
```

| Hyperparameter | Value | Notes |
|---|---|---|
| Base checkpoint | `yolov8m-seg.pt` | Ultralytics-pretrained YOLOv8-medium segmentation model |
| Image size | 896 | Matches the inference-time `IMGSZ` used by the serving app |
| Epochs | 50 | |
| Batch size | 8 | |
| Optimizer | AdamW | |
| Initial LR | `1e-3` | |
| LR schedule | Cosine (`cos_lr=True`) | |
| Mosaic augmentation | Disabled for the final 2 epochs (`close_mosaic=2`) | Standard YOLO practice to stabilize convergence near the end of training |

Weights are saved under `runs_arcade/yolov8m_seg_sanity/weights/`, producing:
- `last.pt` — final-epoch checkpoint
- `best.pt` — checkpoint with the best validation metric during training (**this is the file the serving app deploys**)

**Inference-time defaults differ slightly from what's used in this notebook's own demo/eval cells** (which use `conf` values ranging `0.05–0.25` depending on the cell, and rely on YOLO's built-in NMS). The serving app instead uses `CONF_THR=0.10` plus a custom mask-IoU-based NMS (`MASK_IOU_THR=0.55`) rather than YOLO's built-in box NMS, since detections are deduplicated per ARCADE vessel ID rather than per raw class. If you retrain and the new model's confidence distribution shifts, re-tune those serving-side thresholds against a validation set — don't assume the old values still apply.

---

## 7. Evaluation

The notebook evaluates the trained model using Ultralytics' built-in segmentation validator:

```python
metrics = model.val(
    data=DATA_YAML,     # arcade.yaml
    task="segment",
    imgsz=896,
    conf=0.25,
    iou=0.50,
    split="val",
)
```

It also includes qualitative sanity-check cells:
- Drawing ground-truth polygons directly on preprocessed images, to visually confirm label correctness after the COCO→YOLO conversion.
- Running the trained model on individual test images (`predict_and_overlay()`) to visually compare predicted masks against ground truth, with per-instance labels and confidence scores drawn on the overlay.

---

## 8. Running This Notebook Yourself

1. Get access to the `xca-classification-haryshwa/arcade` Kaggle dataset (or an equivalent COCO-format dataset using the same ARCADE category ID scheme).
2. If running outside Kaggle (Colab, local GPU box, another cloud notebook), update all `/kaggle/input/...` and `/kaggle/working/...` paths at the top of the notebook to point at your environment instead.
3. Run all cells top to bottom. Training (Section 6) is the most time-consuming step — 50 epochs at 896px/batch 8 took multiple hours on a Kaggle GPU instance; expect similar or longer depending on your hardware.
4. After training completes, verify `runs_arcade/yolov8m_seg_sanity/weights/best.pt` and `last.pt` both exist before relying on the run.

### Extending to new/additional vessel classes

If you want to detect more (or different) vessel segments:
1. Add the new ARCADE category IDs to `ALLOWED_CLASSES` in Cell 1.
2. Re-run the full pipeline (dataset filtering → label conversion → training).
3. Update the downstream serving app's `ALLOWED_CLASSES` / `YOLO_TO_ARCADE` mapping (`backend/app.py` in the `xca-ui` repo) to exactly match the new list and order.
4. Add corresponding entries to `vessel_info.json` in the serving app for any new IDs, or the UI will fall back to a generic "Unknown (ID …)" label for them.

---

## 9. Output: Getting `best.pt` Into the App

Once training finishes, take the resulting checkpoint and hand it off to the serving app:

```
runs_arcade/yolov8m_seg_sanity/weights/best.pt  →  xca-ui/backend/model/best.pt
```

The `xca-ui` app's README documents how to place this file and start the backend/frontend. If you're publishing a new trained model for others to use, consider uploading it somewhere durable (Git LFS, a cloud bucket, Hugging Face Hub) rather than passing the file around manually, and update the `xca-ui` README's download links to point at the new version.

---

## 10. Known Gaps / TODO

1. **`segmentation.pt` (U-Net++ model) has no training code here or anywhere else currently.** The serving app also loads a separate `segmentation_models_pytorch` U-Net++ model (`timm-efficientnet-b3` encoder) for a general vessel mask, but there's no notebook documenting how it was trained, on what data/split, or with what hyperparameters. This needs to be located/reconstructed and documented the same way this repo documents the YOLOv8 run — ideally as a sibling notebook in this same repo.
2. **Kaggle-specific paths are hardcoded**, making the notebook non-portable as-is. Parameterizing the dataset root (e.g., via an environment variable or notebook parameter) would make it easier to run in Colab or locally.
3. **No requirements/environment file.** Dependencies are installed inline via `!pip install` in the first cell; there's no `requirements.txt` or `environment.yml` pinning versions, so results may not be exactly reproducible if `ultralytics` or its dependencies change behavior in a later version.
4. **No experiment tracking.** Training relies on Ultralytics' default local logging/checkpointing; there's no integration with an experiment tracker (Weights & Biases, MLflow, etc.), so comparing multiple training runs (different hyperparameters, class sets, augmentations) is manual.
5. **Preprocessing logic is duplicated, not shared.** As noted in Section 4, `preprocess_xca_paper()` exists both here and in the serving app's `backend/app.py`. A shared package/module both repos could import would remove the risk of the two drifting out of sync.
6. **No automated re-training/CI pipeline.** Retraining is a fully manual "run the notebook" process; there's no scripted or scheduled way to retrain on updated data.

---

## License / Attribution

Training data: [ARCADE](https://arcade.grand-challenge.org/) coronary angiography dataset — refer to the dataset's own license/usage terms before redistributing data or trained weights derived from it. Base model: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (see Ultralytics' license for terms covering use of pretrained checkpoints and the `ultralytics` package).
