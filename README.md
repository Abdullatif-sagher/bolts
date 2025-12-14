<div align="center">

# 🔩 Bolt AI Pro  
### Industrial Bolt & Nut Detection – Model Comparator & Analytics Dashboard
### version 12-12-25
</div>

---

> **Bolt AI Pro** is a professional industrial computer vision application built with **Python** and **Gradio** to analyze, compare, and benchmark multiple AI models for **bolt & nut inspection (Tight / Loose)** using a modern UI and advanced analytics.

---
> **Note** Rename model like **yolo8-l-best.pt** > **1-** name of model , **2-** TYPE: L  , **3-** and is Best model

## ✨ Key Features

### 🧠 Multi-Model AI Support
- **YOLOv8**
- **YOLOv11** *(ready – model path can be added later)*
- **YOLOv12** *(ready – model path can be added later)*
- **Faster R-CNN**
- **Keypoint R-CNN**

> Models with empty paths are **automatically skipped** (no crashes).  
> Once a path is added, the model becomes active immediately.

---

### 🎨 Modern Web UI (Gradio)
- Clean, professional dashboard
- Multi-tab layout:
  - **Gallery**
  - **Analytics**
  - **Results Table**
- Scroll-enabled image gallery
- One-click ZIP download of results

---

## 📊 Advanced Analytics (New)

### ✅ Image-Level Metrics (Industrial-Grade)
Designed for real inspection logic instead of pure IoU:

- **Precision**
- **Recall**
- **F1-Score**

Evaluation is done at the **image decision level**:
> *Is the bolt detected correctly or not?*

This approach is far more suitable for industrial QA systems.

---

### 📈 ROC Curve + AUC
- Full **ROC Curve** per model
- Automatic **AUC calculation**
- Threshold-independent comparison
- Ideal for:
  - Model benchmarking
  - Threshold tuning
  - Deployment decisions

---

### ⏱ Performance Metrics
- Inference time per model
- Speed comparison across models

---

## 🖼️ UI Overview

### 🖼 Gallery Tab
- Displays all processed images
- Shows model name + classification result
- Supports scrolling for large batches

### 📊 Analytics Tab
- Performance (Latency)
- Image-level Metrics (Precision / Recall / F1)
- ROC Curve with AUC

### 📄 Table Tab
- Per-image results
- Processing time
- Detection summary

---

## 📦 Export & Reports
- Download all outputs as a **ZIP file**:
  - Annotated images
  - Result tables
- Designed for easy reporting and auditing

---


### 📥 Download the Model

To run the project, you need to download and extract the pre-trained model.

1. 🔗 [**Download model version 12-12-25**](https://drive.google.com/drive/folders/1fm1HWQm9Tp-cAskEXOI7BVbQtGtMBFTT?usp=drive_link)  
2. 📂 Extract the contents of `model.zip` into the **project root directory**, next to these folders:  
   - 📄 `config`
   - 📄 `detector`
   - 📄 `main`
   - 📄 `gui`

After extraction, your project structure should look like this:


### 📁 Project Structure

```
/bolts/
|
├── 📁model          
├── 📄main.py           # Main entry point to launch the application
├── 📄gui.py            # Contains the Gradio UI code
├── 📄detector.py       # The AI engine for model management and inference
├── 📄config.py         # All project settings and model paths
└── 📄requirements.txt  # Python dependencies for easy setup
```

---

### 🚀 Setup and Launch

Follow these steps to set up and run the project on your local machine.

#### 1. Clone the Repository
```bash
git clone https://github.com/Abdullatif-sagher/bolts.git
```
```bash
cd bolts
```

#### 2. Create and Activate a Virtual Environment (Recommended)
```bash
# Create a virtual environment
python -m venv .venv

# Activate on Windows
.\.venv\Scripts\activate

# Activate on macOS/Linux
source .venv/bin/activate
```

#### 3. Install Dependencies
Use the provided `requirements.txt` file to install all necessary libraries with a single command.
```bash
pip install -r requirements.txt
```

#### 4. Configure Models
Before launching, you **must** edit the `config.py` file and update the `MODEL_PATHS` dictionary with the correct paths to your trained `.pt` files.

```python
# Example in config.py
MODEL_PATHS = {
    "YOLOv8": "path/to/your/yolov8_best.pt",
    "YOLOv9": "path/to/your/yolov9_best.pt",
    "YOLOv11": "path/to/your/yolov11_best.pt",
    "Faster R-CNN": "torchvision"
}
```

---

### 🛠️ How to Use

After completing the setup steps, launch the application from your terminal:
```bash
python main.py
```
A local URL will appear in your terminal (usually `http://127.0.0.1:7860`). Open this link in your web browser to start using the application.
