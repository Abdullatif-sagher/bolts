<div align="center">

# 🔩 Bolt & Nut Detection Model Comparator
### A tool to benchmark YOLOv8, YOLOv9, YOLOv11, and Faster R-CNN models

</div>



> This project is a web application built with Python and Gradio to compare the performance of different AI models (YOLOv8, v9, v11, and Faster R-CNN) on the task of detecting bolts and nuts in images.

---

### ✨ Features

-   **Multi-Model Comparison:** Analyze a single image with several models simultaneously and view the results side-by-side.
-   **Detailed Analysis:** For each model, the interface displays the annotated image, inference time, and a list of detected objects with their confidence scores.
-   **Modern Web UI:** A clean, modern, and interactive web interface built with Gradio.
-   **Organized Codebase:** The project is structured into specialized modules (`config`, `detector`, `gui`, `main`) for easy maintenance and development.
-   **Image Preprocessing:** Automatically applies a grayscale filter to input images to match the model's training conditions.

---
### ✨ Core New Features

-   **Batch Processing:** Analyze **multiple images** in a single run for all selected models.
-   **Metrics Dashboard:** Ability to upload **Label files (.txt)** to calculate crucial performance metrics like **True Positives (TP)**, **False Positives (FP)**, **Precision**, **Recall**, and **mAP** for model accuracy evaluation.
-   **Smart Tightness Logic:** Apply a strict geometric logic (Smart OR Logic) for **TIGHT** or **LOOSE** classification while maintaining tolerance for common occlusion or training issues, and correcting for angle-based errors (cheating).
-   **Detailed Report:** Comprehensive table output including image name, **processing time per model**, and Confusion Matrix status (TP/FP/FN).
-   **Enhanced Visual Guidance:**
    *   Color-code **TIGHT** (Green) and **LOOSE** (Red) in both the table and on the images.
    *   Draw **Keypoint Names** ($P1, H1$...) on the output images for clearer debugging.
    *   Selectively draw only the keypoints/groups that passed the tightening logic check, hiding noisy or failing data.
-   **Guaranteed Download:** Download all processed results as a **ZIP file** to ensure correct saving of `.jpg` images.
- 
---

### 📥 Download the Model

To run the project, you need to download and extract the pre-trained model.

1. 🔗 [**Download model.zip**](https://drive.google.com/file/d/1V5XaOJlhDxmUWe3MjGwjSmbGCvIOWBSg/view?usp=drive_link)
2. 🔗 [**Download new model-11-11-25**](https://drive.google.com/drive/folders/17MS1AB-a_nNqwodIZ6tnKE2y9iNGWuNT?usp=sharing)  
3. 📂 Extract the contents of `model.zip` into the **project root directory**, next to these folders:  
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
