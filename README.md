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

### 📁 Project Structure

```
/bolts/
|
├── main.py           # Main entry point to launch the application
├── gui.py            # Contains the Gradio UI code
├── detector.py       # The AI engine for model management and inference
├── config.py         # All project settings and model paths
└── requirements.txt  # Python dependencies for easy setup
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
