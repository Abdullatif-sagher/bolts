# Model paths - VERY IMPORTANT: Update these paths to your trained models
MODEL_PATHS = {
    "YOLOv8": "models/yolo8m.pt",
    "YOLOv9": "models/yolo9l.pt",
    "YOLOv11": "train5details/best_bolt_nut_yolo11.pt",
     "Faster R-CNN": "torchvision"
}

# Default confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.4

# # Window Configuration
# WINDOW_TITLE = "YOLO Model Comparator"
# WINDOW_GEOMETRY = "1200x900"


# BG_COLOR = "#2c3e50"
# TEXT_COLOR = "#ecf0f1"
# ACCENT_COLOR = "#3498db"
# FRAME_COLOR = "#34495e"
# FONT_NORMAL = ("Helvetica", 12)
# FONT_BOLD = ("Helvetica", 12, "bold")