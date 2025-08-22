#config.py
# Model paths - IMPORTANT: Update these with the actual paths to your trained models
MODEL_PATHS = {
    "YOLOv8": "models/yolo model/yolo8m-pose-19-8.pt",
    "YOLOv9": "models/best3y11.pt",
    "YOLOv11": "models/train5details/best_bolt_nut_yolo11.pt",
    "Keypoint R-CNN": "models/keypoint-rcnn-train-1/keypoint_rcnn_full_model.pt" 
}

# Default confidence threshold for predictions
CONFIDENCE_THRESHOLD = 0.90