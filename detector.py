from ultralytics import YOLO
import cv2
from collections import Counter
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np

# (COCO_INSTANCE_CATEGORY_NAMES list remains the same as before)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class ModelManager:
    def __init__(self, model_configs):
        self.models = {}
        print("Loading models...")
        for name, path in model_configs.items():
            if path == "torchvision":
                self.models[name] = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
                self.models[name].eval()
            else:
                self.models[name] = YOLO(path)
        print("All models loaded successfully!")

    def analyze_image(self, image_numpy, conf_threshold):
        gray_image = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2GRAY)
        preprocessed_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        
        all_model_results = []
        for model_name, model in self.models.items():
            composite_image = None
            try:
                if isinstance(model, YOLO):
                    composite_image = self._predict_yolo(model, model_name, preprocessed_image, image_numpy, conf_threshold)
                elif "fasterrcnn" in str(type(model)).lower():
                    composite_image = self._predict_faster_rcnn(model, model_name, preprocessed_image, image_numpy, conf_threshold)
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                # Create a blank image with an error message if something goes wrong
                composite_image = self._create_info_image(f"Error with {model_name}", [])

            all_model_results.append(composite_image)
        return all_model_results

    def _predict_yolo(self, model, model_name, image, original_image, conf):
        result = model.predict(image, conf=conf, verbose=False)[0]
        annotated_image = result.plot(img=cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        
        speed = result.speed
        total_time = speed['preprocess'] + speed['inference'] + speed['postprocess']
        detections = [{"class_name": result.names[int(box.cls[0])], "confidence": float(box.conf[0])} for box in result.boxes]
        
        return self._create_info_image(f"Time: {total_time:.1f} ms | Found: {len(detections)}", detections, annotated_image)

    def _predict_faster_rcnn(self, model, model_name, image, original_image, conf):
        img_tensor = F.to_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            predictions = model(img_tensor)
        
        annotated_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
        detections = []
        for i in range(len(predictions[0]['scores'])):
            score = predictions[0]['scores'][i]
            if score >= conf:
                box = predictions[0]['boxes'][i].numpy().astype(int)
                class_name = COCO_INSTANCE_CATEGORY_NAMES[predictions[0]['labels'][i].item()]
                detections.append({"class_name": class_name, "confidence": float(score)})
                cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        
        return self._create_info_image(f"Time: 0.0 ms | Found: {len(detections)}", detections, annotated_image)
    
    # --- NEW: Function to draw the details table onto an image ---
    def _create_info_image(self, header_text, detections, base_image=None):
        # Define dimensions and colors
        img_width = 600 if base_image is None else base_image.shape[1]
        img_height = 400 if base_image is None else base_image.shape[0]
        
        footer_height = 40 + (len(detections) * 25) # Dynamic height for the table
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (230, 230, 230) # Light gray
        bg_color = (45, 55, 72) # Dark gray-blue

        # Create a new canvas with space for the footer
        composite_image = np.full((img_height + footer_height, img_width, 3), bg_color, dtype=np.uint8)
        
        # Place the annotated image at the top
        if base_image is not None:
            composite_image[0:img_height, 0:img_width] = base_image

        # --- Draw the table ---
        # Draw header
        cv2.putText(composite_image, header_text, (10, img_height + 25), font, font_scale, font_color, 1, cv2.LINE_AA)

        # Draw detection details
        if not detections:
            cv2.putText(composite_image, "No objects detected.", (10, img_height + 55), font, 0.5, font_color, 1, cv2.LINE_AA)
        else:
            y_pos = img_height + 55
            for i, det in enumerate(detections):
                text = f"ID {i+1}: {det['class_name']} ({det['confidence']:.2%})"
                cv2.putText(composite_image, text, (15, y_pos), font, font_scale, font_color, 1, cv2.LINE_AA)
                y_pos += 25
        
        return cv2.cvtColor(composite_image, cv2.COLOR_BGR2RGB)