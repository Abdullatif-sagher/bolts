# detector.py
import cv2
import torch
import torchvision
from ultralytics import YOLO
from torchvision.transforms import functional as F
import numpy as np

# Names of keypoints in the training order
KEYPOINT_NAMES = ["plate_2", "plate_1", "head"]

class ModelManager:
    """
    Manages loading, running, and visualizing predictions 
    from YOLO and custom Keypoint R-CNN models.
    """

    def __init__(self, model_configs):
        """
            Initialize the ModelManager and load all configured models.
            
            Args:
                model_configs (dict): A dictionary mapping model names to file paths.
        """
        self.models = {}
        print("Loading models...")


        for name, path in model_configs.items():
            if "Keypoint R-CNN" in name:
                # Load custom Keypoint R-CNN checkpoint
                print(f"- Loading Custom Keypoint R-CNN from: {path}")
                model = self._load_keypoint_rcnn_auto(path)
                model.eval()
                self.models[name] = model
            else:
                # Load YOLO model
                print(f"- Loading {name} from: {path}")
                self.models[name] = YOLO(path)
        print("All models loaded successfully!")

    # ---------- Public API ----------
    def analyze_image(self, image_numpy, conf_threshold):
        """
        Run all loaded models on a single RGB image.
        
        Args:
            image_numpy (np.ndarray): Input image in RGB format (HxWx3).
            conf_threshold (float): Confidence threshold for detections.
        
        Returns:
            list[np.ndarray]: A list of annotated RGB images (one per model).
        """
        original_bgr = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
        all_model_results = []

        for model_name, model in self.models.items():
            try:
                # Run YOLO or Keypoint R-CNN depending on model type
                if isinstance(model, YOLO):
                    result_data = self._predict_yolo(model, original_bgr, conf_threshold)
                else:  # Keypoint R-CNN
                    result_data = self._predict_keypoint_rcnn(model, original_bgr, conf_threshold)

                # Draw detections and compute visual results
                annotated_image, results_list = self._draw_results(original_bgr.copy(), result_data["detections"], model_name)
                
                # Create composite image with info footer
                composite = self._create_info_image(result_data, annotated_image, results_list)
                
                all_model_results.append(composite)
            except Exception as e:
                print(f"Error processing with {model_name}: {e}")
                all_model_results.append(self._create_error_image(f"Error with {model_name}"))
        return all_model_results

    # ---------- Loaders ----------
    def _load_keypoint_rcnn_auto(self, path):
        """
        Load a Keypoint R-CNN model automatically from checkpoint.
        
        Supports both:
            - Full model checkpoints (torch.save(model, path))
            - State dict checkpoints (torch.save(model.state_dict(), path))
        """
        ckpt = torch.load(path, map_location=torch.device('cpu'))

        # Case 1: Full model object
        if isinstance(ckpt, torch.nn.Module):
            print("  • Detected full model checkpoint. Loading directly.")
            return ckpt

        # Case 2: State dict
        if not isinstance(ckpt, dict):
            raise RuntimeError("Unsupported checkpoint format for Keypoint R-CNN.")

        sd = ckpt

        # Infer number of classes from classifier weights
        num_classes = None
        cls_w_key = "roi_heads.box_predictor.cls_score.weight"
        if cls_w_key in sd:
            num_classes = sd[cls_w_key].shape[0]
            print(f"  • Inferred num_classes from checkpoint: {num_classes}")
        else:
            # Fallback to 2 (background + bolt)
            num_classes = 2
            print("  • Could not infer num_classes; defaulting to 2 (background + bolt).")

        # Infer number of keypoints from keypoint predictor weights
        num_keypoints = None
        kps_w_key = "roi_heads.keypoint_predictor.kps_score_lowres.weight"
        if kps_w_key in sd:
            num_keypoints = sd[kps_w_key].shape[0]
            print(f"  • Inferred num_keypoints from checkpoint: {num_keypoints}")
        else:
            num_keypoints = len(KEYPOINT_NAMES)
            print(f"  • Could not infer num_keypoints; defaulting to {num_keypoints}.")

        # Build the architecture with inferred sizes, then load weights
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=None,
            num_classes=num_classes,
            num_keypoints=num_keypoints
        )
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  • load_state_dict with strict=False.\n    Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
        return model

    # ---------- Predictors ----------

    def _predict_yolo(self, model, image_bgr, conf):
        """
        Run inference using YOLO model and extract detections.
        """

        results = model.predict(image_bgr, conf=conf, verbose=False)[0]
        speed = results.speed
        total_time = float(speed.get('preprocess', 0.0) + speed.get('inference', 0.0) + speed.get('postprocess', 0.0))

        detections = []
        
        # If YOLO returned keypoints
        if hasattr(results, 'keypoints') and results.keypoints is not None:
            
            keypoints_data = results.keypoints.xy[0].cpu().numpy()
            
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                box_xyxy = box.xyxy[0].cpu().numpy().astype(int)

                # Keypoints per instance
                keypoints_instance = results.keypoints.xy[i].cpu().numpy() if results.keypoints.xy.shape[0] > i else np.empty((0, 2))
                
                # Add visibility flag (2 = visible)
                kps_with_visibility = np.hstack([keypoints_instance, np.full((keypoints_instance.shape[0], 1), 2)])

                detections.append({
                    "id": i + 1,
                    "class_name": results.names.get(cls_id, str(cls_id)),
                    "confidence": float(box.conf[0]),
                    "box": box_xyxy,
                    "keypoints": kps_with_visibility
                })

        else:
            # Fallback: boxes only
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                detections.append({
                    "id": i + 1,
                    "class_name": results.names.get(cls_id, str(cls_id)),
                    "confidence": float(box.conf[0]),
                    "box": box.xyxy[0].cpu().numpy().astype(int),
                    "keypoints": None
                })

        return {"detections": detections, "inference_time": total_time}


    def _predict_keypoint_rcnn(self, model, image_bgr, conf):
        img_tensor = F.to_tensor(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)

        out = outputs[0]
        scores = out.get('scores', torch.empty(0)).cpu().numpy()
        boxes = out.get('boxes', torch.empty(0)).cpu().numpy().astype(np.float32)
        labels = out.get('labels', torch.empty(0)).cpu().numpy().astype(np.int64)
        keypoints = out.get('keypoints', torch.empty(0))  # Tensor [N, K, 3]
        keypoints = keypoints.cpu().numpy() if isinstance(keypoints, torch.Tensor) else np.empty((0, 0, 3))

        detections = []
        for i in range(len(scores)):
            if scores[i] < conf:
                continue
            box = boxes[i].astype(int)
            kps = keypoints[i] if keypoints.shape[0] > i else None  # [K, 3]
            detections.append({
                "id": len(detections) + 1,
                "class_name": "bolt",   # Single class
                "confidence": float(scores[i]),
                "box": box,
                "keypoints": kps        
            })

        return {"detections": detections, "inference_time": 0.0}


     # ---------- Drawing Helpers ----------

    def _draw_results(self, image_bgr, detections, model_name):
        """
        Draw bounding boxes, keypoints, and status indicators on the image.
        """
        color = (0, 255, 0) if "Keypoint" in model_name else (0, 150, 255)
        results_text = []

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = f"{det['class_name']} {det['confidence']:.2f}"

            # Draw bounding box
            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_bgr, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            
            status_text = "Status: N/A"

            # Draw keypoints and geometry checks if available
            if det.get("keypoints") is not None and len(det["keypoints"]) > 0:
                plate1_coords = None
                plate2_coords = None
                head_coords = None
                
                for idx, (kx, ky, kv) in enumerate(det["keypoints"]):
                    if kv < 1:
                        continue
                    
                    cx, cy = int(kx), int(ky)
                    
                    if idx < len(KEYPOINT_NAMES):
                        kp_name = KEYPOINT_NAMES[idx]
                        if kp_name == "plate_1":
                            plate1_coords = (cx, cy)
                        elif kp_name == "plate_2":
                            plate2_coords = (cx, cy)
                        elif kp_name == "head":
                            head_coords = (cx, cy)

                    cv2.circle(image_bgr, (cx, cy), 3, (0, 0, 0), -1)
                    kp_name_to_draw = KEYPOINT_NAMES[idx] if idx < len(KEYPOINT_NAMES) else f"kp{idx}"
                    cv2.putText(image_bgr, kp_name_to_draw, (cx + 4, cy - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

                if plate1_coords and plate2_coords and head_coords:
                    h, w, _ = image_bgr.shape

                    # 1. Draw the infinite line between plate_1 and plate_2 in yellow
                    if plate1_coords != plate2_coords:
                        m_yellow_denom = (plate2_coords[0] - plate1_coords[0])
                        m_yellow = (plate2_coords[1] - plate1_coords[1]) / m_yellow_denom if m_yellow_denom != 0 else float('inf')
                        c_yellow = plate1_coords[1] - m_yellow * plate1_coords[0]

                        if m_yellow == float('inf'):
                            pt1_yellow = (plate1_coords[0], 0)
                            pt2_yellow = (plate1_coords[0], h)
                        else:
                            pt1_yellow = (0, int(m_yellow * 0 + c_yellow))
                            pt2_yellow = (w, int(m_yellow * w + c_yellow))
                        cv2.line(image_bgr, pt1_yellow, pt2_yellow, (0, 255, 255), 2, cv2.LINE_AA)
                    
                    # 2. Draw the infinite perpendicular line from head in orange
                    if plate1_coords != plate2_coords:
                        m_yellow_denom = (plate2_coords[0] - plate1_coords[0])
                        m_yellow = (plate2_coords[1] - plate1_coords[1]) / m_yellow_denom if m_yellow_denom != 0 else float('inf')
                        
                        m_perp = -1/m_yellow if m_yellow != 0 else float('inf')
                        c_perp = head_coords[1] - m_perp * head_coords[0]

                        if m_perp == float('inf'): # Vertical line
                            pt1_orange = (head_coords[0], 0)
                            pt2_orange = (head_coords[0], h)
                        else:
                            pt1_orange = (0, int(m_perp * 0 + c_perp))
                            pt2_orange = (w, int(m_perp * w + c_perp))

                        cv2.line(image_bgr, pt1_orange, pt2_orange, (0, 165, 255), 2, cv2.LINE_AA)

                        # Calculate intersection and check if it's on the segment
                        intersection_x = (c_perp - c_yellow) / (m_yellow - m_perp) if m_yellow - m_perp != 0 else head_coords[0]
                        intersection_y = m_yellow * intersection_x + c_yellow
                        
                        is_on_segment = False
                        if intersection_x is not None:
                            min_x, max_x = min(plate1_coords[0], plate2_coords[0]), max(plate1_coords[0], plate2_coords[0])
                            min_y, max_y = min(plate1_coords[1], plate2_coords[1]), max(plate1_coords[1], plate2_coords[1])
                            
                            if min_x <= intersection_x <= max_x and min_y <= intersection_y <= max_y:
                                is_on_segment = True
                        
                        # Draw the intersection point in white if it's on the segment
                        if is_on_segment:
                            cv2.circle(image_bgr, (int(intersection_x), int(intersection_y)), 5, (255, 255, 255), -1)

                        # Determine the status
                        if is_on_segment:
                            status_text = "Status: Tight"
                        else:
                            status_text = "Status: Losse"
                    else:
                        status_text = "Status: N/A"

            results_text.append({
                "id": det['id'],
                "class_name": det['class_name'],
                "confidence": det['confidence'],
                "status": status_text
            })
        
        return image_bgr, results_text
    
    # ---------- UI helpers ----------

    def _create_info_image(self, result_data, base_bgr, results_list):
        """
        Create a composite image with detections and footer text summary.
        """
        detections = result_data.get("detections", [])
        time_ms = float(result_data.get("inference_time", 0.0))

       
        footer_h = 40 + (len(results_list) * 25)
        
        h, w = base_bgr.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (230, 230, 230)
        bg_color = (45, 55, 72)

        composite = np.full((h + footer_h, w, 3), bg_color, dtype=np.uint8)
        composite[0:h, 0:w] = base_bgr

        header = f"Time: {time_ms:.1f} ms | Found: {len(detections)}"
        cv2.putText(composite, header, (10, h + 25), font, font_scale, font_color, 1, cv2.LINE_AA)

        if not results_list:
            cv2.putText(composite, "No objects detected.", (10, h + 55), font, 0.5, font_color, 1, cv2.LINE_AA)
        else:
            y = h + 55
            for res in results_list:
                txt = f"ID {res['id']}: {res['class_name']} ({res['confidence']:.2%}) | {res['status']}"
                cv2.putText(composite, txt, (15, y), font, font_scale, font_color, 1, cv2.LINE_AA)
                y += 25

        return cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)

    def _create_error_image(self, text):
        """
        Create an error placeholder image if inference fails.
        """
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.putText(img, "Analysis Failed", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(img, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)