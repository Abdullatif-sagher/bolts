import cv2
import torch
import torchvision
from ultralytics import YOLO
from torchvision.transforms import functional as F
import numpy as np

KEYPOINT_NAMES = [
    "plate1", "plate2", "head1",
    "plate3", "plate4", "head2"
]

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_IMAGE = 0.9 
FONT_SCALE_FOOTER = 0.7 
FONT_THICKNESS = 2
POINT_RADIUS = 6
CROP_PADDING = 100

class ModelManager:
    def __init__(self, model_configs):
        self.models = {}
        for name, path in model_configs.items():
            if "Keypoint R-CNN" in name:
                model = self._load_keypoint_rcnn_auto(path)
                model.eval()
                self.models[name] = model
            else:
                self.models[name] = YOLO(path)

    def analyze_image(self, image_numpy, conf_threshold, selected_models=None):
        original_bgr = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
        analysis_results = []

        for model_name, model in self.models.items():
            if selected_models is not None and model_name not in selected_models:
                continue

            try:
                if isinstance(model, YOLO):
                    result_data = self._predict_yolo(model, original_bgr, conf_threshold)
                else:
                    result_data = self._predict_keypoint_rcnn(model, original_bgr, conf_threshold)

                annotated_image, results_list = self._draw_results(original_bgr.copy(), result_data["detections"], model_name)
                
                if result_data["detections"]:
                    x1, y1, x2, y2 = result_data["detections"][0]["box"]
                    annotated_image = self._crop_image_with_padding(annotated_image, x1, y1, x2, y2, CROP_PADDING)

                composite = self._create_info_image(result_data, annotated_image, results_list, model_name)
                
                overall_status = "PASS"
                status_html = "<span style='color:green; font-weight:bold'>PASS</span>"

                for res in results_list:
                    if "LOOSE" in res['status']:
                        overall_status = "LOOSE"
                        status_html = "<span style='color:red; font-weight:bold'>LOOSE</span>"
                        break
                
                if not results_list:
                    overall_status = "NO DETECTIONS"
                    status_html = "<span style='color:gray'>NO DETECTIONS</span>"
                elif overall_status == "PASS":
                     overall_status = "TIGHT"
                     status_html = "<span style='color:green; font-weight:bold'>TIGHT</span>"

                analysis_results.append({
                    "model_name": model_name,
                    "image": composite,
                    "status_text": overall_status,
                    "status_html": status_html,
                    "detections": result_data["detections"],
                    "details": f"Detected: {len(results_list)}"
                })

            except Exception as e:
                print(f"Error processing with {model_name}: {e}")
                err_img = self._create_error_image(f"Error with {model_name}")
                analysis_results.append({
                    "model_name": model_name,
                    "image": err_img,
                    "status_text": "ERROR",
                    "status_html": "<span style='color:red'>ERROR</span>",
                    "detections": [],
                    "details": str(e)
                })
        
        return analysis_results
    
    def _crop_image_with_padding(self, image, x1, y1, x2, y2, padding):
        H, W = image.shape[:2]
        x_min = max(0, x1 - padding)
        y_min = max(0, y1 - padding)
        x_max = min(W, x2 + padding)
        y_max = min(H, y2 + padding) 
        return image[int(y_min):int(y_max), int(x_min):int(x_max)]

    def _load_keypoint_rcnn_auto(self, path):
        ckpt = torch.load(path, map_location=torch.device('cpu'))
        if isinstance(ckpt, torch.nn.Module): return ckpt
        if not isinstance(ckpt, dict): raise RuntimeError("Unsupported checkpoint format.")
        sd = ckpt.get('model_state_dict', ckpt)
        cls_w_key = "roi_heads.box_predictor.cls_score.weight"
        num_classes = sd[cls_w_key].shape[0] if cls_w_key in sd else 3
        kps_w_key = "roi_heads.keypoint_predictor.kps_score.weight" 
        if kps_w_key not in sd: kps_w_key = "roi_heads.keypoint_predictor.kps_score_lowres.weight" 
        num_keypoints = sd[kps_w_key].shape[1] if kps_w_key in sd else len(KEYPOINT_NAMES)
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=None, num_classes=num_classes, num_keypoints=num_keypoints
        )
        model.load_state_dict(sd, strict=False)
        return model

    def _predict_yolo(self, model, image_bgr, conf):
        results = model.predict(image_bgr, conf=conf, verbose=False)[0]
        detections = []
        if hasattr(results, 'keypoints') and results.keypoints is not None:
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                box_xyxy = box.xyxy[0].cpu().numpy().astype(int)
                if results.keypoints.xy.shape[0] > i:
                    keypoints_instance = results.keypoints.xy[i].cpu().numpy()
                else:
                    keypoints_instance = np.empty((0, 2))
                
                kps_with_visibility = np.hstack([keypoints_instance, np.full((keypoints_instance.shape[0], 1), 1)])
                
                detections.append({
                    "id": i + 1, "class_name": results.names.get(cls_id, str(cls_id)), "confidence": float(box.conf[0]),
                    "box": box_xyxy, "keypoints": kps_with_visibility
                })
        else:
            for i, box in enumerate(results.boxes):
                cls_id = int(box.cls[0])
                detections.append({
                    "id": i + 1, "class_name": results.names.get(cls_id, str(cls_id)), "confidence": float(box.conf[0]),
                    "box": box.xyxy[0].cpu().numpy().astype(int), "keypoints": None
                })
        return {"detections": detections}

    def _predict_keypoint_rcnn(self, model, image_bgr, conf):
        img_tensor = F.to_tensor(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            outputs = model(img_tensor)

        out = outputs[0]
        scores = out.get('scores', torch.empty(0)).cpu().numpy()
        boxes = out.get('boxes', torch.empty(0)).cpu().numpy().astype(np.float32)
        labels = out.get('labels', torch.empty(0)).cpu().numpy().astype(np.int64)
        keypoints = out.get('keypoints', torch.empty(0)).cpu().numpy() if isinstance(out.get('keypoints', None), torch.Tensor) else np.empty((0,0,3))

        detections = []
        for i in range(len(scores)):
            if scores[i] < conf: continue
            box = boxes[i].astype(int)
            kps = keypoints[i] if keypoints.shape[0] > i else None
            class_id = labels[i] if labels is not None and len(labels) > i else 1
            class_name = "bolt" if class_id == 1 else "nut"
            detections.append({
                "id": len(detections) + 1, "class_name": class_name, "confidence": float(scores[i]),
                "box": box, "keypoints": kps
            })
        return {"detections": detections}

    def _draw_results(self, image_bgr, detections, model_name):
        results_text = []

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            kps = det.get("keypoints")
            
            status_text = "Status: N/A"
            current_box_color = (0, 165, 255) 
            
            draw_left = False
            draw_right = False
            is_tight = False

            if kps is not None and len(kps) >= 6: 
                P1, P2, H1 = kps[0], kps[1], kps[2]
                P3, P4, H2 = kps[3], kps[4], kps[5]
                
                tight_G1 = False
                tight_G2 = False

                if P1[2] > 0 and P2[2] > 0 and H1[2] > 0:
                    vector_p1_p2 = P2[:2] - P1[:2]
                    vector_p1_h1 = H1[:2] - P1[:2]
                    mag_sq = np.dot(vector_p1_p2, vector_p1_p2)
                    if mag_sq > 1e-6:
                        t = np.dot(vector_p1_h1, vector_p1_p2) / mag_sq
                        if 0.1 <= t <= 0.9:
                            tight_G1 = True

                if P3[2] > 0 and P4[2] > 0 and H2[2] > 0:
                    vector_p3_p4 = P4[:2] - P3[:2]
                    vector_p3_h2 = H2[:2] - P3[:2]
                    mag_sq = np.dot(vector_p3_p4, vector_p3_p4)
                    if mag_sq > 1e-6:
                        t = np.dot(vector_p3_h2, vector_p3_p4) / mag_sq
                        if 0.1 <= t <= 0.9:
                            tight_G2 = True

                if tight_G1 or tight_G2:
                    status_text = "TIGHT"
                    current_box_color = (0, 255, 0)
                    is_tight = True
                    if tight_G1: draw_left = True
                    if tight_G2: draw_right = True
                else:
                    status_text = "LOOSE"
                    current_box_color = (0, 0, 255)
                    draw_left = True
                    draw_right = True

            cv2.rectangle(image_bgr, (x1, y1), (x2, y2), current_box_color, 2)
            cv2.putText(image_bgr, label, (x1, max(0, y1 - 10)),
                        FONT, FONT_SCALE_IMAGE, current_box_color, FONT_THICKNESS, cv2.LINE_AA)
            cv2.putText(image_bgr, status_text, (x1, y2 + 25), FONT, 0.8, current_box_color, 1, cv2.LINE_AA)

            if kps is not None:
           
                if draw_left:
                    for idx in [0, 1, 2]:
                        if idx < len(kps) and kps[idx][2] > 0:
                            cx, cy = int(kps[idx][0]), int(kps[idx][1])
                            color = (0, 255, 0) if is_tight else (0, 0, 255)
                            cv2.circle(image_bgr, (cx, cy), POINT_RADIUS, color, -1)
                     
                            cv2.putText(image_bgr, KEYPOINT_NAMES[idx], (cx + 8, cy + 8),
                                    FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
             
                if draw_right:
                    for idx in [3, 4, 5]:
                        if idx < len(kps) and kps[idx][2] > 0:
                            cx, cy = int(kps[idx][0]), int(kps[idx][1])
                            color = (0, 255, 0) if is_tight else (0, 0, 255)
                            cv2.circle(image_bgr, (cx, cy), POINT_RADIUS, color, -1)
                            # ✅ إعادة كتابة الاسم هنا
                            cv2.putText(image_bgr, KEYPOINT_NAMES[idx], (cx + 8, cy + 8),
                                    FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            results_text.append({
                "id": det['id'], 
                "class_name": det['class_name'], 
                "confidence": det['confidence'], 
                "status": status_text
            })

        return image_bgr, results_text

    def _create_info_image(self, result_data, base_bgr, results_list, model_name):
        detections = result_data.get("detections", [])
        footer_h = max(70, len(results_list) * 30 + 40)
        h, w = base_bgr.shape[:2]

        font_color = (230, 230, 230)
        bg_color = (35, 35, 35)
        composite = np.full((h + footer_h, w, 3), bg_color, dtype=np.uint8)
        composite[0:h, 0:w] = base_bgr

        header = f"{model_name} | Detected: {len(detections)}"
        cv2.putText(composite, header, (12, h + 28), FONT, FONT_SCALE_FOOTER, font_color, FONT_THICKNESS, cv2.LINE_AA)

        y = h + 60
        if not results_list:
            cv2.putText(composite, "No objects detected.", (12, y), FONT, 0.8, font_color, 1, cv2.LINE_AA)
        else:
            for res in results_list:
                st = res['status']
                c = font_color
                if "TIGHT" in st: c = (0, 255, 0)
                if "LOOSE" in st: c = (0, 0, 255)
                
                txt = f"ID {res['id']}: {res['class_name']} | {st}"
                cv2.putText(composite, txt, (15, y), FONT, FONT_SCALE_FOOTER, c, 1, cv2.LINE_AA)
                y += 30

        return cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)

    def _create_error_image(self, text):
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.putText(img, "Analysis Failed", (30, 180), FONT, 1.2, (0, 0, 255), 3)
        cv2.putText(img, text, (20, 240), FONT, 0.8, (0, 0, 255), 2)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)