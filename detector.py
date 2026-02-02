import time
import cv2
import torch
import torchvision
from ultralytics import YOLO
from torchvision.transforms import functional as F
import numpy as np
import traceback

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
    def __init__(self, model_configs, device=None):
        self.models = {}
        
     
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
               
                gpu_name = torch.cuda.get_device_name(0)
                print(f"\n[ModelManager] ✅ GPU DETECTED: {gpu_name}")
                print(f"[ModelManager] Mode: CUDA (Fast)\n")
            else:
                self.device = "cpu"
                print(f"\n[ModelManager] ⚠️ WARNING: GPU NOT DETECTED! Running on CPU (Slow).")
                print(f"[ModelManager] Ensure you have NVIDIA Drivers & PyTorch CUDA installed.\n")

     
        self.model_class_map = {}
        
        for name, path in model_configs.items():

            # ===== SKIP MODELS WITH EMPTY PATH =====
            if not path or not isinstance(path, str) or path.strip() == "":
                print(f"[SKIP] {name} has no model path yet")
                continue

            try:
                start = time.time()

                # -------- Keypoint R-CNN --------
                if "keypoint" in name.lower():
                    print(f"[Loading Keypoint R-CNN] {name}  path={path}")
                    model = self._load_keypoint_rcnn_auto(path)
                    model.to(self.device)
                    model.eval()
                    self.models[name] = ("keypoint_rcnn", model)
                    print(f"[Loaded Keypoint R-CNN] {name} in {time.time()-start:.2f}s")
                    continue

                # -------- Try YOLO (Ultralytics) --------
                try:
                    print(f"[Trying YOLO load] {name}  path={path}")
                  
                    y = YOLO(path)
                    
                    if self.device == "cuda":
                        y.to("cuda")
                    
                    self.models[name] = ("yolo", y)
                    print(f"[Loaded YOLO] {name} via Ultralytics in {time.time()-start:.2f}s")
                    continue
                except Exception as e_y:
                    print(f"[YOLO load failed] {name}: {e_y}")

                # -------- Fallback: Faster R-CNN --------
                print(f"[Loading Detection Model] {name} as torchvision detection")
                model = self._load_detection_auto(path)
                model.to(self.device)
                model.eval()
                self.models[name] = ("detection", model)

                # Default class mapping
                self.model_class_map[name] = {1: "bolt-loose", 2: "bolt-tight"}

                print(f"[Loaded Detection Model] {name} in {time.time()-start:.2f}s")

            except Exception as e:
                print(f"[Failed loading model {name}] {e}")
                traceback.print_exc()

    # ---------------------------
    # Public analyze interface
    # ---------------------------
    def analyze_image(self, image_numpy, conf_threshold,
                      selected_models=None, keypoint_mode=False):
        """
        image_numpy: RGB image (H,W,3) uint8
        conf_threshold: float
        selected_models: list of model keys to run (None => run all appropriate)
        keypoint_mode: if True -> run ONLY keypoint model (Keypoint R-CNN)
                       if False -> run only box models (YOLO / detection)
        """
        original_bgr = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2BGR)
        out_results = []

        # --- KEYPOINT MODE: only run keypoint model ---
        if keypoint_mode:
            kp_names = [n for n in self.models.keys() if "keypoint" in n.lower()]
            if not kp_names:
                print("[analyze_image] keypoint_mode=True but no Keypoint model loaded")
                return []
            # assume the first keypoint model
            kp_name = kp_names[0]
            model_type, model = self.models[kp_name]
            print(f"[analyze_image] Running Keypoint model: {kp_name} on {self.device}")
            try:
                dets = self._predict_keypoint_safe(model, original_bgr, conf_threshold)
                annotated, parsed = self._draw_keypoint_results(original_bgr.copy(), dets)
                composite = self._info_image_keypoint(annotated, parsed, kp_name)
                overall_status = "NO DETECTIONS" if len(parsed) == 0 else (
                    "TIGHT" if any(r["status"] == "TIGHT" for r in parsed) else "LOOSE"
                )
                out_results.append({
                    "model_name": kp_name,
                    "image": composite,
                    "status_text": overall_status,
                    "status_html": self._status_color(overall_status),
                    "detections": dets,
                    "details": f"Detected: {len(parsed)}"
                })
            except Exception as e:
                print(f"[Error running keypoint model {kp_name}] {e}")
                traceback.print_exc()
            return out_results

        # --- NORMAL mode: run only box models (ignore keypoint models) ---
        for model_name, (mtype, mobj) in self.models.items():
            if "keypoint" in model_name.lower():
                continue
            if selected_models and model_name not in selected_models:
                continue

            try:
                start = time.time()
                if mtype == "yolo":
                    dets = self._predict_yolo_safe(mobj, original_bgr, conf_threshold)
                else:
                    dets = self._predict_box_model_safe(mobj, original_bgr, conf_threshold)
                elapsed = time.time() - start
                print(f"[predict] model={model_name} type={mtype} detections={len(dets)} time={elapsed:.3f}s device={self.device}")

                annotated, parsed = self._draw_box_results(original_bgr.copy(), dets)
                composite = self._info_image_box(annotated, parsed, model_name)
                overall_status = "NO DETECTIONS" if len(parsed) == 0 else (
                    "TIGHT" if any(r["status"] == "TIGHT" for r in parsed) else "LOOSE"
                )

                out_results.append({
                    "model_name": model_name,
                    "image": composite,
                    "status_text": overall_status,
                    "status_html": self._status_color(overall_status),
                    "detections": dets,
                    "details": f"Detected: {len(parsed)}"
                })

            except Exception as e:
                print(f"[Error in model {model_name}] {e}")
                traceback.print_exc()

        return out_results

    # ---------------------------
    # Model loaders (smart)
    # ---------------------------
    def _infer_num_classes_from_state(self, sd):
        keys = [
            "roi_heads.box_predictor.cls_score.weight",
            "roi_heads.box_predictor.cls_score.weight_orig",
            "box_predictor.cls_score.weight"
        ]
        for k in keys:
            if k in sd:
                w = sd[k]
                try:
                    return int(w.shape[0])
                except:
                    pass
        return None

    def _infer_num_keypoints_from_state(self, sd):
        keys = [
            "roi_heads.keypoint_predictor.kps_score.weight",
            "roi_heads.keypoint_predictor.kps_score_lowres.weight",
            "keypoint_predictor.kps_score.weight"
        ]
        for k in keys:
            if k in sd:
                w = sd[k]
                try:
                    if len(w.shape) >= 2:
                        return int(w.shape[1])
                except:
                    pass
        return None

    def _load_keypoint_rcnn_auto(self, path):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        print("[_load_keypoint_rcnn_auto] state_dict keys sample:", list(sd.keys())[:8])

        inferred_classes = self._infer_num_classes_from_state(sd)
        inferred_kps = self._infer_num_keypoints_from_state(sd)
        print(f"[_load_keypoint_rcnn_auto] inferred num_classes={inferred_classes}, num_keypoints={inferred_kps}")

        if inferred_classes is None:
            num_classes = 2
        else:
            num_classes = inferred_classes

        num_keypoints = inferred_kps if (inferred_kps and inferred_kps > 0) else len(KEYPOINT_NAMES)

        print(f"[_load_keypoint_rcnn_auto] building KeypointRCNN num_classes={num_classes} num_keypoints={num_keypoints}")
        model = torchvision.models.detection.keypointrcnn_resnet50_fpn(
            weights=None, num_classes=num_classes, num_keypoints=num_keypoints
        )
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[_load_keypoint_rcnn_auto] load_state_dict missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
        return model

    def _load_detection_auto(self, path):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
        inferred_classes = self._infer_num_classes_from_state(sd)
        print(f"[_load_detection_auto] inferred num_classes={inferred_classes}")
        num_classes = inferred_classes if inferred_classes is not None else 2
        print(f"[_load_detection_auto] building FasterRCNN with num_classes={num_classes}")
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[_load_detection_auto] load_state_dict missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
        return model

    # ---------------------------
    # Prediction helpers (safe)
    # ---------------------------
    def _predict_yolo_safe(self, model, image_bgr, conf):
        try:
            # FORCE DEVICE HERE: نمرر الجهاز المحدد (cuda) لإجبار الموديل على استخدامه
            res = model.predict(image_bgr, conf=conf, verbose=False, device=self.device)[0]
        except Exception as e:
            print(f"[_predict_yolo_safe] YOLO predict failed: {e}")
            traceback.print_exc()
            return []

        dets = []
        for i, box in enumerate(res.boxes):
            try:
                cls_id = int(box.cls[0])
                cls_name = res.names.get(cls_id, str(cls_id)) if hasattr(res, "names") else str(cls_id)
                conf_score = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                xyxy = box.xyxy[0].detach().cpu().numpy().astype(int) if isinstance(box.xyxy[0], torch.Tensor) else np.array(box.xyxy[0]).astype(int)
                dets.append({
                    "id": i + 1,
                    "class_name": cls_name,
                    "confidence": conf_score,
                    "box": xyxy,
                    "keypoints": None
                })
            except Exception as e:
                print(f"[_predict_yolo_safe] per-box error: {e}")
                traceback.print_exc()
        return dets

    def _predict_box_model_safe(self, model, image_bgr, conf):
        # Move inputs to DEVICE explicitly
        img_t = F.to_tensor(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = model(img_t)
        if not out:
            return []
        out0 = out[0]

        scores = out0.get("scores", torch.tensor([])).detach().cpu().numpy()
        boxes = out0.get("boxes", torch.tensor([])).detach().cpu().numpy().astype(int)
        labels = out0.get("labels", torch.tensor([])).detach().cpu().numpy().astype(int) if "labels" in out0 else np.zeros(len(scores), dtype=int)

        dets = []
        for i in range(len(scores)):
            try:
                if scores[i] < conf:
                    continue
                cls = int(labels[i]) if len(labels) > i else 1

                # --- use per-model mapping if present ---
                # default mapping set in __init__ was: {1: "bolt-loose", 2: "bolt-tight"}
                mm = self.model_class_map.get("faster rcnn", None) or {}
                # if running other detection model, try its mapping
                mm_model_specific = self.model_class_map.get(model.__class__.__name__, None)
                if mm_model_specific:
                    mm = mm_model_specific

                # resolve class name using map, fallback to simple rule
                cls_name = mm.get(cls) if mm and cls in mm else ("bolt-loose" if cls == 1 else "bolt-tight")

                dets.append({
                    "id": len(dets) + 1,
                    "class_name": cls_name,
                    "confidence": float(scores[i]),
                    "box": boxes[i],
                    "keypoints": None
                })
            except Exception as e:
                print(f"[_predict_box_model_safe] per-detection error: {e}")
                traceback.print_exc()
        return dets

    def _predict_keypoint_safe(self, model, image_bgr, conf):
        # Move inputs to DEVICE explicitly
        img_t = F.to_tensor(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = model(img_t)
        if not out:
            return []
        out0 = out[0]

        scores = out0.get("scores", torch.tensor([])).detach().cpu().numpy()
        boxes = out0.get("boxes", torch.tensor([])).detach().cpu().numpy().astype(int)
        kps_tensor = out0.get("keypoints", None)
        if isinstance(kps_tensor, torch.Tensor):
            kps_all = kps_tensor.detach().cpu().numpy()
        else:
            kps_all = np.empty((0, 0, 3))

        dets = []
        for i in range(len(scores)):
            try:
                if scores[i] < conf:
                    continue
                box_i = boxes[i]
                kps_i = kps_all[i] if (kps_all.shape[0] > i) else np.empty((0, 3))
                dets.append({
                    "id": len(dets) + 1,
                    "class_name": "bolt",
                    "confidence": float(scores[i]),
                    "box": box_i,
                    "keypoints": kps_i
                })
            except Exception as e:
                print(f"[_predict_keypoint_safe] per-detection error: {e}")
                traceback.print_exc()
        return dets

    # ---------------------------
    # Drawing / formatting helpers
    # ---------------------------
    def _draw_box_results(self, img, dets):
        res_list = []
        for d in dets:
            x1, y1, x2, y2 = map(int, d["box"])
            color_box = (255, 128, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color_box, 2)

            if "tight" in d["class_name"].lower():
                status = "TIGHT"
                color = (0, 255, 0)
            else:
                status = "LOOSE"
                color = (0, 0, 255)

            try:
                cv2.putText(img, f"{d['class_name']} {d['confidence']:.2f}", (x1, max(0, y1 - 8)),
                            FONT, 0.6, color, 2, cv2.LINE_AA)
                cv2.putText(img, status, (x1, y2 + 22), FONT, 0.7, color, 2, cv2.LINE_AA)
            except Exception:
                pass

            res_list.append({
                "id": d["id"],
                "class_name": d["class_name"],
                "status": status,
                "confidence": d["confidence"]
            })
        return img, res_list

    def _draw_keypoint_results(self, img, dets):
        res_list = []
        for d in dets:
            x1, y1, x2, y2 = map(int, d["box"])
            kps = d.get("keypoints", np.empty((0, 3)))

            tight = self._is_tight(kps)
            status = "TIGHT" if tight else "LOOSE"
            color = (0, 255, 0) if tight else (0, 0, 255)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"bolt {d['confidence']:.2f}", (x1, max(0, y1 - 8)), FONT, 0.6, color, 2)
            cv2.putText(img, status, (x1, y2 + 22), FONT, 0.7, color, 2)

            try:
                for i, (px, py, pv) in enumerate(kps):
                    if pv > 0:
                        cv2.circle(img, (int(px), int(py)), POINT_RADIUS // 1, color, -1)
                        if i < len(KEYPOINT_NAMES):
                            cv2.putText(img, KEYPOINT_NAMES[i], (int(px) + 5, int(py) + 5), FONT, 0.45, (255, 255, 255), 1)
            except Exception as e:
                print(f"[_draw_keypoint_results] draw kp error: {e}")
                traceback.print_exc()

            res_list.append({
                "id": d["id"],
                "class_name": d["class_name"],
                "status": status,
                "confidence": d["confidence"]
            })
        return img, res_list

    def _is_tight(self, kps):
        try:
            if kps is None:
                return False
            kps = np.asarray(kps)
            if kps.size == 0:
                return False
            vis = (kps[:, 2] > 0).sum() if kps.ndim == 2 and kps.shape[1] >= 3 else 0
            return vis >= 6
        except Exception:
            return False

    # ---------------------------
    # Info images
    # ---------------------------
    def _info_image_box(self, img, res, model_name):
        h, w = img.shape[:2]
        footer = max(70, len(res) * 30 + 20)
        canvas = np.full((h + footer, w, 3), 35, dtype=np.uint8)
        canvas[:h, :w] = img

        cv2.putText(canvas, f"{model_name} | {len(res)} detections", (10, h + 28), FONT, FONT_SCALE_FOOTER, (255, 255, 255), 2)
        y = h + 56
        for r in res:
            color = (0, 255, 0) if r["status"] == "TIGHT" else (0, 0, 255)
            try:
                cv2.putText(canvas, f"ID {r['id']} | {r['class_name']} | {r['status']}", (10, y), FONT, 0.6, color, 1)
            except Exception:
                pass
            y += 24
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def _info_image_keypoint(self, img, res, model_name):
        h, w = img.shape[:2]
        footer = max(70, len(res) * 30 + 20)
        canvas = np.full((h + footer, w, 3), 35, dtype=np.uint8)
        canvas[:h, :w] = img

        cv2.putText(canvas, f"{model_name} | {len(res)} detections", (10, h + 28), FONT, FONT_SCALE_FOOTER, (255, 255, 255), 2)
        y = h + 56
        for r in res:
            color = (0, 255, 0) if r["status"] == "TIGHT" else (0, 0, 255)
            try:
                cv2.putText(canvas, f"ID {r['id']} | bolt | {r['status']}", (10, y), FONT, 0.6, color, 1)
            except Exception:
                pass
            y += 24
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def _status_color(self, status):
        if status == "TIGHT":
            return "<span style='color:green;font-weight:bold'>TIGHT</span>"
        if status == "LOOSE":
            return "<span style='color:red;font-weight:bold'>LOOSE</span>"
        return "<span style='color:gray'>NO DETECTIONS</span>"