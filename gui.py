import gradio as gr
import numpy as np
import cv2
import os
import time
import pandas as pd
import tempfile
import shutil
from detector import ModelManager
import config

model_manager = None

def get_model_manager():
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(config.MODEL_PATHS)
    return model_manager

def parse_yolo_label(file_path, img_width, img_height):
    boxes = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(parts[0])
                    x_c = float(parts[1]) * img_width
                    y_c = float(parts[2]) * img_height
                    w = float(parts[3]) * img_width
                    h = float(parts[4]) * img_height
                    x1 = int(x_c - w/2)
                    y1 = int(y_c - h/2)
                    x2 = int(x_c + w/2)
                    y2 = int(y_c + h/2)
                    boxes.append({"class": cls, "box": [x1, y1, x2, y2]})
    except Exception as e:
        print(f"Error reading label {file_path}: {e}")
    return boxes

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0: return 0
    return inter_area / union_area

def evaluate_frame(pred_boxes, gt_boxes, iou_thresh=0.5):
    tp = 0
    fp = 0
    fn = 0
    
    matched_gt = set()
    
    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(gt_boxes):
            iou = compute_iou(pred['box'], gt['box'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_thresh and best_gt_idx not in matched_gt:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
            
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn

def analyze_batch_images(image_files, label_files, selected_models):
    if not image_files:
        return pd.DataFrame(), pd.DataFrame(), [], None

    manager = get_model_manager()
    table_data = []
    gallery_images = []
    
   
    output_dir = tempfile.mkdtemp()
    
    label_map = {}
    if label_files:
        for lf in label_files:
            base = os.path.splitext(os.path.basename(lf.name))[0]
            label_map[base] = lf.name

    model_metrics = {m: {"tp": 0, "fp": 0, "fn": 0, "time": 0.0, "count": 0} for m in config.MODEL_PATHS}

    for file_obj in image_files:
        file_path = file_obj.name
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]
        
        try:
            original_img = cv2.imread(file_path)
            if original_img is None: continue
            
            h, w = original_img.shape[:2]
            rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            gt_boxes = []
            if base_name in label_map:
                gt_boxes = parse_yolo_label(label_map[base_name], w, h)
            
            for model in selected_models:
                start_time = time.time()
                results = manager.analyze_image(rgb_img, config.CONFIDENCE_THRESHOLD, [model])
                end_time = time.time()
                elapsed = end_time - start_time
                
                if not results: continue
                res = results[0]
                
                tp, fp, fn = 0, 0, 0
                has_gt = (base_name in label_map)
                
                if has_gt:
                    tp, fp, fn = evaluate_frame(res['detections'], gt_boxes, config.IOU_THRESHOLD)
                    model_metrics[model]["tp"] += tp
                    model_metrics[model]["fp"] += fp
                    model_metrics[model]["fn"] += fn
                
                model_metrics[model]["time"] += elapsed
                model_metrics[model]["count"] += 1
                
                table_data.append([
                    file_name,
                    model,
                    f"{elapsed:.3f}s",
                    res["status_html"],
                    f"TP:{tp} FP:{fp} FN:{fn}" if has_gt else "No Label"
                ])
                
              
                save_name = f"{base_name}_{model.replace(' ', '_')}_result.jpg"
                save_path = os.path.join(output_dir, save_name)
         
                cv2.imwrite(save_path, cv2.cvtColor(res["image"], cv2.COLOR_RGB2BGR))
                
                label_str = f"{file_name} | {model} | {res['status_text']}"
              
                gallery_images.append((save_path, label_str))
                
        except Exception as e:
            print(f"Failed {file_name}: {e}")

    metrics_summary = []
    for m, data in model_metrics.items():
        if m not in selected_models: continue
        tp, fp, fn = data["tp"], data["fp"], data["fn"]
        total_time = data["time"]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        map_val = precision 
        
        metrics_summary.append([
            m,
            f"{total_time:.2f}s",
            tp, fp, fn,
            f"{precision:.2%}",
            f"{recall:.2%}",
            f"{map_val:.2%}"
        ])

    df_main = pd.DataFrame(table_data, columns=["Image", "Model", "Time", "Status", "Conf Matrix (TP/FP/FN)"])
    df_metrics = pd.DataFrame(metrics_summary, columns=["Model", "Total Time", "TP", "FP", "FN", "Precision", "Recall", "mAP@50 (Approx)"])
    
 
    zip_path = shutil.make_archive(os.path.join(output_dir, "results"), 'zip', output_dir)
    
    return df_main, df_metrics, gallery_images, zip_path

custom_css = """
.results-table { border-collapse: collapse; width: 100%; }
.analyze-btn { background-color: #41B6A5; color: white; font-weight: bold; border-radius: 8px; padding: 10px 20px; }
"""

theme = gr.themes.Base()
theme.font = gr.themes.GoogleFont("Inter")

def create_interface():
    with gr.Blocks(theme=theme, css=custom_css, title="Batch Comparator") as demo:
        gr.Markdown("# 🛠Bolts Detection")

        with gr.Row():
            with gr.Column(scale=1):
                file_uploader = gr.File(file_count="multiple", file_types=["image"], label="1. Upload Images")
                label_uploader = gr.File(file_count="multiple", file_types=[".txt"], label="2. Upload Labels (YOLO format) [Optional]")
            
            with gr.Column(scale=1):
                model_selector = gr.CheckboxGroup(choices=list(config.MODEL_PATHS.keys()), value=list(config.MODEL_PATHS.keys()), label="3. Select Models")
                analyze_button = gr.Button("Run Evaluation", variant="primary", elem_classes="analyze-btn")

        gr.Markdown("### 📊 Model Performance Summary")
        metrics_table = gr.Dataframe(headers=["Model", "Total Time", "TP", "FP", "FN", "Precision", "Recall", "mAP@50"], datatype=["str", "str", "number", "number", "number", "str", "str", "str"])

        gr.Markdown("### 📝 Detailed Report per Image")
        results_table = gr.Dataframe(
            headers=["Image", "Model", "Time", "Status", "Conf Matrix (TP/FP/FN)"],
            datatype=["str", "str", "str", "markdown", "str"],
            interactive=False,
            wrap=True
        )

        gr.Markdown("### 📥 Download Results")
        zip_output = gr.File(label="Download All Results (ZIP)")

        gr.Markdown("### 🖼 Visual Gallery (Click download icon on image for JPG)")
        image_gallery = gr.Gallery(label="Processed Images", show_label=True, columns=3, height="auto", object_fit="contain")

        analyze_button.click(
            fn=analyze_batch_images,
            inputs=[file_uploader, label_uploader, model_selector],
            outputs=[results_table, metrics_table, image_gallery, zip_output]
        )

    return demo

demo = create_interface()