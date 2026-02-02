import gradio as gr
import numpy as np
import cv2
import os
import time
import pandas as pd
import tempfile
import shutil
import plotly.express as px
import plotly.graph_objects as go
from detector import ModelManager
import config
from difflib import SequenceMatcher

# ==================== CONFIGURATION ====================
YOLO_ID_MAP = {
    0: "bolt-loose",
    1: "bolt-tight",
    2: "bolt-missing"
}

# ==================== Global ====================
_manager = None

def get_manager():
    global _manager
    if _manager is None:
        _manager = ModelManager(config.MODEL_PATHS)
    return _manager

# ==================== Utils & Math ====================
def clean(obj):
    if isinstance(obj, dict): return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list): return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def normalize_class_name(name):
    name = str(name).lower().strip()
    if "loose" in name: return "bolt-loose"
    if "tight" in name: return "bolt-tight"
    if "miss" in name: return "bolt-missing"
    return name

def match_image_to_label(img_name, label_files):
    if not label_files: return None
    for lbl in label_files:
        lbl_name = os.path.splitext(os.path.basename(lbl.name))[0]
        if img_name == lbl_name or img_name in lbl_name or lbl_name in img_name:
            return lbl.name
    return None

def parse_yolo_label(path, w, h):
    targets = []
    try:
        if path is None or not os.path.exists(path): return []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cls_name = YOLO_ID_MAP.get(cls_id, str(cls_id))
                    
                    x_c = float(parts[1])
                    y_c = float(parts[2])
                    bw = float(parts[3])
                    bh = float(parts[4])
                    
                    x1 = int((x_c - bw / 2) * w)
                    y1 = int((y_c - bh / 2) * h)
                    x2 = int((x_c + bw / 2) * w)
                    y2 = int((y_c + bh / 2) * h)
                    
                    norm_name = normalize_class_name(cls_name)
                    targets.append({'class_name': norm_name, 'box': [x1, y1, x2, y2]})
    except Exception as e:
        print(f"[ERROR] Parsing label {path}: {e}")
    return targets

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# --- Standard AP Calculation ---
def calculate_ap_per_class_standard(detections, ground_truths, iou_threshold=0.5):
    if not ground_truths: return 0.0 if not detections else 0.0
    if not detections: return 0.0 
    
    npos = len(ground_truths)
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    tp = np.zeros(len(detections))
    fp = np.zeros(len(detections))
    det_gt = [False] * len(ground_truths)
    
    for i, det in enumerate(detections):
        bb = det['box']
        ovmax = -1
        jmax = -1
        for j, gt in enumerate(ground_truths):
            ov = compute_iou(bb, gt['box'])
            if ov > ovmax:
                ovmax = ov
                jmax = j
        
        if ovmax >= iou_threshold:
            if not det_gt[jmax]:
                tp[i] = 1.
                det_gt[jmax] = True
            else:
                fp[i] = 1.
        else:
            fp[i] = 1.
            
    acc_fp = np.cumsum(fp)
    acc_tp = np.cumsum(tp)
    rec = acc_tp / npos
    prec = acc_tp / (acc_tp + acc_fp + 1e-6)
    
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))

# ==================== Core Pipeline ====================
def run(images, labels, models, kp_mode, conf_th, iou_th):
    if not images: return None, None, None, None, None, None, None, None
    
    if not isinstance(images, list): images = [images]
    if labels is None: labels = []
    if not isinstance(labels, list): labels = [labels]

    manager = get_manager()
    

    base_dir = tempfile.mkdtemp()
    

    images_dir = os.path.join(base_dir, "gallery_images")
    os.makedirs(images_dir, exist_ok=True)
    

    reports_dir = os.path.join(base_dir, "Bolt_Benchmark_Reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    gallery = []
    active_models = ["Keypoint R-CNN"] if kp_mode else models
    global_metrics = {m: {} for m in active_models}
    detailed_rows = []

    print("\n--- Starting Analysis ---")

    # Processing Loop
    for img_f in images:
        img_basename = os.path.basename(img_f.name)
        img_name = os.path.splitext(img_basename)[0]
        
        img_cv2 = cv2.imread(img_f.name)
        if img_cv2 is None: continue
        h, w = img_cv2.shape[:2]
        rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        
        gt_path = match_image_to_label(img_name, labels)
        gt_targets = parse_yolo_label(gt_path, w, h)
        gt_boxes = [t['box'] for t in gt_targets]
        
        for t in gt_targets:
            c = t['class_name']
            for m in active_models:
                if c not in global_metrics[m]: global_metrics[m][c] = {'preds': [], 'gts': []}
                global_metrics[m][c]['gts'].append({'box': t['box']})

        if not gt_targets: pass

        for model_name in active_models:
            t0 = time.time()
            results = manager.analyze_image(rgb, conf_th, [model_name], kp_mode)
            elapsed = time.time() - t0
            
            preds_boxes = []
            status_text = "NO DET"
            out_img = rgb
            
            if results:
                r = results[0]
                status_text = r["status_text"]
                out_img = r["image"]
                
                for d in r["detections"]:
                    cname = normalize_class_name(d["class_name"])
                    preds_boxes.append(d['box'])
                    if cname not in global_metrics[model_name]:
                         global_metrics[model_name][cname] = {'preds': [], 'gts': []}
                    global_metrics[model_name][cname]['preds'].append({'confidence': d['confidence'], 'box': d['box']})

            # Stats
            tp, fp = 0, 0
            matched_gt_indices = set()
            for p_box in preds_boxes:
                best_iou = 0
                best_idx = -1
                for idx, g_box in enumerate(gt_boxes):
                    if idx in matched_gt_indices: continue
                    iou = compute_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                if best_iou >= iou_th and best_idx != -1:
                    tp += 1
                    matched_gt_indices.add(best_idx)
                else:
                    fp += 1
            fn = len(gt_boxes) - len(matched_gt_indices)
            tn = 1 if (len(gt_boxes) == 0 and len(preds_boxes) == 0) else 0

            # Metrics
            denom_prec = tp + fp
            denom_rec = tp + fn
            prec = tp / denom_prec if denom_prec > 0 else (1.0 if tn else 0.0)
            rec = tp / denom_rec if denom_rec > 0 else (1.0 if tn else 0.0)
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else (1.0 if tn else 0.0)

            detailed_rows.append({
                "Image": img_name,
                "Model": model_name,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Precision": round(prec, 2),
                "Recall": round(rec, 2),
                "F1": round(f1, 2),
                "Latency (s)": round(elapsed, 4)
            })
            
       
            out_path = os.path.join(images_dir, f"{img_name}_{model_name}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR))
            gallery.append((out_path, f"{model_name} | {status_text}"))

    # ==================== Aggregation & Summary ====================
    df_detailed = pd.DataFrame(detailed_rows)
    if not df_detailed.empty:
        df_detailed.sort_values(by=["Model", "Image"], inplace=True)

    summary_rows = []
    map_rows = []   
    class_rows = [] 
    
    for m in active_models:
        aps = []
        ap_dict = {} 
        
        for cls_name, data in global_metrics[m].items():
            gts_fmt = [{'box': g['box']} for g in data['gts']]
            preds_fmt = data['preds']
            
            if not gts_fmt and not preds_fmt: 
                ap = 0.0
            else:
                ap = calculate_ap_per_class_standard(preds_fmt, gts_fmt, iou_threshold=iou_th)
            
            aps.append(ap)
            ap_dict[cls_name] = ap
            class_rows.append({"Model": m, "Class": cls_name, "Value": ap})
            
        mAP = np.mean(aps) if aps else 0.0
        map_rows.append({"Model": m, "Value": mAP})

        if not df_detailed.empty:
            model_slice = df_detailed[df_detailed["Model"] == m]
            total_tp = model_slice["TP"].sum()
            total_fp = model_slice["FP"].sum()
            total_fn = model_slice["FN"].sum()
            total_tn = model_slice["TN"].sum()
            avg_lat = model_slice["Latency (s)"].mean()
        else:
            total_tp, total_fp, total_fn, total_tn, avg_lat = 0, 0, 0, 0, 0

        summary_entry = {
            "Model": m,
            "mAP": round(mAP, 3),
            "Total TP": total_tp,
            "Total FP": total_fp,
            "Total FN": total_fn,
            "Avg Latency": round(avg_lat, 4)
        }
        for cls_k, cls_v in ap_dict.items():
            summary_entry[f"AP ({cls_k})"] = round(cls_v, 3)
            
        summary_rows.append(summary_entry)

    df_summary = pd.DataFrame(summary_rows)

    # ==================== Plots ====================
    fig_map = px.bar(pd.DataFrame(map_rows), x="Model", y="Value", title="🏆 mAP (Accuracy)", text_auto='.3f', range_y=[0, 1.1], color="Model")
    if class_rows:
        fig_cls = px.bar(pd.DataFrame(class_rows), x="Class", y="Value", color="Model", barmode="group", title="🧐 Per-Class AP", text_auto='.2f', range_y=[0, 1.1])
    else:
        fig_cls = go.Figure()
    
    if not df_detailed.empty:
        sums = df_detailed.groupby("Model")[["TP", "FP", "FN", "TN"]].sum().reset_index().melt(id_vars="Model", var_name="Type", value_name="Count")
        fig_cnt = px.bar(sums, x="Model", y="Count", color="Type", title="🔢 Total Counts", barmode="group", color_discrete_map={"TP":"green", "FP":"red", "FN":"orange", "TN":"blue"})
        
        avg_metrics = df_detailed.groupby("Model")[["Precision", "Recall", "F1"]].mean().reset_index().melt(id_vars="Model", var_name="Metric", value_name="Value")
        fig_f1 = px.bar(avg_metrics, x="Model", y="Value", color="Metric", barmode="group", title="📊 Avg Image Metrics", range_y=[0, 1.1])
        
        lat_data = df_detailed.groupby("Model")["Latency (s)"].mean().reset_index()
        fig_lat = px.bar(lat_data, x="Model", y="Latency (s)", title="⚡ Latency", text_auto='.3f')
    else:
        fig_cnt, fig_f1, fig_lat = go.Figure(), go.Figure(), go.Figure()

    # ==================== Outputs & Saving (Excel Only) ====================
    
  
    excel_det_path = os.path.join(reports_dir, "detailed_report.xlsx")
    df_detailed.to_excel(excel_det_path, index=False)
    
    excel_sum_path = os.path.join(reports_dir, "summary_report.xlsx")
    df_summary.to_excel(excel_sum_path, index=False)
    
  
    zip_path = shutil.make_archive(os.path.join(base_dir, "Bolt_Benchmark_Reports"), "zip", reports_dir)
    
    return gallery, df_summary, df_detailed, fig_map, fig_cls, fig_cnt, fig_f1, fig_lat, zip_path

# ==================== UI ====================
theme = gr.themes.Soft(primary_hue="indigo")
css = "#gallery_scroll { max-height: 600px !important; overflow-y: auto !important; }"

with gr.Blocks(theme=theme, css=css, title="Bolt Benchmark Ultimate") as demo:
    gr.Markdown("# 🔩 Bolt Benchmark Ultimate\n### Full Reporting & Model Comparison")
    
    with gr.Row():
        imgs = gr.File(label="Images (jpg/png)", file_count="multiple", file_types=["image"])
        lbls = gr.File(label="Labels (.txt only)", file_count="multiple", file_types=[".txt"])
    
    with gr.Row():
        models = gr.CheckboxGroup(list(config.MODEL_PATHS.keys()), value=["YOLOv8"], label="Models")
        kp = gr.Checkbox(label="Keypoint Mode")
        
    with gr.Row():
        conf = gr.Slider(0.1, 1.0, 0.5, label="Conf Th")
        iou = gr.Slider(0.1, 0.95, 0.5, label="IoU Th")

    btn = gr.Button("🚀 Run Analysis", variant="primary")

    with gr.Tabs():
        with gr.Tab("📊 Dashboard"):
            with gr.Row():
                p1 = gr.Plot(label="mAP (Overall Accuracy)")
                p2 = gr.Plot(label="Per-Class AP")
            with gr.Row():
                p3 = gr.Plot(label="Total Counts")
                p4 = gr.Plot(label="Avg Metrics")
            p5 = gr.Plot(label="Latency")
            
        with gr.Tab("📋 Data Tables"):
            gr.Markdown("### 1. Model Summary (Comparison)")
            t_sum = gr.Dataframe(label="Model Summary")
            
            gr.Markdown("### 2. Detailed Report (Per Image)")
            t_det = gr.Dataframe(label="Detailed Stats")
            
            dl = gr.DownloadButton("📥 Download Excel Reports (ZIP)", visible=True)

        with gr.Tab("🖼️ Gallery"):
            gal = gr.Gallery(columns=4, height=600, elem_id="gallery_scroll")

    btn.click(
        run, 
        inputs=[imgs, lbls, models, kp, conf, iou], 
        outputs=[gal, t_sum, t_det, p1, p2, p3, p4, p5, dl]
    )

if __name__ == "__main__":
    demo.launch()