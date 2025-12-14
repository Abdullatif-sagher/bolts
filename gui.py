
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

# ==================== Global ====================
_manager = None

def get_manager():
    global _manager
    if _manager is None:
        _manager = ModelManager(config.MODEL_PATHS)
    return _manager

# ==================== Utils ====================
def clean(obj):
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def parse_yolo_label(path, w, h):
    boxes = []
    try:
        with open(path) as f:
            for line in f:
                _, x, y, bw, bh = map(float, line.split()[:5])
                x1 = int((x - bw / 2) * w)
                y1 = int((y - bh / 2) * h)
                x2 = int((x + bw / 2) * w)
                y2 = int((y + bh / 2) * h)
                boxes.append([x1, y1, x2, y2])
    except:
        pass
    return boxes

# ==================== Core ====================
def run(images, labels, models, kp_mode, conf, iou_th):
    if not images:
        return None, None, None, None, None, None, None

    manager = get_manager()
    out_dir = tempfile.mkdtemp()

    gallery = []
    table_rows = []
    image_level_rows = []

    label_map = (
        {os.path.splitext(f.name)[0]: f.name for f in labels}
        if labels else {}
    )

    for img_f in images:
        img_name = os.path.splitext(os.path.basename(img_f.name))[0]
        img = cv2.imread(img_f.name)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt_exists = img_name in label_map

        active_models = ["Keypoint R-CNN"] if kp_mode else models

        for model_name in active_models:
            t0 = time.time()
            results = manager.analyze_image(rgb, conf, [model_name], kp_mode)
            if not results:
                continue

            r = results[0]
            elapsed = time.time() - t0
            detections = clean(r["detections"])
            detected = len(detections) > 0
            max_score = max([d["confidence"] for d in detections], default=0.0)

            # ---------- Image-level confusion ----------
            if gt_exists and detected:
                TP, FP, FN, TN = 1, 0, 0, 0
            elif gt_exists and not detected:
                TP, FP, FN, TN = 0, 0, 1, 0
            elif not gt_exists and detected:
                TP, FP, FN, TN = 0, 1, 0, 0
            else:
                TP, FP, FN, TN = 0, 0, 0, 1

            image_level_rows.append({
                "Model": model_name,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN,
                "Score": max_score
            })

            out_path = os.path.join(out_dir, f"{img_name}_{model_name}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(r["image"], cv2.COLOR_RGB2BGR))
            gallery.append((out_path, f"{model_name} | {r['status_text']}"))

            table_rows.append({
                "Image": img_name,
                "Model": model_name,
                "Time (s)": round(elapsed, 3),
                "Objects": len(detections),
                "Status": r["status_text"]
            })

    df = pd.DataFrame(table_rows)
    img_df = pd.DataFrame(image_level_rows)
    zip_path = shutil.make_archive(os.path.join(out_dir, "results"), "zip", out_dir)


    # ==================== Safety ====================
    if img_df.empty or img_df[["TP", "FP", "FN"]].sum().sum() == 0:
        warn = go.Figure()
        warn.add_annotation(
            text="⚠️ No valid Image-level data → Metrics disabled",
            x=0.5, y=0.5, showarrow=False, font=dict(size=18)
        )
        return gallery, df, warn, warn, warn, warn, zip_path

    # ==================== Image-level Metrics ====================
    m = img_df.groupby("Model").sum()
    m["Precision"] = m.TP / (m.TP + m.FP + 1e-6)
    m["Recall"] = m.TP / (m.TP + m.FN + 1e-6)
    m["F1"] = 2 * m.Precision * m.Recall / (m.Precision + m.Recall + 1e-6)

    metrics_fig = px.bar(
        m.reset_index(),
        x="Model",
        y=["Precision", "Recall", "F1"],
        barmode="group",
        title="Image-level Model Comparison"
    )

    # ==================== ROC Curve + AUC ====================
    roc_fig = go.Figure()

    for model in img_df.Model.unique():
        sub = img_df[img_df.Model == model]
        tprs, fprs = [], []

        for th in np.linspace(0, 1, 25):
            TP = ((sub.Score >= th) & (sub.TP == 1)).sum()
            FP = ((sub.Score >= th) & (sub.FP == 1)).sum()
            FN = ((sub.Score < th) & (sub.FN == 1)).sum()
            TN = ((sub.Score < th) & (sub.TN == 1)).sum()

            TPR = TP / (TP + FN + 1e-6)
            FPR = FP / (FP + TN + 1e-6)

            tprs.append(TPR)
            fprs.append(FPR)

        auc = np.trapz(sorted(tprs), sorted(fprs))
        roc_fig.add_trace(
            go.Scatter(
                x=fprs,
                y=tprs,
                mode="lines+markers",
                name=f"{model} (AUC={auc:.3f})"
            )
        )

    roc_fig.update_layout(
        title="ROC Curve (Image-level)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    perf_fig = px.bar(df, x="Model", y="Time (s)", title="Latency per Model")

    return gallery, df, perf_fig, metrics_fig, roc_fig, zip_path

# ==================== UI ====================
theme = gr.themes.Soft(primary_hue="indigo")

css = """
#gallery_scroll {
    max-height: 600px !important;
    overflow-y: auto !important;
}
"""

with gr.Blocks(theme=theme, css=css, title="Bolt & Nut Detection Model Comparator") as demo:
    gr.Markdown("# 🔩 Bolt & Nut Detection Model Comparator\n### Image-level Analytics")

    with gr.Row():
        imgs = gr.File(file_types=["image"], file_count="multiple", label="Images")
        lbls = gr.File(file_types=[".txt"], file_count="multiple", label="Labels (YOLO)")

    with gr.Row():
        models = gr.CheckboxGroup(
            choices=list(config.MODEL_PATHS.keys()),
            value=["YOLOv8","YOLOv11","YOLOv12","faster rcnn"],
            label="Models"
        )
        kp = gr.Checkbox(label="Keypoint Mode")
        conf = gr.Slider(0.1, 1.0, 0.6, 0.05, label="Confidence")

    btn = gr.Button("🚀 Run Analysis", variant="primary")

    with gr.Tabs():
        with gr.Tab("Gallery"):
            gal = gr.Gallery(
                columns=4,
                height=600,
                object_fit="contain",
                allow_preview=True,
                elem_id="gallery_scroll"
            )
        with gr.Tab("Analytics"):
            perf_p = gr.Plot(label="Performance")
            metrics_p = gr.Plot(label="Image-level Metrics")
            roc_p = gr.Plot(label="ROC Curve + AUC")
        with gr.Tab("Table"):
            table = gr.Dataframe()
            dl = gr.DownloadButton("📦 Download Results")

    btn.click(
        run,
        inputs=[imgs, lbls, models, kp, conf, gr.State(0.5)],
        outputs=[gal, table, perf_p, metrics_p, roc_p, dl]
    )

if __name__ == "__main__":
    demo.launch()
