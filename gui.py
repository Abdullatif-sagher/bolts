import gradio as gr
import numpy as np
from detector import ModelManager
import config

# ---------------------------
# 1️ Lazy-loading for models
# ---------------------------

model_manager = None

def get_model_manager():
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(config.MODEL_PATHS)
    return model_manager



# ---------------------------
# 2️ Image analysis function
# ---------------------------

def analyze_image_for_gradio(input_image):
    """
    Run inference on the uploaded image using all configured models.

    Args:
        input_image (np.ndarray): The input image in RGB (HxWx3).
    
    Returns:
        list[np.ndarray]: A list of result images (one for each model).
    """

    if input_image is None:
        # If no image provided, return empty slots
        return [None] * len(config.MODEL_PATHS)
    
    manager = get_model_manager()
  
    results = manager.analyze_image(input_image, config.CONFIDENCE_THRESHOLD)
    
    # Ensure result length matches the number of models
    if len(results) != len(config.MODEL_PATHS):
        
        return [None] * len(config.MODEL_PATHS)
    return results

# ---------------------------
# 3️ Custom CSS & Theme
# ---------------------------

custom_css = """
.results-table { border-collapse: collapse; width: 100%; }
.annotated-image img { border-radius: 8px; max-width: 100%; }
.analyze-btn { background-color: #41B6A5; color: white; font-weight: bold; border-radius: 8px; padding: 10px 20px; }
.result-card { padding: 10px; border: 1px solid #ddd; border-radius: 10px; background-color: #fafafa; margin: 5px; text-align: center; }
"""
# Base theme + custom fonts
theme = gr.themes.Base()
theme.font = gr.themes.GoogleFont("Inter")
theme.font_mono = gr.themes.GoogleFont("JetBrains Mono")

# ---------------------------
# 4️ Build Gradio interface
# ---------------------------

def create_interface():
    with gr.Blocks(theme=theme, css=custom_css, title="Advanced Model Comparator") as demo:
        gr.Markdown("# 🛠 Advanced Model Comparator")

        with gr.Row():
            input_image = gr.Image(type="numpy", label="Upload Your Image Here", sources=["upload"], height=400, width=400)

        analyze_button = gr.Button("Analyze Image", variant="primary", elem_classes="analyze-btn")
        
        outputs = []
        with gr.Row():
            for name in config.MODEL_PATHS.keys():
                with gr.Column(elem_classes="result-card"):
                    gr.Markdown(f"### {name} Result")
                    img_output = gr.Image(label="Result with Details", interactive=False)
                    outputs.append(img_output)

        # Analyze button
        analyze_button.click(
            fn=analyze_image_for_gradio,
            inputs=input_image,
            outputs=outputs,
            api_name="analyze"
        )
    return demo

# ---------------------------
# 5️ Launch the app
# ---------------------------

demo = create_interface()
