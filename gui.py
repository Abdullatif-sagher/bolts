import gradio as gr
from detector import ModelManager
import config

# --- Custom theme to change the font ---
theme = gr.themes.Base()
theme.font = gr.themes.GoogleFont("Inter")
theme.font_mono = gr.themes.GoogleFont("JetBrains Mono")

# --- Load the model manager ---
model_manager = ModelManager(config.MODEL_PATHS)

# --- The main analysis function that Gradio will call ---
def analyze_image_for_gradio(input_image):
    if input_image is None:
        return [None] * len(config.MODEL_PATHS)
    
    # The manager now returns a list of final images
    return model_manager.analyze_image(input_image, config.CONFIDENCE_THRESHOLD)

# --- Build the Gradio Interface ---
def create_interface():
    with gr.Blocks(theme=theme, title="Model Comparator") as demo:
        gr.Markdown("# YOLO & Faster R-CNN Model Comparator")
        
        with gr.Row():
            input_image = gr.Image(type="numpy", label="Upload Your Image Here", sources=["upload"], height=400, width=400)
        
        analyze_button = gr.Button("Analyze Image", variant="primary")
        
        # The outputs are now just a list of images
        outputs = []
        with gr.Row():
            for name in config.MODEL_PATHS.keys():
                with gr.Column():
                    gr.Markdown(f"### {name} Result")
                    # Each model only has one output component: the composite image
                    image_output = gr.Image(label="Result with Details", interactive=False)
                    outputs.append(image_output)

        analyze_button.click(
            fn=analyze_image_for_gradio,
            inputs=input_image,
            outputs=outputs
        )
    return demo

# Create the interface object
demo = create_interface()