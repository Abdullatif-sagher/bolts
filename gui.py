import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from detector import Detector
import config

class YOLO_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title(config.WINDOW_TITLE)
        self.root.geometry(config.WINDOW_GEOMETRY)
        self.root.configure(bg=config.BG_COLOR)
        
        try:
            self.detector = Detector(model_path=config.MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model: {e}")
            self.root.quit()
            return
            
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")
        self.configure_styles()
        
        self.create_widgets()

    def configure_styles(self):
        self.style.configure("TFrame", background=config.BG_COLOR)
        self.style.configure("TLabel", background=config.BG_COLOR, foreground=config.TEXT_COLOR, font=config.FONT_NORMAL)
        self.style.configure("TButton", background=config.ACCENT_COLOR, foreground="white", font=config.FONT_BOLD, borderwidth=0)
        self.style.map("TButton", background=[('active', '#2980b9')])
        self.style.configure("TLabelframe", background=config.FRAME_COLOR, bordercolor=config.ACCENT_COLOR)
        self.style.configure("TLabelframe.Label", background=config.FRAME_COLOR, foreground=config.TEXT_COLOR, font=config.FONT_BOLD)

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.load_button = ttk.Button(main_frame, text="Load and Analyze Image", command=self.run_prediction, width=30)
        self.load_button.pack(pady=(0, 20))

        image_display_frame = ttk.Frame(main_frame)
        image_display_frame.pack(fill=tk.BOTH, expand=True)
        image_display_frame.columnconfigure(0, weight=1)
        image_display_frame.columnconfigure(1, weight=1)
        image_display_frame.rowconfigure(0, weight=1)

        self.original_frame = ttk.LabelFrame(image_display_frame, text="Original Image")
        self.original_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.original_label = ttk.Label(self.original_frame)
        self.original_label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        self.result_frame = ttk.LabelFrame(image_display_frame, text="Detection Result")
        self.result_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.result_label = ttk.Label(self.result_frame)
        self.result_label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        self.status_var = tk.StringVar(value="Status: Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def run_prediction(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image File",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if not file_path:
            return

        try:
            self.status_var.set("Status: Analyzing image...")
            self.root.update_idletasks()
            
            annotated_image, num_objects = self.detector.predict(file_path, config.CONFIDENCE_THRESHOLD)
            
            self.display_image(self.original_label, self.original_frame, file_path)
            self.display_image(self.result_label, self.result_frame, annotated_image, is_numpy_array=True)
            
            self.status_var.set(f"Status: Analysis complete. Found {num_objects} objects.")
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred during analysis: {e}")
            self.status_var.set("Status: Error")

    def display_image(self, label_widget, parent_frame, image_source, is_numpy_array=False):
        # Get the actual size of the frame that will contain the image
        frame_width = parent_frame.winfo_width()
        frame_height = parent_frame.winfo_height()

        # --- IMPROVEMENT: Add a fallback size if the widget size isn't calculated yet ---
        if frame_width <= 1: frame_width = 600
        if frame_height <= 1: frame_height = 600

        if is_numpy_array:
            img = Image.fromarray(image_source)
        else:
            img = Image.open(image_source)
        
        img_width, img_height = img.size
        
        # Prevent division by zero error for invalid images
        if img_width == 0 or img_height == 0:
            return

        # Calculate scaling factor to maintain aspect ratio
        scale_factor = min(frame_width / img_width, frame_height / img_height)
        
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)
        
        # Resize the image using the calculated dimensions
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to PhotoImage and update the label
        photo_img = ImageTk.PhotoImage(resized_img)
        label_widget.config(image=photo_img, text="")
        label_widget.image = photo_img