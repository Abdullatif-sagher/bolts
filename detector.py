from ultralytics import YOLO
import cv2

class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def predict(self, image_path, conf_threshold):
        # 1. Load the image using OpenCV
        original_image = cv2.imread(image_path)
        if original_image is None:
            raise ValueError(f"Could not read image from path: {image_path}")

   
        # 2. Convert the image to Grayscale
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        # 3. Convert the Grayscale image back to 3-channels (BGR)
        # The YOLO model expects a 3-channel input, so we duplicate the gray channel
        image_for_model = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

        # 4. Run prediction on the preprocessed image
        results = self.model.predict(image_for_model, conf=conf_threshold)
        
        # 5. Get the annotated image (this also comes from the preprocessed image)
        annotated_image_bgr = results[0].plot()
        
        # 6. Convert final image to RGB for display in the GUI
        annotated_image_rgb = cv2.cvtColor(annotated_image_bgr, cv2.COLOR_BGR2RGB)
        
        num_objects = len(results[0].boxes)
        
        return annotated_image_rgb, num_objects