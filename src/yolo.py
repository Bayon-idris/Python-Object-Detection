from ultralytics import YOLO
import os


class YoloPipeline:
    def __init__(self, model_path, image_dir, output_dir):
        self.model_path = model_path
        self.image_dir = image_dir
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        self.model = self.load_model()

    def load_model(self):
        print(f"Loading YOLO model from {self.model_path}")
        return YOLO(self.model_path)

    def predict_and_save(self, image_path):
        print(f"Running inference on: {image_path}")

        results = self.model(
            image_path,
            save=True,                
            project=self.output_dir, 
            name="predictions",
            exist_ok=True
        )

        return results

    def perform_predict_on_dataset(self):
        for img_file in sorted(os.listdir(self.image_dir)):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(self.image_dir, img_file)
            self.predict_and_save(img_path)
