from ultralytics import YOLO
import os
import tkinter as tk
from tkinter import filedialog


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

    def predict_from_upload_file_from_path(self, file_path):
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        print(f"Uploading file from path: {file_path}")
        self.predict_and_save(file_path)


    def select_image_via_dialog(self):
        root = tk.Tk()
        root.withdraw()  
        file_path = filedialog.askopenfilename(
            title="Select an image to test YOLO",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
                ("All files", "*.*")
            ]
        )

        root.destroy()
        return file_path   

    def main_dialog_tester(self):
        while True:
            path = self.select_image_via_dialog()

            if not path:
                print("No file selected. Exiting dialog tester.")
                break

            print(f"Selected image: {path}")

            self.predict_from_upload_file_from_path(path)

            again = tk.messagebox.askyesno("Continue", "Test another image?")
            if not again:
                break 