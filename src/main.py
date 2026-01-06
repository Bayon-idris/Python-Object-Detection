from ultralytics import YOLO
from constant import yolo_model, image_dir, image_outputs_dir

from yolo import YoloPipeline


def main():
    yoloPipeline = YoloPipeline(
        model_path=yolo_model, image_dir=image_dir, output_dir=image_outputs_dir
    )
    print("Starting predictions on dataset images...")
    results = yoloPipeline.perform_predict_on_dataset()
    print("Predictions completed.")
    print(results)


if __name__ == "__main__":
    main()
