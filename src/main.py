from ultralytics import YOLO

def yolo_startup():
    model = YOLO("yolov8n.pt")
    results = model(
        "D:/Personal Research/computer-vision-project/object-detection-using-yolo-pytorch/data/JPEGImages/2007_000033.jpg",
        show=True
    )

def main():
    yolo_startup()

if __name__ == "__main__":
    main()
