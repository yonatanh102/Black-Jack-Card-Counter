import os
from ultralytics import YOLO

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    yaml_path = os.path.join(project_root, "dataset", "data.yaml")

    print(f"Targeting dataset at: {yaml_path}")

    if not os.path.exists(yaml_path):
        print(f"❌ Error: Could not find data.yaml at {yaml_path}")
        print("Please make sure the Roboflow files are extracted directly into the 'dataset' folder.")
        return

    print("✅ data.yaml found! Starting training...")
    
    model = YOLO("yolov8n.pt") 
    results = model.train(data=yaml_path, epochs=50, imgsz=640)

if __name__ == "__main__":
    main()