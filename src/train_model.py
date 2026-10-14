import os
from ultralytics import YOLO

def main():
    """
    Initializes and runs the training process for the YOLOv8 object detection model.
    Optimized for NVIDIA RTX 4070 Ti SUPER (16GB VRAM, Ada Lovelace).
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    yaml_path = os.path.join(project_root, "dataset", "data.yaml")

    print(f"Targeting dataset at: {yaml_path}")

    if not os.path.exists(yaml_path):
        print(f"❌ Error: Could not find data.yaml at {yaml_path}")
        return

    print("✅ data.yaml found! Starting hardware-optimized training...")
    
    model = YOLO("yolov8n.pt") 
    
    results = model.train(
        data=yaml_path, 
        epochs=50, 
        imgsz=1280, 
        
        # --- Hardware Optimizations ---
        batch=-1,         # AutoBatch: Dynamically finds the optimal batch size for 16GB VRAM
        workers=8,        # Increases CPU-to-GPU data loading threads (adjust based on CPU cores)
        cache=True,       # Loads the entire dataset into system RAM to eliminate disk I/O bottlenecks
        device=0,         # Forces execution on the primary CUDA GPU
        amp=True          # Automatic Mixed Precision: Uses Tensor Cores (FP16) to double training speed
    )

if __name__ == "__main__":
    main()