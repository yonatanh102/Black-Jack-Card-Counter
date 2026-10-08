import mss
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

class ScreenReader:
    def __init__(self):
        # Define the screen capture region (Bounding Box).
        self.monitor = {
            "top": 400,
            "left": 400,
            "width": 1150,
            "height": 550
        }
        
        # Updated to MSS() to resolve deprecation warning
        self.sct = mss.MSS()
        
        print("Loading YOLO model...")
        
        # Dynamically build the absolute path to the model file
        # This ensures the model is found regardless of where the terminal is running from
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "playing_cards.pt")
        
        self.model = YOLO(model_path)
        print("Model loaded successfully.")

    def parse_card_name(self, yolo_class_name):
        """
        Converts YOLO dataset classes (e.g., '10c', 'ah') into 
        StrategyEngine compatible ranks (e.g., '10', 'A').
        Ignores non-card classes like 'discard-tray'.
        """
        # Ignore table elements detected by the model
        ignore_list = ['freeseat', 'shoe']
        if yolo_class_name.lower() in ignore_list:
            return None

        # Handle '10' specifically since it's the only 2-character number
        if yolo_class_name.startswith('10'):
            return '10'

        # For all other cards (2-9, j, q, k, a), the rank is the first character
        rank = yolo_class_name[0].upper()
        valid_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'J', 'Q', 'K', 'A']
        
        if rank in valid_ranks:
            return rank
            
        return None

    def start_stream(self):
        print("Starting screen capture. Press 'q' in the video window to quit, 's' to save snapshot.")
        
        while True:
            start_time = time.time()

            sct_img = self.sct.grab(self.monitor)
            img_bgr = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

            # Run YOLO inference with higher resolution to preserve small card details
            results = self.model(img_bgr, imgsz=1280, verbose=False)[0]

            detected_cards = []
            
            for box in results.boxes:
                confidence = float(box.conf[0])
                
                # Lowered confidence threshold to 5% to catch any hesitation from the model
                if confidence > 0.05:
                    class_id = int(box.cls[0])
                    raw_card_name = self.model.names[class_id]

                    # Filter console output - ignore trays and slots
                    ignore_list = ['freeseat', 'shoe']
                    if raw_card_name.lower() not in ignore_list:
                        print(f"[DEBUG] Potential card sees: {raw_card_name} - {confidence*100:.1f}%")

                    parsed_rank = self.parse_card_name(raw_card_name)
                    
                    # Draw bounding box only for valid cards with confidence > 65%
                    if parsed_rank and confidence > 0.65:
                        detected_cards.append(parsed_rank)
                        
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Draw a thin green bounding box (thickness 2)
                        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw small, clean text above the bounding box
                        label = f"{parsed_rank} ({int(confidence*100)}%)"
                        cv2.putText(img_bgr, label, (x1, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Display FPS and detected cards on the top status bar
            fps = int(1.0 / (time.time() - start_time))
            display_text = f"FPS: {fps} | Detected: {', '.join(detected_cards)}"
            
            # Semi-transparent black background for the top text to ensure readability
            cv2.rectangle(img_bgr, (5, 5), (600, 40), (0, 0, 0), -1)
            cv2.putText(img_bgr, display_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Blackjack Vision - Live', img_bgr)

            # Listen for 'q' to quit or 's' to save a snapshot
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_{timestamp}.png"
                cv2.imwrite(filename, img_bgr)
                print(f"[DEBUG] Snapshot saved as '{filename}'.")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    reader = ScreenReader()
    reader.start_stream()