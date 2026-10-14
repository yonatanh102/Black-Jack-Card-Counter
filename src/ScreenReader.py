import mss
import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

from GameConfig import GameConfig
from Shoe import Shoe
from Counter import Counter
from StrategyEngine import StrategyEngine
from TableManager import TableManager
from SetupMenu import run_setup_menu

class ScreenReader:
    """
    The orchestrator class for the application's computer vision pipeline.
    It manages high-speed screen grabbing, invokes the YOLO object detection model, 
    delegates spatial analysis to the TableManager, and composites the graphical HUD.
    """
    def __init__(self, config_data):
        self.config_data = config_data
        
        # Define the screen bounding box targeted for capture
        self.monitor = self.config_data.get('monitor', {"top": 400, "left": 400, "width": 1150, "height": 550})
        self.sct = mss.MSS()
        
        print("Loading YOLO model...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "playing_cards.pt")
        self.model = YOLO(model_path)
        print("Model loaded successfully.")

        # Initialize the state management and strategy hierarchy
        config = GameConfig(
            num_decks=self.config_data['decks'],
            penetration=self.config_data['penetration'],
            s17=self.config_data['s17'],
            das=self.config_data['das'],
            peek=self.config_data['peek'],
            surrender=self.config_data['surrender'],
            blackjack_payout=self.config_data['payout']
        )
        shoe = Shoe(config)
        self.counter = Counter(shoe)
        self.strategy = StrategyEngine(self.counter)
        self.table_manager = TableManager(self.counter, self.strategy)
        
        # State variables for interactive UI elements (zone dragging)
        self.dragging_point = None
        self.show_zones = True 

    def mouse_events(self, event, x, y, flags, param):
        """
        OpenCV callback function to enable dynamic dragging of detection polygon vertices.
        Allows the user to align the dealer and player detection zones with the live video feed.
        """
        if not self.show_zones: 
            return 
            
        polys = [self.table_manager.dealer_polygon, self.table_manager.my_seat_polygon]
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if the user clicked near any polygon vertex
            for poly_idx, poly in enumerate(polys):
                for pt_idx, pt in enumerate(poly):
                    if np.linalg.norm(pt - np.array([x, y])) < 15:
                        self.dragging_point = (poly_idx, pt_idx)
                        return
                        
        elif event == cv2.EVENT_MOUSEMOVE:
            # Update the coordinates of the selected vertex in real-time
            if self.dragging_point is not None:
                poly_idx, pt_idx = self.dragging_point
                if poly_idx == 0:
                    self.table_manager.dealer_polygon[pt_idx] = [x, y]
                else:
                    self.table_manager.my_seat_polygon[pt_idx] = [x, y]
                    
        elif event == cv2.EVENT_LBUTTONUP:
            # Release the vertex and persist the new polygon configuration to disk
            if self.dragging_point is not None:
                self.dragging_point = None
                self.table_manager.save_zones(self.monitor['width'], self.monitor['height'])
                print("[INFO] Zones configuration saved.")

    def parse_card_name(self, yolo_class_name):
        """
        Normalizes arbitrary YOLO class labels into standardized card ranks.
        Provides a layer of abstraction between the specific training dataset format 
        and the engine's internal logic, making future model swaps seamless.
        """
        name = yolo_class_name.lower().strip()
        
        # Defensive filtering for metadata classes or known artifacts
        if any(ignore_term in name for ignore_term in ['freeseat', 'shoe', 'joker', 'background']): 
            return None
            
        mapping = {
            '2': '2', 'two': '2',
            '3': '3', 'three': '3',
            '4': '4', 'four': '4',
            '5': '5', 'five': '5',
            '6': '6', 'six': '6',
            '7': '7', 'seven': '7',
            '8': '8', 'eight': '8',
            '9': '9', 'nine': '9',
            '10': '10', 'ten': '10', 't': '10',
            'j': 'J', 'jack': 'J',
            'q': 'Q', 'queen': 'Q',
            'k': 'K', 'king': 'K',
            'a': 'A', 'ace': 'A'
        }
        
        # Check for direct matches first
        if name in mapping: 
            return mapping[name]
            
        # Fallback to prefix matching (sorted by length to prioritize longer prefixes like 'three' over 't')
        for key in sorted(mapping.keys(), key=len, reverse=True):
            if name.startswith(key):
                return mapping[key]
                
        return None

    def get_recommended_bet(self):
        """
        Calculates optimal bet sizing based on the Fractional Kelly Criterion.
        Adjusts the mathematical baseline edge based on specific table rules (H17 penalty, 6:5 payout penalty).
        """
        tc = self.counter.true_count
        bankroll = self.config_data['bankroll']
        min_bet = self.config_data['min_bet']
        max_bet = self.config_data.get('max_bet', float('inf'))
        kelly_fraction = self.config_data.get('kelly_fraction', 0.5)
        
        # Calculate standard base edge for a typical 6-deck, S17, 3:2, DAS game
        base_edge = -0.005 
        
        # Apply rule-specific penalties
        if self.config_data.get('payout') == 1.2:
            base_edge -= 0.014 
        if not self.config_data.get('s17'):
            base_edge -= 0.002 
            
        # Each true count integer adds approximately 0.5% edge to the player
        edge = base_edge + (tc * 0.005)
        
        # If the player lacks a mathematical advantage, bet the absolute minimum table limit
        if edge <= 0:
            return min_bet, edge
            
        # Kelly formula: f* = (Edge / Variance) * fraction
        variance = 1.25 # Standard variance proxy for typical blackjack distributions
        optimal_fraction = (edge / variance) * kelly_fraction
        optimal_bet = bankroll * optimal_fraction
        
        if optimal_bet < min_bet:
            return min_bet, edge
            
        # Snap the optimal bet to the nearest table increment (multiples of min_bet)
        rounded_bet = round(optimal_bet / min_bet) * min_bet
        return int(min(rounded_bet, max_bet)), edge

    def draw_hud(self, img, fps, action_result):
        """
        Composites telemetry, statistics, and strategic recommendations onto the raw image frame.
        """
        h, w = img.shape[:2]
        overlay = img.copy()

        # Render translucent background panels for readability
        cv2.rectangle(overlay, (0, 0), (w, 65), (10, 10, 10), -1)
        cv2.rectangle(overlay, (0, h - 85), (w, h), (15, 30, 15), -1)

        alpha = 0.8
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        # Render Count Telemetry
        count_text = f"RUNNING: {self.counter.running_count}   |   TRUE: {self.counter.true_count}"
        cv2.putText(img, count_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Render Betting Recommendation
        rec_bet, edge = self.get_recommended_bet()
        edge_color = (0, 255, 0) if edge > 0 else (0, 0, 255) 
        bet_text = f"REC BET: ${rec_bet}  |  EDGE: {edge*100:+.2f}%"
        cv2.putText(img, bet_text, (330, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, edge_color, 2)

        # Render System Status
        rules = f"S17: {self.config_data['s17']} | DAS: {self.config_data['das']}"
        fps_text = f"FPS: {fps} | Decks: {self.config_data['decks']} | {rules}"
        cv2.putText(img, fps_text, (w - 480, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cards_played = sum(self.counter.seen.values())
        total_cards = self.counter.shoe.total_initial_cards
        penetration = (cards_played / total_cards) * 100 if total_cards > 0 else 0

        is_stable = self.table_manager.is_vision_stable
        stab_text = "VISION: STABLE" if is_stable else "VISION: DETECTING..."
        stab_color = (0, 255, 0) if is_stable else (0, 165, 255) 

        metrics_text = f"{stab_text}  |  PENETRATION: {penetration:.1f}%"
        cv2.putText(img, metrics_text, (w - 480, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, stab_color, 2)

        # --- Render Strategy Output ---
        if action_result:
            dealer_up = action_result['dealer_upcard']
            hands_actions = action_result['hands_actions']
            
            # Display Insurance recommendation prominently if applicable
            if dealer_up == 'A' and 'insurance_recommended' in action_result:
                buy_ins = action_result['insurance_recommended']
                ins_text = " BUY INSURANCE (+EV) " if buy_ins else " DO NOT BUY INSURANCE (-EV) "
                ins_color = (0, 255, 0) if buy_ins else (0, 0, 255)
                
                cv2.rectangle(img, (w//2 - 220, h//2 - 30), (w//2 + 220, h//2 + 20), (10, 10, 10), -1)
                cv2.rectangle(img, (w//2 - 220, h//2 - 30), (w//2 + 220, h//2 + 20), ins_color, 2)
                
                text_size = cv2.getTextSize(ins_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                text_x = (w - text_size[0]) // 2
                cv2.putText(img, ins_text, (text_x, h//2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ins_color, 2)
            
            if len(hands_actions) > 1:
                cv2.putText(img, f"DEALER: {dealer_up}    --- SPLIT DETECTED ---", (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Display strategy recommendation for each detected player hand
            for i, hand_info in enumerate(hands_actions):
                rec_action = hand_info['action_data']['recommended_action']
                ev = hand_info['action_data']['expected_value']
                note = hand_info['action_data'].get('deviation_note', '') 
                my_cards = hand_info['cards']

                # Assign color coding to actions for quick visual parsing
                color = (0, 255, 0) 
                if rec_action == 'Hit': color = (0, 200, 255) 
                elif rec_action == 'Double': color = (255, 255, 0) 
                elif rec_action == 'Split': color = (255, 50, 255) 

                action_str = rec_action.upper()
                if note:
                    action_str += f"  [{note}]"

                if len(hands_actions) == 1:
                    strat_text = f"HAND: {my_cards}   VS   DEALER: {dealer_up}    =>    ACTION: {action_str}  (EV: {ev:+.2f})"
                    cv2.putText(img, strat_text, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
                else:
                    y_offset = h - 35 + (i * 25)
                    strat_text = f"HAND {i+1}: {my_cards}   =>   ACTION: {action_str}  (EV: {ev:+.2f})"
                    cv2.putText(img, strat_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(img, "WAITING FOR CARDS...", (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)

    def start_stream(self):
        """
        The primary execution loop. Continuously captures screen data, feeds it to the neural network,
        parses the bounding boxes, delegates to the TableManager, evaluates strategy, and renders the result.
        """
        print("\n--- Controls ---")
        print("'q' = Quit")
        print("'s' = Save Snapshot")
        print("'r' = Reset Shoe (Shuffle)")
        print("'v' = Toggle Zones View (Show/Hide Polygons)\n")
        
        cv2.namedWindow('Blackjack Vision - Live')
        cv2.setMouseCallback('Blackjack Vision - Live', self.mouse_events)
        
        while True:
            start_time = time.time()
            sct_img = self.sct.grab(self.monitor)
            img_bgr = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            
            # Execute YOLO inference. Agnostic NMS is disabled to prevent overlapping cards (like a face-down and face-up) from suppressing each other.
            results = self.model(img_bgr, imgsz=1280, conf=0.65, verbose=False)[0]

            detected_boxes = []
            for box in results.boxes:
                class_id = int(box.cls[0])
                parsed_rank = self.parse_card_name(self.model.names[class_id])
                
                if parsed_rank:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detected_boxes.append({'rank': parsed_rank, 'x': (x1+x2)//2, 'y': (y1+y2)//2})
                    
                    # Draw visual bounding boxes directly on the feed
                    if self.show_zones:
                        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_bgr, f"{parsed_rank}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Delegate spatial logic and strategy execution
            action_result = self.table_manager.process_frame(detected_boxes, self.monitor['width'], self.monitor['height'])
            
            # --- Auto-Shuffle Enforcement ---
            # Identifies when the dealer has reached the physical cut card and automatically resets the count.
            cards_played = sum(self.counter.seen.values())
            total_cards = self.counter.shoe.total_initial_cards
            cut_at = total_cards - self.counter.shoe.cut_card_threshold
            
            # Safely trigger auto-shuffle only when a round is not actively in progress
            if self.table_manager.round_just_ended and cards_played >= cut_at:
                print(f"\n[AUTO] Cut card threshold reached ({cards_played}/{cut_at}). Auto-resetting shoe!\n")
                self.counter.reset_count()
                self.table_manager.reset()

            fps = int(1.0 / (max(time.time() - start_time, 0.001)))

            # Render detection polygons 
            if self.show_zones:
                dealer_poly = self.table_manager.dealer_polygon
                my_poly = self.table_manager.my_seat_polygon
                
                cv2.polylines(img_bgr, [dealer_poly], True, (255, 0, 0), 2)
                cv2.putText(img_bgr, "Dealer", (dealer_poly[0][0], dealer_poly[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                for pt in dealer_poly: cv2.circle(img_bgr, tuple(pt), 6, (0, 165, 255), -1)
                    
                cv2.polylines(img_bgr, [my_poly], True, (0, 165, 255), 2)
                cv2.putText(img_bgr, "My Seat", (my_poly[0][0], my_poly[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                for pt in my_poly: cv2.circle(img_bgr, tuple(pt), 6, (0, 165, 255), -1)

            self.draw_hud(img_bgr, fps, action_result)
            cv2.imshow('Blackjack Vision - Live', img_bgr)

            # Keyboard event listeners
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')): 
                break
            elif key in (ord('s'), ord('S')): 
                cv2.imwrite(f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png", img_bgr)
            elif key in (ord('v'), ord('V')): 
                self.show_zones = not self.show_zones
            elif key in (ord('r'), ord('R')):
                self.counter.reset_count()
                self.table_manager.reset()
                print("\n[INFO] Shoe reset manually by user! Count is back to 0.\n")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    user_config = run_setup_menu()
    if user_config:
        reader = ScreenReader(user_config)
        reader.start_stream()