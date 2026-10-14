import cv2
import numpy as np
import json
import os
from SessionLogger import SessionLogger

class TableManager:
    """
    Manages the spatial and temporal state of the Blackjack table.
    Handles coordinate zones, debouncing card detections, mapping cards to hands, 
    and triggering strategy evaluations.
    """
    def __init__(self, counter, strategy_engine, **kwargs):
        self.counter = counter
        self.strategy = strategy_engine
        self.zones_file = "zones_config.json"
        self.logger = SessionLogger() 
        
        # Default polygon coordinates for the dealer and player seating areas
        self.default_dealer = [[460, 200], [680, 200], [680, 280], [460, 280]]
        self.default_seat = [[470, 240], [640, 240], [660, 550], [460, 550]]
        
        self.dealer_polygon = np.array(self.default_dealer, np.int32)
        self.my_seat_polygon = np.array(self.default_seat, np.int32)
        
        # State tracking matrices for debouncing and sequence progression
        self.counted_in_round = []
        self.empty_frames = 0
        self.drop_frames = 0
        self.round_just_ended = False
        self.pending_cards = {r: 0 for r in ['2','3','4','5','6','7','8','9','10','J','Q','K','A']}
        
        self._last_raw_state = None
        self._raw_state_counter = 0
        self._stable_dealer = None
        self._stable_hands = []
        
        self._last_eval_key = None
        self._last_eval_result = None
        
        # Tracks hands already logged in the current round to prevent duplicates
        self._logged_hands_this_round = set()

        self.load_zones()

    @property
    def is_vision_stable(self):
        """
        Returns True if the current spatial arrangement of cards has remained consistent 
        across multiple consecutive frames, preventing strategy shifts from transient OCR glitches.
        """
        has_content = self._last_raw_state is not None and any(self._last_raw_state)
        return has_content and self._raw_state_counter >= 3

    def load_zones(self):
        """Loads custom coordinate zones if configured previously by the user."""
        if os.path.exists(self.zones_file):
            try:
                with open(self.zones_file, 'r') as f:
                    data = json.load(f)
                    if data['dealer'][0][0] < 2.0: 
                        self._rel_dealer = data['dealer']
                        self._rel_seat = data['seat']
                    else:
                        self.dealer_polygon = np.array(data['dealer'], np.int32)
                        self.my_seat_polygon = np.array(data['seat'], np.int32)
                print("[TABLE] Loaded custom zones from config.")
            except Exception as e:
                print(f"[TABLE] Error loading zones: {e}")

    def save_zones(self, frame_w, frame_h):
        """Persists the user-adjusted zone polygons to disk as relative coordinates."""
        rel_dealer = [[float(x)/frame_w, float(y)/frame_h] for x, y in self.dealer_polygon.tolist()]
        rel_seat = [[float(x)/frame_w, float(y)/frame_h] for x, y in self.my_seat_polygon.tolist()]
        try:
            with open(self.zones_file, 'w') as f:
                json.dump({'dealer': rel_dealer, 'seat': rel_seat}, f)
        except Exception as e:
            print(f"[TABLE] Error saving zones: {e}")

    def _update_polys_from_relative(self, w, h):
        """Scales the loaded relative coordinates dynamically against the current window size."""
        if hasattr(self, '_rel_dealer'):
            self.dealer_polygon = np.array([[int(x*w), int(y*h)] for x, y in self._rel_dealer], np.int32)
            self.my_seat_polygon = np.array([[int(x*w), int(y*h)] for x, y in self._rel_seat], np.int32)
            del self._rel_dealer
            del self._rel_seat

    def reset(self, visible_cards=None):
        """
        Wipes the local table tracking state. Triggered natively at the end of a round
        or manually by the user upon a shoe shuffle.
        Accepts visible_cards to prevent double-counting cards that remain on the table during a drop-frame reset.
        """
        self.counted_in_round = visible_cards if visible_cards is not None else []
        self.pending_cards = {r: 0 for r in ['2','3','4','5','6','7','8','9','10','J','Q','K','A']}
        self._stable_dealer = None
        self._stable_hands = []
        self._logged_hands_this_round.clear()
        self._last_raw_state = None
        self._raw_state_counter = 0
        self._last_eval_key = None
        self._last_eval_result = None
        self.round_just_ended = False

    def process_frame(self, detected_boxes, frame_w, frame_h):
        """
        The central execution loop invoked per visual frame.
        Handles geometric sorting of raw YOLO outputs, determines round lifecycles, 
        and acts as the gatekeeper before passing data to the Strategy Engine.
        """
        self._update_polys_from_relative(frame_w, frame_h)
        self.round_just_ended = False
            
        # --- Round Lifecycle Management ---
        # Detects if the table has been cleared (all cards removed)
        if not detected_boxes:
            self.empty_frames += 1
            self.drop_frames = 0
            
            # Increased threshold to 30 frames to account for lower FPS at higher resolutions
            if self.empty_frames > 30:
                if self.counted_in_round or self._stable_hands or self._logged_hands_this_round:
                    print("[TABLE] Cards cleared. Round ended.")
                    self.reset()
                    self.round_just_ended = True
            
            return self._last_eval_result
        else:
            self.empty_frames = 0
            # Identify sudden massive drops in detections (dealer clearing the cards)
            if len(self.counted_in_round) >= 3 and len(detected_boxes) <= len(self.counted_in_round) // 2:
                self.drop_frames += 1
            else:
                self.drop_frames = 0
                
            if self.drop_frames > 15:
                print("[TABLE] Sudden drop in cards detected. Round ended.")
                visible = [b['rank'] for b in detected_boxes]
                self.reset(visible_cards=visible)
                self.round_just_ended = True
                self.drop_frames = 0
                return self._last_eval_result

        # --- Geometric Sorting ---
        dealer_cards_data = [] 
        my_cards_data = [] 
        all_current_cards = []
        
        for box in detected_boxes:
            rank = box['rank']
            x, y = box['x'], box['y']
            all_current_cards.append(rank) 
            
            if cv2.pointPolygonTest(self.dealer_polygon, (x, y), False) >= 0:
                dealer_cards_data.append({'rank': rank, 'x': x})
            elif cv2.pointPolygonTest(self.my_seat_polygon, (x, y), False) >= 0:
                my_cards_data.append({'rank': rank, 'x': x})
        
        # Pass raw detections to the global counting ledger (updates True Count)
        self._update_new_cards_only(all_current_cards)
        
        # --- Local Gatekeeper ---
        relevant_ranks = [c['rank'] for c in dealer_cards_data] + [c['rank'] for c in my_cards_data]
        if any(self.counted_in_round.count(r) < relevant_ranks.count(r) for r in set(relevant_ranks)):
            return self._last_eval_result
            
        # --- Spatial Clustering (Splits) ---
        my_hands = []
        if my_cards_data:
            my_cards_data.sort(key=lambda c: c['x'])
            current_hand = [my_cards_data[0]['rank']]
            
            split_gap = frame_w * 0.05 
            
            for i in range(1, len(my_cards_data)):
                if my_cards_data[i]['x'] - my_cards_data[i-1]['x'] > split_gap:
                    my_hands.append(current_hand)
                    current_hand = [my_cards_data[i]['rank']]
                else:
                    current_hand.append(my_cards_data[i]['rank'])
            my_hands.append(current_hand)

        dealer_upcard = None
        if dealer_cards_data:
            dealer_cards_data.sort(key=lambda c: c['x'])
            dealer_upcard = dealer_cards_data[0]['rank']

        current_raw_state = (dealer_upcard, tuple(tuple(h) for h in my_hands))
        if current_raw_state == self._last_raw_state:
            self._raw_state_counter += 1
        else:
            self._raw_state_counter = 0
            self._last_raw_state = current_raw_state

        if self._raw_state_counter >= 3:
            self._stable_dealer = dealer_upcard
            self._stable_hands = my_hands

        # --- Strategy Invocation ---
        if self._stable_dealer and self._stable_hands:
            eval_key = (self._stable_dealer, tuple(tuple(h) for h in self._stable_hands), tuple(sorted(self.counter.seen.items())))
            
            if eval_key == self._last_eval_key:
                return self._last_eval_result
            
            hands_actions = []
            is_split_round = len(self._stable_hands) > 1 
            
            for i, hand in enumerate(self._stable_hands):
                if len(hand) < 2:
                    continue

                is_first = len(hand) == 2
                action_data = self.strategy.get_best_action(hand, self._stable_dealer, is_first_action=is_first, is_split_hand=is_split_round)
                hands_actions.append({'cards': hand, 'action_data': action_data})
                
                hand_signature = (self._stable_dealer, i, tuple(hand))
                if hand_signature not in self._logged_hands_this_round:
                    self.logger.log_decision(
                        dealer_up=self._stable_dealer,
                        hand=hand,
                        running_count=self.counter.running_count,
                        true_count=self.counter.true_count,
                        action=action_data['recommended_action'],
                        ev=action_data['expected_value'],
                        note=action_data.get('deviation_note', '')
                    )
                    self._logged_hands_this_round.add(hand_signature)
                
            result = {
                'dealer_upcard': self._stable_dealer,
                'hands_actions': hands_actions 
            }
            
            if self._stable_dealer == 'A':
                result['insurance_recommended'] = self.strategy.should_buy_insurance()
                
            self._last_eval_key = eval_key
            self._last_eval_result = result
            return result
            
        return None

    def _update_new_cards_only(self, all_current_cards):
        """
        Implements a 'Leaky Bucket' debounce algorithm.
        Ensures a card is consistently detected across multiple consecutive frames before 
        it is officially registered in the global game state ledger (True Count).
        """
        valid_ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        current_counts = {rank: all_current_cards.count(rank) for rank in valid_ranks}
        
        for rank in valid_ranks:
            needed_count = current_counts[rank]
            counted_count = self.counted_in_round.count(rank)
            
            if needed_count > counted_count:
                self.pending_cards[rank] = min(self.pending_cards[rank] + 1, 5)
                if self.pending_cards[rank] >= 5:
                    self.counter.update_count(rank)
                    self.counted_in_round.append(rank)
                    print(f"[COUNTER] -> Handled: {rank} | Running: {self.counter.running_count} | True: {self.counter.true_count}")
                    self.pending_cards[rank] = 0
            else:
                self.pending_cards[rank] = max(0, self.pending_cards[rank] - 1)