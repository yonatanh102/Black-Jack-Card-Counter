import csv
import time

class SessionLogger:
    """
    Handles logging of all strategic decisions, counts, and expected values 
    made during the session into a local CSV file.
    """
    def __init__(self, filename_prefix="blackjack_session"):
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.filename = f"{filename_prefix}_{timestamp}.csv"
        self.enabled = True
        
        try:
            with open(self.filename, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp", "Dealer_Upcard", "Player_Hand", 
                    "Running_Count", "True_Count", "Recommended_Action", "EV", "Deviation_Note"
                ])
            print(f"[LOGGER] Session log created: {self.filename}")
        except Exception as e:
            print(f"[LOGGER] WARNING: Failed to create log file (file might be locked/open). Logging disabled. Error: {e}")
            self.enabled = False
                
    def log_decision(self, dealer_up, hand, running_count, true_count, action, ev, note=""):
        """
        Appends a single hand decision to the CSV log.
        Safely disables itself if the file becomes locked or inaccessible.
        """
        if not self.enabled:
            return
            
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        try:
            with open(self.filename, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp, dealer_up, str(hand), 
                    running_count, true_count, action, f"{ev:.2f}", note
                ])
        except Exception as e:
            print(f"[LOGGER] ERROR: Cannot write to CSV (disabling logger for this session): {e}")
            self.enabled = False