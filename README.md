# 🃏 Blackjack Card Counter & Strategy Bot

An advanced, real-time Blackjack card counting and optimal strategy engine. This application uses computer vision (YOLOv8) to read live casino streams, maintains a precise count of the shoe using the Hi-Lo system, and employs a Depth-First Search (DFS) algorithm to calculate the mathematically optimal move (Expected Value) for every hand.

## 🌟 Key Features

*   **Real-Time Computer Vision:** Captures live gameplay using high-speed screen grabbing and detects cards using a custom-trained YOLOv8 object detection model.
*   **Dynamic Strategy Engine (DFS):** Calculates exact Expected Value (EV) for Hit, Stand, Double, Split, and Surrender by projecting the dealer's final hand distribution based on the *exact remaining composition* of the shoe.
*   **Advanced Card Counting:** Implements the Hi-Lo system, tracking Running Count, True Count, and specific deck penetration.
*   **Deviation Support:** Automatically cross-references optimal plays with the **Illustrious 18** and **Fab 4** deviation matrices.
*   **Bankroll Management:** Calculates dynamically optimal bet sizing using the fractional Kelly Criterion based on the current mathematical edge.
*   **Robust Table Tracking:** Uses a "Leaky Bucket" debouncing algorithm to filter out optical flickering and dynamically clusters cards into distinct player hands (supporting splits).
*   **Configurable Ruleset:** Supports S17/H17, DAS (Double After Split), European No Hole Card (ENHC) vs. US Peek, and varying blackjack payouts (3:2 vs 6:5).
*   **Session Logging:** Automatically logs all decisions, true counts, and expected values to a CSV file for post-session analysis.

## 🛠️ Technology Stack

*   **Python 3.10+** - Core application logic.
*   **Ultralytics (YOLOv8)** - Deep learning framework for real-time playing card object detection.
*   **OpenCV (`cv2`)** - Image processing, geometric polygon zones, and rendering the graphical Heads-Up Display (HUD).
*   **MSS** - Ultra-fast, cross-platform screen capture.
*   **Tkinter** - Graphical User Interface (GUI) for the pre-session configuration menu.
*   **Pytest** - Comprehensive test suite for mathematical verification and edge-case handling.

## ⚙️ How It Works (Architecture)

1.  **ScreenReader:** Captures the defined screen area, passes the frame to YOLOv8, and extracts bounding boxes for detected cards. Renders the final HUD overlay.
2.  **TableManager:** Acts as the spatial and temporal gatekeeper. Groups bounding boxes into dealer and player zones, debounces flickering detections, and identifies hand splits based on geometric gaps.
3.  **Counter:** The Single Source of Truth (SSOT) for the shoe. Updates the global Hi-Lo count and maintains the exact tally of every card rank remaining.
4.  **StrategyEngine:** When the table state stabilizes, this engine runs a recursive DFS simulation of the dealer's hidden cards to build a probability distribution, then calculates the exact EV for all legal player actions.

## 🚀 Installation

1. Clone the repository:
   ```bash
   git clone Repository_Name
   cd Black-Jack-Card-Counter
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your trained YOLOv8 model (playing_cards.pt) is located in the src/ directory.

## 🎮 Usage Guide

1. Start the Bot:
   Run the main script from your terminal:

   ```bash
   python src/ScreenReader.py
   ```

2. Configure the Setup Menu:
   A Tkinter window will appear. Fill in your screen capture coordinates and table rules (see Setup Menu Explained below). Click "Start Bot".

3. Align the Zones:
   The live HUD will appear. Press v to toggle the visibility of the detection polygons. Use your mouse to drag the vertices of the Blue polygon over the Dealer's cards, and the Orange polygon over your seating area.

4. Play:
   The bot will automatically detect cards, update the count, and recommend actions (Hit, Stand, Double, Split, Surrender) along with the optimal bet size.

# ⌨️ Hotkeys

* q - Quit the application.

* v - Toggle visibility of the zone adjustment polygons.

* r - Manually reset the shoe (resets counts and table state to 0).

* s - Save a screenshot of the current frame for debugging or annotation.

## 🎛️ Setup Menu Explained
The pre-session configuration menu ensures the engine's math perfectly matches the casino's rules:

- Screen Capture Area: Define the Top, Left, Width, and Height (in pixels) of the area you want the bot to analyze.

- Bankroll & Kelly: Enter your total bankroll and table limits. The bot uses the Kelly Criterion to suggest bet sizes relative to your edge and risk tolerance (Quarter, Half, or Full Kelly).

- Number of Decks: Total decks in a fresh shoe (usually 6 or 8).

- Deck Penetration: The percentage of the shoe dealt before the dealer shuffles (e.g., 75%). The bot uses this to automatically reset the count.

- Blackjack Payout: 3:2 (standard) or 6:5 (reduces player edge).

- Dealer Stands on Soft 17 (S17): Check if the dealer stands on a soft 17. Uncheck if they Hit (H17).

- Double After Split (DAS): Check if you are allowed to double down after splitting a pair.

- Dealer Peeks (US): Check if the dealer checks for a natural blackjack before players act. Uncheck for European rules (ENHC).

- Surrender Rule: Select Late, Early, or None based on casino offerings.

## 🧪 Running Tests

The mathematical integrity of the StrategyEngine, Counter, and TableManager is verified via a comprehensive test suite. To run the tests:

```bash
cd src
pytest test_engine.py -v
```

## 🧠 Training Your Own Model

Two helper scripts support retraining the YOLOv8 card-detection model:

*   **`auto_annotate.py`** - Uses an existing model to auto-label new screenshots, speeding up dataset creation for new table layouts or card styles.
*   **`train_model.py`** - Trains/fine-tunes a YOLOv8 model against the dataset defined in `dataset/data.yaml`, and saves the resulting weights for use by `ScreenReader.py`.

## ⚠️ Disclaimer

This project is intended strictly for educational and research purposes in the fields of computer vision, probability, and software engineering. The use of automated assistance tools may violate the Terms of Service of online casinos or live gaming establishments. The developers assume no liability for financial losses or account bans resulting from the use of this software.
