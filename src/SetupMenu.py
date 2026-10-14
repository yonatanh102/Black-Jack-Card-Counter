import tkinter as tk
from tkinter import ttk, messagebox

def run_setup_menu():
    """
    Constructs and launches the initial graphical user interface (GUI) setup menu.
    This menu collects critical game parameters (rules, payouts, bankroll settings) 
    and screen capture coordinates before initializing the main bot loop.
    
    Returns:
        dict: A configuration dictionary containing all user-defined parameters.
    """
    config = {}
    
    # Initialize main window
    root = tk.Tk()
    root.title("Blackjack Bot - Configuration")
    root.geometry("380x620") 
    root.attributes('-topmost', True) # Ensures the menu spawns over other windows
    
    style = ttk.Style()
    style.theme_use('clam')
    
    ttk.Label(root, text="Table, Bankroll & Capture Setup", font=("Helvetica", 14, "bold")).pack(pady=10)

    # --- Initialize UI Variables ---
    
    # Screen Capture Coordinates
    var_top = tk.IntVar(value=400)
    var_left = tk.IntVar(value=400)
    var_width = tk.IntVar(value=1150)
    var_height = tk.IntVar(value=550)

    # Bankroll Management
    var_bankroll = tk.IntVar(value=1000)
    var_min_bet = tk.IntVar(value=5)
    var_max_bet = tk.IntVar(value=500)
    var_kelly = tk.StringVar(value="Half Kelly (2% RoR)") 

    # Table Rules
    var_decks = tk.IntVar(value=6)
    var_penetration = tk.StringVar(value="75%")
    var_payout = tk.StringVar(value="3:2")
    var_s17 = tk.BooleanVar(value=True)
    var_das = tk.BooleanVar(value=True)
    var_peek = tk.BooleanVar(value=True)
    var_surrender = tk.StringVar(value="late")
    
    # --- Build UI Layout ---
    
    frame = ttk.Frame(root)
    frame.pack(padx=20, pady=5, fill="x")

    # Capture Area Section
    ttk.Label(frame, text="Screen Capture Area (px):", foreground="purple").grid(row=0, column=0, sticky="w", pady=5)
    cap_frame = ttk.Frame(frame)
    cap_frame.grid(row=0, column=1, pady=5)
    
    ttk.Label(cap_frame, text="T:").pack(side="left")
    ttk.Entry(cap_frame, textvariable=var_top, width=4).pack(side="left", padx=2)
    ttk.Label(cap_frame, text="L:").pack(side="left")
    ttk.Entry(cap_frame, textvariable=var_left, width=4).pack(side="left", padx=2)
    ttk.Label(cap_frame, text="W:").pack(side="left")
    ttk.Entry(cap_frame, textvariable=var_width, width=4).pack(side="left", padx=2)
    ttk.Label(cap_frame, text="H:").pack(side="left")
    ttk.Entry(cap_frame, textvariable=var_height, width=4).pack(side="left", padx=2)

    ttk.Separator(frame, orient='horizontal').grid(row=1, columnspan=2, sticky="ew", pady=10)

    # Bankroll Section
    ttk.Label(frame, text="Total Bankroll ($):", foreground="green").grid(row=2, column=0, sticky="w", pady=5)
    ttk.Entry(frame, textvariable=var_bankroll, width=8).grid(row=2, column=1, pady=5)

    ttk.Label(frame, text="Table Min Bet ($):", foreground="green").grid(row=3, column=0, sticky="w", pady=5)
    ttk.Entry(frame, textvariable=var_min_bet, width=8).grid(row=3, column=1, pady=5)
    
    ttk.Label(frame, text="Table Max Bet ($):", foreground="red").grid(row=4, column=0, sticky="w", pady=5)
    ttk.Entry(frame, textvariable=var_max_bet, width=8).grid(row=4, column=1, pady=5)
    
    ttk.Label(frame, text="Betting Risk (Kelly):", foreground="blue").grid(row=5, column=0, sticky="w", pady=5)
    ttk.Combobox(frame, textvariable=var_kelly, values=["Quarter Kelly (~0% RoR)", "Half Kelly (2% RoR)", "Full Kelly (13% RoR)"], width=18, state="readonly").grid(row=5, column=1, pady=5)
    
    ttk.Separator(frame, orient='horizontal').grid(row=6, columnspan=2, sticky="ew", pady=10)

    # Table Rules Section
    ttk.Label(frame, text="Number of Decks:").grid(row=7, column=0, sticky="w", pady=5)
    ttk.Combobox(frame, textvariable=var_decks, values=[1,2,4,6,8], width=5, state="readonly").grid(row=7, column=1, pady=5)

    ttk.Label(frame, text="Deck Penetration:").grid(row=8, column=0, sticky="w", pady=5)
    ttk.Combobox(frame, textvariable=var_penetration, values=["50%", "75%", "85%"], width=5, state="readonly").grid(row=8, column=1, pady=5)

    ttk.Label(frame, text="Blackjack Payout:").grid(row=9, column=0, sticky="w", pady=5)
    ttk.Combobox(frame, textvariable=var_payout, values=["3:2", "6:5"], width=5, state="readonly").grid(row=9, column=1, pady=5)

    ttk.Checkbutton(frame, text="Dealer Stands on Soft 17 (S17)", variable=var_s17).grid(row=10, columnspan=2, sticky="w", pady=5)
    ttk.Checkbutton(frame, text="Double After Split Allowed (DAS)", variable=var_das).grid(row=11, columnspan=2, sticky="w", pady=5)
    ttk.Checkbutton(frame, text="Dealer Peeks for Blackjack (US)", variable=var_peek).grid(row=12, columnspan=2, sticky="w", pady=5)

    ttk.Label(frame, text="Surrender Rule:").grid(row=13, column=0, sticky="w", pady=5)
    ttk.Combobox(frame, textvariable=var_surrender, values=["late", "early", "none"], width=7, state="readonly").grid(row=13, column=1, pady=5)

    # --- Submission and Validation ---
    
    def on_start():
        """
        Validates user inputs, maps UI selections to programmatic values, 
        populates the configuration dictionary, and destroys the setup window.
        """
        try:
            # Enforce numeric boundaries and logical constraints
            br = var_bankroll.get()
            mb = var_min_bet.get()
            xb = var_max_bet.get()
            t = var_top.get()
            l = var_left.get()
            w = var_width.get()
            h = var_height.get()
            
            if br <= 0 or mb <= 0 or xb < mb or w <= 0 or h <= 0:
                raise ValueError("Logical value error")
                
        except Exception:
            # Catches both TypeErrors (e.g., text in an integer field) and our logical ValueErrors
            messagebox.showerror("Input Error", "Please ensure all fields contain valid numbers, and Max Bet >= Min Bet.")
            return

        # Map dropdown selections to their corresponding internal values
        surr_val = None if var_surrender.get() == "none" else var_surrender.get()
        payout_val = 1.5 if var_payout.get() == "3:2" else 1.2
        pen_val = float(var_penetration.get().strip('%')) / 100.0
        
        k_val = 0.5
        if "Quarter" in var_kelly.get(): k_val = 0.25
        elif "Full" in var_kelly.get(): k_val = 1.0
        
        # Populate the global config object
        config.update({
            'monitor': {"top": t, "left": l, "width": w, "height": h},
            'bankroll': br,
            'min_bet': mb,
            'max_bet': xb,
            'kelly_fraction': k_val,
            'decks': var_decks.get(),
            'penetration': pen_val,
            's17': var_s17.get(),
            'das': var_das.get(),
            'peek': var_peek.get(),
            'surrender': surr_val,
            'payout': payout_val
        })
        
        root.destroy() 

    ttk.Button(root, text="Start Bot", command=on_start).pack(pady=10)
    
    # Block execution here until the user submits the form or closes the window
    root.mainloop()
    return config

if __name__ == "__main__":
    print(run_setup_menu())