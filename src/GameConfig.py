class GameConfig:
    """
    Holds the casino table rules and parameters for the blackjack session.
    
    Attributes:
        num_decks (int): Number of decks in the shoe.
        penetration (float): Percentage of cards dealt before a shuffle (e.g., 0.75).
        s17 (bool): True if dealer stands on Soft 17, False if dealer hits (H17).
        das (bool): True if Double After Split is allowed.
        surrender (str): Surrender rules ("late", "early", or None).
        blackjack_payout (float): Payout ratio for natural blackjack (1.5 for 3:2, 1.2 for 6:5).
        peek (bool): True if dealer peeks for blackjack with a 10 or A showing (US rules).
    """
    def __init__(self, num_decks=6,
                       penetration=0.75,
                       s17=True,
                       das=True,
                       surrender="late",
                       blackjack_payout=1.5,
                       peek=True):

        self.num_decks = num_decks
        self.penetration = penetration
        self.s17 = s17 
        self.das = das 
        self.surrender = surrender 
        self.blackjack_payout = blackjack_payout 
        self.peek = peek

    def get_total_cards(self):
        """Returns the total physical cards present in a fresh shoe."""
        return self.num_decks * 52

    def get_cut_card_threshold(self):
        """Returns the exact number of cards that remain in the shoe when the cut card is reached."""
        return int(self.get_total_cards() * (1 - self.penetration))