
class GameConfig:
    def __init__(self, num_decks=6,
                       penetration=0.75,
                       s17=True,
                       das=True,
                       surrender="late",
                       blackjack_payout=1.5):

        self.num_decks = num_decks
        self.penetration = penetration
        self.s17 = s17 # True = Dealer Stands on Soft 17. False = Hits on Soft 17 (H17)
        self.das = das # Double After Split
        self.surrender = surrender # "late", "early", or None
        self.blackjack_payout = blackjack_payout # 1.5 (3:2) or 1.2 (6:5)

    def get_total_cards(self):
        return self.num_decks * 52

    def get_cut_card_threshold(self):
        return int(self.get_total_cards() * (1 - self.penetration))

    