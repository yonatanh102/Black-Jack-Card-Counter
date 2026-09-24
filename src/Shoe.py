from Card import Card
import random

class Shoe:
    def __init__(self, config):
        self.config = config
        self.cards = self._generate_shoe()
        self.total_initial_cards = self.config.get_total_cards()
        self.cut_card_threshold = self.config.get_cut_card_threshold()

    def _generate_shoe(self):
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']

        shoe = [Card(rank, suit) for rank in ranks for suit in suits] * self.config.num_decks
        random.shuffle(shoe)
        return shoe

    def draw_card(self):
        if not self.cards:
            return None
        return self.cards.pop()

    def decks_remaining(self):
        exact_decks = len(self.cards) / 52.0
        return max(0.5, round(exact_decks * 2) / 2)

    def shuffle(self):
        return len(self.cards) <= self.cut_card_threshold

    def reset_shoe(self):
        self.cards = self._generate_shoe()