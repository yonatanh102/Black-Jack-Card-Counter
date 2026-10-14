import math

class Counter:
    """
    Tracks the state of the card shoe, including seen cards, 
    running count, and true count for card counting strategies.
    """
    def __init__(self, shoe):
        self.shoe = shoe
        # Initialize count for all possible card ranks
        self.seen = {r: 0 for r in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']}

    def update_count(self, card):
        """
        Registers a drawn card. 
        Returns False if the card exceeds the physical limit of the shoe to prevent duplicate scans.
        """
        max_per_rank = 4 * self.shoe.config.num_decks
        if self.seen[card] >= max_per_rank:
            print(f"[WARN] Ignored suspicious over-count for rank '{card}' "
                  f"(already at max {max_per_rank}). Possible duplicate detection.")
            return False
            
        self.seen[card] += 1
        return True

    def reset_count(self):
        """
        Resets the counting state (typically used after a shuffle).
        """
        for k in self.seen.keys():
            self.seen[k] = 0

    def get_remaining_cards(self):
        """
        Returns a dictionary mapping each rank to the number of remaining cards in the shoe.
        """
        remaining = {}
        initial_count = 4 * self.shoe.config.num_decks
        for rank, seen_count in self.seen.items():
            remaining[rank] = initial_count - seen_count
        return remaining

    @property
    def running_count(self):
        """
        Calculates the standard Hi-Lo running count.
        Low cards (2-6) = +1, High cards (10-A) = -1.
        """
        lower_cards = sum(self.seen[rank] for rank in ['2', '3', '4', '5', '6'])
        high_cards = sum(self.seen[rank] for rank in ['10', 'J', 'Q', 'K', 'A'])
        return lower_cards - high_cards

    @property
    def true_count(self):
        """
        Calculates the true count strictly derived from the internal state.
        Rounds remaining decks to the nearest half-deck for division using math.floor for predictability.
        """
        cards_played = sum(self.seen.values())
        decks_left_exact = (self.shoe.total_initial_cards - cards_played) / 52.0
        decks_left = max(0.5, math.floor(decks_left_exact * 2 + 0.5) / 2)
        
        return math.trunc(self.running_count / decks_left)

    def exact_probability_of_rank(self, target_rank):
        """
        Calculates the exact probability of drawing a specific rank as the next card.
        """
        cards_played = sum(self.seen.values())
        total_cards_remaining = self.shoe.total_initial_cards - cards_played

        if total_cards_remaining == 0:
            return 0.0

        initial_rank_count = 4 * self.shoe.config.num_decks
        rank_remaining = initial_rank_count - self.seen[target_rank]
        return rank_remaining / total_cards_remaining

    def player_bust_probability(self, current_sum, usable_aces):
        """
        Calculates the exact probability of the player busting on the very next card drawn.
        """
        # Soft hands cannot bust on a single hit
        if usable_aces > 0: 
            return 0.0

        total_cards_left = 0
        bust_cards_count = 0
        remaining_cards = self.get_remaining_cards()

        for rank, remaining in remaining_cards.items():
            if remaining > 0:
                total_cards_left += remaining
                card_val = 1 if rank == 'A' else (10 if rank in ['J', 'Q', 'K'] else int(rank))

                if current_sum + card_val > 21:
                    bust_cards_count += remaining

        if total_cards_left == 0: 
            return 0.0
            
        return bust_cards_count / total_cards_left