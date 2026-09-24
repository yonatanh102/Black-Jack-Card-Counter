import math

class Counter:
    def __init__(self, shoe):
        self.shoe = shoe
        # Tracks exactly how many times each card has been played
        self.seen = {
            '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0,
            '8': 0, '9': 0, '10': 0, 'J': 0, 'Q': 0, 'K': 0, 'A': 0
        }

    def update_count(self, card):
        """Called by the screen reader to register a newly exposed card."""
        self.seen[card] += 1

    def reset_count(self):
        """Clears the count history when the dealer shuffles the shoe."""
        self.seen = {
            '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0,
            '8': 0, '9': 0, '10': 0, 'J': 0, 'Q': 0, 'K': 0, 'A': 0
        }

    def get_remaining_cards(self):
        """
        Single source of truth for cards left in the shoe.
        Returns a dictionary with the exact count of remaining cards per rank.
        """
        remaining = {}
        initial_count = 4 * self.shoe.config.num_decks
        for rank, seen_count in self.seen.items():
            remaining[rank] = initial_count - seen_count
        return remaining

    @property
    def running_count(self):
        """Calculates the standard Hi-Lo running count."""
        lower_cards = sum(self.seen[rank] for rank in ['2', '3', '4', '5', '6'])
        high_cards = sum(self.seen[rank] for rank in ['10', 'J', 'Q', 'K', 'A'])
        return lower_cards - high_cards

    @property
    def true_count(self):
        """
        Normalizes the running count based on remaining decks.
        Uses math.trunc to correctly handle negative counts (e.g., -1.5 becomes -1).
        """
        decks_left = self.shoe.decks_remaining()
        if decks_left <= 0:
            return self.running_count
        return math.trunc(self.running_count / decks_left)

    def exact_probability_of_rank(self, target_rank):
        """Calculates the exact probability of drawing a specific card rank."""
        cards_played = sum(self.seen.values())
        total_cards_remaining = self.shoe.total_initial_cards - cards_played

        if total_cards_remaining == 0:
            return 0.0

        initial_rank_count = 4 * self.shoe.config.num_decks
        rank_remaining = initial_rank_count - self.seen[target_rank]
        return rank_remaining / total_cards_remaining

    def dealer_bust_probability(self, dealer_upper_rank):
        """Calculates the probability of the dealer busting given their upcard."""
        remaining_cards = self.get_remaining_cards()

        def dfs_probability(current_sum, usable_aces, current_remaining):
            # Base cases: Dealer busts or stands
            if current_sum > 21:
                return 1.0
            if current_sum >= 17:
                return 0.0

            total_prob = 0.0
            total_cards_left = sum(current_remaining.values())

            if total_cards_left == 0:
                return 0.0

            # Traverse all possible next card draws
            for rank, count in current_remaining.items():
                if count > 0:
                    prob_of_draw = count / total_cards_left

                    # Determine card value
                    if rank == 'A':
                        card_val = 11
                    elif rank in ['J', 'Q', 'K']:
                        card_val = 10
                    else:
                        card_val = int(rank)

                    new_sum = current_sum + card_val
                    new_usable_aces = usable_aces + (1 if rank == 'A' else 0)

                    # Demote an ace from 11 to 1 if the sum exceeds 21
                    if new_sum > 21 and new_usable_aces > 0:
                        new_sum -= 10
                        new_usable_aces -= 1

                    # Backtracking: Temporarily remove the drawn card and recurse
                    current_remaining[rank] -= 1
                    branch_prob = dfs_probability(new_sum, new_usable_aces, current_remaining)
                    total_prob += prob_of_draw * branch_prob
                    current_remaining[rank] += 1
                    
            return total_prob

        initial_val = 11 if dealer_upper_rank == 'A' else (10 if dealer_upper_rank in ['J', 'Q', 'K'] else int(dealer_upper_rank))
        initial_aces = 1 if dealer_upper_rank == 'A' else 0

        # Note: The dealer's upcard is assumed to be already registered in self.seen 
        # by the screen reader, so we don't manually deduct it here.
        return dfs_probability(initial_val, initial_aces, remaining_cards)

    def player_bust_probability(self, current_sum, usable_aces):
        """Calculates the probability of the player busting on the very next hit."""
        # A soft hand (usable ace) can never bust on a single hit
        if usable_aces > 0:
            return 0.0

        total_cards_left = 0
        bust_cards_count = 0
        remaining_cards = self.get_remaining_cards()

        for rank, remaining in remaining_cards.items():
            if remaining > 0:
                total_cards_left += remaining

                if rank == 'A':
                    card_val = 1
                elif rank in ['J', 'Q', 'K']:
                    card_val = 10
                else:
                    card_val = int(rank)

                if current_sum + card_val > 21:
                    bust_cards_count += remaining

        if total_cards_left == 0:
            return 0.0

        return bust_cards_count / total_cards_left