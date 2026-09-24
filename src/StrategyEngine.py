class StrategyEngine: 
    def __init__(self, counter):
        self.counter = counter

    def calculate_new_sum(self, current_sum, rank, usable_aces):
        """
        Helper to compute hand value and manage soft aces dynamically.
        Converts a soft ace (11) to a hard ace (1) if the sum exceeds 21.
        """
        if rank == 'A':
            card_val = 11
            usable_aces += 1
        elif rank in ['J', 'Q', 'K']:
            card_val = 10
        else:
            card_val = int(rank)

        new_sum = current_sum + card_val

        if new_sum > 21 and usable_aces > 0:
            new_sum -= 10
            usable_aces -= 1
        return new_sum, usable_aces

    def get_dealer_distribution(self, dealer_upcard, current_remaining):
        """
        Builds a complete probability distribution of the dealer's final hand.
        Accounts for S17/H17 table rules via GameConfig.
        Utilizes memoization to reduce time complexity from exponential to polynomial.
        """
        is_s17 = self.counter.shoe.config.s17
        memo = {}

        def dfs(current_sum, usable_aces, current_remaining):
            # Base cases for dealer final outcomes
            if current_sum > 21: 
                return {17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 'Bust': 1.0}
            if current_sum > 17:
                dist = {17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 'Bust': 0.0}
                dist[current_sum] = 1.0
                return dist
                
            # Handle Soft 17 vs Hard 17 rules
            if current_sum == 17:
                if usable_aces == 0 or is_s17:
                    dist = {17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 'Bust': 0.0}
                    dist[17] = 1.0
                    return dist

            # State key for memoization. Tuple is used to ensure it is hashable.
            state_key = (current_sum, usable_aces, tuple(current_remaining.items()))
            if state_key in memo:
                return memo[state_key]

            total_prob = {17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 'Bust': 0.0}
            total_cards_left = sum(current_remaining.values())

            if total_cards_left == 0:
                return total_prob

            for rank, count in current_remaining.items():
                if count > 0:
                    prob_of_draw = count / total_cards_left
                    new_sum, new_aces = self.calculate_new_sum(current_sum, rank, usable_aces)

                    # Backtracking: Temporarily remove the drawn card and recurse
                    current_remaining[rank] -= 1
                    branch_dist = dfs(new_sum, new_aces, current_remaining)
                    current_remaining[rank] += 1

                    for outcome in total_prob.keys():
                        total_prob[outcome] += prob_of_draw * branch_dist[outcome]

            memo[state_key] = total_prob
            return total_prob

        initial_sum, initial_aces = self.calculate_new_sum(0, dealer_upcard, 0)
        return dfs(initial_sum, initial_aces, current_remaining)

    def should_buy_insurance(self):
        """
        Determines if insurance is mathematically profitable (+EV).
        Requires the remaining tens ratio to be strictly greater than 1/3.
        """
        current_remaining_cards = self.counter.get_remaining_cards()
        total_cards_left = sum(current_remaining_cards.values())

        if total_cards_left == 0:
            return False

        tens_remaining = sum(current_remaining_cards[rank] for rank in ['10', 'J', 'Q', 'K'])
        probability_of_tens = tens_remaining / total_cards_left

        return probability_of_tens > (1 / 3)

    def ev_surrender(self):
        """Surrender always forfeits exactly half the original bet."""
        return -0.5

    def ev_stand(self, player_sum, dealer_distribution):
        """Calculates the expected value of standing against the dealer's distribution."""
        if player_sum > 21:
            return -1.0
        
        expected_value = 0.0

        for dealer_result, probability in dealer_distribution.items():
            if dealer_result == 'Bust':
                expected_value += probability * 1.0
            elif dealer_result > player_sum:
                expected_value += probability * -1.0
            elif dealer_result < player_sum:
                expected_value += probability * 1.0
            else:
                expected_value += probability * 0.0 # Push (Tie)

        return expected_value

    def ev_double(self, player_sum, usable_aces, dealer_distribution, current_remaining):
        """
        Calculates EV of doubling down (one card only, bet is doubled).
        Takes 'current_remaining' as an argument to maintain thread-safe simulation state.
        """
        total_ev = 0.0
        total_cards_left = sum(current_remaining.values())
        if total_cards_left == 0:
            return 0.0

        for rank, count in current_remaining.items():
            if count > 0:
                prob_of_draw = count / total_cards_left
                new_sum, new_aces = self.calculate_new_sum(player_sum, rank, usable_aces)
                
                stand_ev = self.ev_stand(new_sum, dealer_distribution)
                total_ev += prob_of_draw * (stand_ev * 2)

        return total_ev

    def ev_hit(self, player_sum, usable_aces, dealer_distribution, current_remaining, depth=1, max_depth=5, memo=None):
        """
        Recursively calculates the EV of hitting.
        Uses a local copy of remaining cards and memoization to prevent race conditions 
        and optimize real-time calculations.
        """
        if memo is None:
            memo = {}
            
        # State key for memoization
        state_key = (player_sum, usable_aces, depth, tuple(current_remaining.items()))
        if state_key in memo:
            return memo[state_key]

        if player_sum > 21:
            return -1.0
        
        # Depth limit to prevent infinite recursion and ensure high performance
        if depth >= max_depth:
            return self.ev_stand(player_sum, dealer_distribution)

        total_ev = 0.0
        total_cards_left = sum(current_remaining.values())

        if total_cards_left == 0:
            return 0.0

        for rank, count in current_remaining.items():
            if count > 0:
                prob_of_draw = count / total_cards_left
                new_sum, new_aces = self.calculate_new_sum(player_sum, rank, usable_aces)

                if new_sum > 21:
                    branch_ev = -1.0
                else:
                    # Modify local state only for backtracking
                    current_remaining[rank] -= 1

                    stand_ev = self.ev_stand(new_sum, dealer_distribution)
                    hit_ev = self.ev_hit(new_sum, new_aces, dealer_distribution, current_remaining, depth+1, max_depth, memo)

                    current_remaining[rank] += 1
                    
                    # Optimal play: player chooses the action with the highest EV
                    branch_ev = max(stand_ev, hit_ev)

                total_ev += prob_of_draw * branch_ev

        memo[state_key] = total_ev
        return total_ev

    def ev_split(self, split_card_rank, dealer_distribution, current_remaining):
        """
        Calculates EV of splitting pairs.
        Approximates EV by solving for one hand and multiplying by 2.

        ASSUMPTION: This engine assumes a maximum of one split (2 hands total). 
        It does not currently calculate the marginal EV gains from resplitting 
        (e.g., splitting to 3 or 4 hands), which is computationally heavier but 
        generally a minor deviation from optimal play.
        """
        total_cards_left = sum(current_remaining.values())
        if total_cards_left == 0:
            return 0.0

        single_hand_expected_value = 0.0
        das_allowed = self.counter.shoe.config.das

        for rank, count in current_remaining.items():
            if count > 0:
                prob_of_draw = count / total_cards_left
                new_sum, usable_aces = self.calculate_new_sum(0, split_card_rank, 0)
                new_sum, usable_aces = self.calculate_new_sum(new_sum, rank, usable_aces)

                # Special rule: Splitting aces usually restricts the player to exactly one draw
                if split_card_rank == 'A':
                    best_ev_for_this_draw = self.ev_stand(new_sum, dealer_distribution)
                else:
                    current_remaining[rank] -= 1 

                    stand_ev = self.ev_stand(new_sum, dealer_distribution)
                    hit_ev = self.ev_hit(new_sum, usable_aces, dealer_distribution, current_remaining)

                    evs = [stand_ev, hit_ev]

                    if das_allowed:
                        double_ev = self.ev_double(new_sum, usable_aces, dealer_distribution, current_remaining)
                        evs.append(double_ev)

                    current_remaining[rank] += 1
                    best_ev_for_this_draw = max(evs)

                single_hand_expected_value += prob_of_draw * best_ev_for_this_draw

        return single_hand_expected_value * 2 

    def get_best_action(self, player_cards, dealer_upcard, is_first_action=True):
        """
        Core entry point for the recommendation engine.
        Evaluates all legal moves and returns the optimal action with its associated EV.
        
        CRITICAL CONTRACT FOR SCREEN READER (INTEGRATION):
        This function assumes that `counter.update_count()` HAS ALREADY BEEN CALLED 
        for all `player_cards` and the `dealer_upcard` currently on the table.
        Failure to register these cards in the Counter before calling this function 
        will result in mathematically inaccurate EV calculations (silent failure).
        """
        current_remaining = self.counter.get_remaining_cards()
        player_sum = 0
        usable_aces = 0
        for card_rank in player_cards:
            player_sum, usable_aces = self.calculate_new_sum(player_sum, card_rank, usable_aces)

        # Handle natural Blackjack immediately without computing the dealer's DFS tree
        if is_first_action and len(player_cards) == 2 and player_sum == 21:
            payout = self.counter.shoe.config.blackjack_payout
            
            # Check if dealer can also have a natural blackjack
            dealer_natural_bj_prob = 0.0
            total_cards_left = sum(current_remaining.values())
            
            if total_cards_left > 0:
                if dealer_upcard == 'A':
                    tens_count = sum(current_remaining[r] for r in ['10', 'J', 'Q', 'K'])
                    dealer_natural_bj_prob = tens_count / total_cards_left
                elif dealer_upcard in ['10', 'J', 'Q', 'K']:
                    dealer_natural_bj_prob = current_remaining['A'] / total_cards_left

            # Player wins the BJ payout unless dealer also has a natural BJ (Push)
            ev = (1.0 - dealer_natural_bj_prob) * payout
            
            return {
                'recommended_action': 'Stand',
                'expected_value': ev,
                'all_evs': {'Stand': ev},
                'dealer_bust_prob': 0.0 # Hand ends immediately
            }

        # For all other hands, calculate the full dealer distribution based on the current state
        dealer_distribution = self.get_dealer_distribution(dealer_upcard, current_remaining)
        
        evs = {
            'Stand': self.ev_stand(player_sum, dealer_distribution),
            'Hit': self.ev_hit(player_sum, usable_aces, dealer_distribution, current_remaining)
        }
        
        if is_first_action:
            evs['Double'] = self.ev_double(player_sum, usable_aces, dealer_distribution, current_remaining)
            
            # Check table config if surrender is allowed
            if self.counter.shoe.config.surrender is not None:
                evs['Surrender'] = self.ev_surrender()
                
            # Identify splittable pairs
            if len(player_cards) == 2 and player_cards[0] == player_cards[1]:
                evs['Split'] = self.ev_split(player_cards[0], dealer_distribution, current_remaining)

        best_action = max(evs, key=evs.get)
        best_ev = evs[best_action]

        return {
            'recommended_action': best_action,
            'expected_value': best_ev,
            'all_evs': evs,
            'dealer_bust_prob': dealer_distribution.get('Bust', 0.0)
        }