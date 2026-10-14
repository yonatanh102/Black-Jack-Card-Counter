class StrategyEngine:
    """
    The statistical core of the Blackjack bot.
    Calculates the Expected Value (EV) for all possible player actions (Hit, Stand, Double, Split, Surrender).
    It uses Depth-First Search (DFS) to project the dealer's final hand distribution based on the exact
    composition of the remaining shoe, dynamically adapting to specific table rules (S17/H17, DAS, Peek).
    """
    def __init__(self, counter):
        self.counter = counter
        # Cache the config to avoid redundant lookups and prevent AttributeError during deep recursion
        self.config = counter.shoe.config

    def calculate_new_sum(self, current_sum, card_rank, usable_aces):
        """
        Calculates the updated numerical value of a hand after drawing a new card.
        Dynamically downgrades 'soft' aces (11 to 1) if the hand value exceeds 21.
        """
        val = 11 if card_rank == 'A' else (10 if card_rank in ['J', 'Q', 'K'] else int(card_rank))
        new_sum = current_sum + val
        new_aces = usable_aces + (1 if card_rank == 'A' else 0)
        
        # Retroactively adjust aces if the new card busts the hand
        while new_sum > 21 and new_aces > 0:
            new_sum -= 10
            new_aces -= 1
            
        return new_sum, new_aces

    def get_dealer_distribution(self, dealer_upcard, current_remaining=None):
        """
        Generates a probability distribution of the dealer's final hand (17, 18, 19, 20, 21, or Bust).
        Uses a memoized DFS algorithm to traverse all possible dealer drawing sequences based on the exact cards left.
        """
        if current_remaining is None:
            current_remaining = self.counter.get_remaining_cards()
            
        memo = {}
        s17_rule = getattr(self.config, 's17', True)
        
        def dfs(current_sum, usable_aces, remaining):
            # Memoization key encapsulates the exact gamestate to prevent duplicate subtree calculations
            state_key = (current_sum, usable_aces, tuple(remaining.values()))
            if state_key in memo:
                return memo[state_key]

            # Base Case 1: Dealer busts
            if current_sum > 21: 
                res = {'Bust': 1.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0}
                memo[state_key] = res
                return res
                
            # Base Case 2: Dealer must stand. 
            # Differentiates between Soft 17 and Hard 17 based on table configuration.
            if current_sum > 17 or (current_sum == 17 and (s17_rule or usable_aces == 0)):
                res = {'Bust': 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0}
                res[current_sum] = 1.0
                memo[state_key] = res
                return res

            # Recursive step: Calculate probabilities for all valid next card draws
            total_prob = {17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 'Bust': 0.0}
            total_cards_left = sum(remaining.values())
            
            if total_cards_left == 0: 
                memo[state_key] = total_prob
                return total_prob

            for rank, count in remaining.items():
                if count > 0:
                    prob = count / total_cards_left
                    s, a = self.calculate_new_sum(current_sum, rank, usable_aces)
                    
                    # Temporarily remove the drawn card to maintain state accuracy down the tree
                    remaining[rank] -= 1
                    branch_prob = dfs(s, a, remaining)
                    remaining[rank] += 1  # Backtrack
                    
                    # Accumulate probabilities from this specific branch
                    for k in total_prob:
                        total_prob[k] += prob * branch_prob[k]
                        
            memo[state_key] = total_prob
            return total_prob

        # Initialize dealer's starting variables based on the visible upcard
        initial_val = 11 if dealer_upcard == 'A' else (10 if dealer_upcard in ['J', 'Q', 'K'] else int(dealer_upcard))
        initial_aces = 1 if dealer_upcard == 'A' else 0
        
        # Hole Card / Peek Logic:
        # If the casino rules state the dealer peeks for blackjack (US rules), we know the hole card
        # cannot be one that completes a blackjack (otherwise the round would have already ended).
        # We must filter these specific cards out of the initial probability calculation.
        if getattr(self.config, 'peek', True):
            TENS = ('10', 'J', 'Q', 'K')
            banned = TENS if dealer_upcard == 'A' else (('A',) if dealer_upcard in TENS else ())
        else:
            banned = () # European No Hole Card (ENHC): No cards are restricted.
        
        if not banned:
            return dfs(initial_val, initial_aces, current_remaining)
            
        total_valid_cards = sum(c for r, c in current_remaining.items() if r not in banned)
        if total_valid_cards == 0: 
            return {17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 'Bust': 0.0}
            
        result = {17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 'Bust': 0.0}
        
        # Distribute probabilities only across valid (non-blackjack) hole cards
        for rank, count in current_remaining.items():
            if rank in banned or count == 0: 
                continue
            
            s, a = self.calculate_new_sum(initial_val, rank, initial_aces)
            current_remaining[rank] -= 1
            d = dfs(s, a, current_remaining)
            current_remaining[rank] += 1
            
            for k in result: 
                result[k] += (count / total_valid_cards) * d[k]
                
        return result

    def ev_stand(self, player_sum, dealer_distribution):
        """
        Calculates the expected value if the player chooses to Stand.
        Compares the player's static total against the dealer's projected outcome distribution.
        """
        if player_sum > 21:
            return -1.0 # Player busts automatically lose regardless of dealer outcome
        
        expected_value = 0.0
        for dealer_result, probability in dealer_distribution.items():
            if dealer_result == 'Bust':
                expected_value += probability * 1.0
            elif dealer_result > player_sum:
                expected_value += probability * -1.0
            elif dealer_result < player_sum:
                expected_value += probability * 1.0
            # Ties (pushes) add 0.0 to the EV, so they are implicitly handled.

        return expected_value

    def ev_hit(self, player_sum, usable_aces, dealer_distribution, current_remaining, memo=None):
        """
        Recursively calculates the expected value if the player chooses to Hit.
        Assumes optimal subsequent play (the player will always choose the max EV between hitting again or standing).
        """
        if memo is None:
            memo = {}
            
        state_key = (player_sum, usable_aces, tuple(current_remaining.values()))
        if state_key in memo:
            return memo[state_key]

        if player_sum > 21:
            return -1.0
        
        total_ev = 0.0
        total_cards_left = sum(current_remaining.values())

        if total_cards_left == 0:
            return 0.0

        for rank, count in current_remaining.items():
            if count > 0:
                prob_of_draw = count / total_cards_left
                new_s, new_a = self.calculate_new_sum(player_sum, rank, usable_aces)

                if new_s > 21:
                    branch_ev = -1.0
                else:
                    current_remaining[rank] -= 1
                    # Evaluate the two possible futures after drawing this card: Stand or Hit again
                    stand_ev = self.ev_stand(new_s, dealer_distribution)
                    hit_ev = self.ev_hit(new_s, new_a, dealer_distribution, current_remaining, memo)
                    current_remaining[rank] += 1
                    
                    # The player acts optimally, choosing whichever action yields higher EV
                    branch_ev = max(stand_ev, hit_ev)

                total_ev += prob_of_draw * branch_ev

        memo[state_key] = total_ev
        return total_ev

    def get_best_action(self, player_cards, dealer_upcard, is_first_action=True, is_split_hand=False):
        """
        Main decision-making router. Evaluates all legal moves for the current hand state
        and identifies the action with the highest Expected Value (EV).
        """
        current_remaining = self.counter.get_remaining_cards()
        
        def card_val(c):
            return 11 if c == 'A' else (10 if c in ['J', 'Q', 'K'] else int(c))

        # Splitting is strictly limited to identical physical ranks (e.g., blocks splitting a K and a Q)
        can_split = len(player_cards) == 2 and player_cards[0] == player_cards[1]
        
        player_sum = 0
        usable_aces = 0
        for card in player_cards:
            player_sum += card_val(card)
            if card == 'A':
                usable_aces += 1
                
        while player_sum > 21 and usable_aces > 0:
            player_sum -= 10
            usable_aces -= 1

        # Evaluate Natural Blackjack
        # A natural is only valid on the first two cards of a non-split hand.
        if is_first_action and not is_split_hand and len(player_cards) == 2:
            if player_sum == 21:
                payout = getattr(self.config, 'blackjack_payout', 1.5)
                dealer_natural_bj_prob = 0.0
                total_cards_left = sum(current_remaining.values())
                
                # Calculate the probability of a dealer Blackjack (resulting in a Push, EV = 0)
                if total_cards_left > 0:
                    if dealer_upcard == 'A':
                        tens_count = sum(current_remaining[r] for r in ['10', 'J', 'Q', 'K'])
                        dealer_natural_bj_prob = tens_count / total_cards_left
                    elif dealer_upcard in ['10', 'J', 'Q', 'K']:
                        dealer_natural_bj_prob = current_remaining.get('A', 0) / total_cards_left
                
                ev = payout * (1.0 - dealer_natural_bj_prob)
                return {
                    'recommended_action': 'Stand', 
                    'expected_value': ev, 
                    'all_evs': {'Stand': ev},
                    'dealer_bust_prob': 0.0,
                    'deviation_note': 'Natural Blackjack'
                }

        # Pre-calculate the dealer's distribution, as it acts as the baseline for all subsequent EV evaluations
        dealer_dist = self.get_dealer_distribution(dealer_upcard, current_remaining)
        
        # Special rule constraint: Splitting Aces usually forces a stand after receiving exactly one card
        if is_split_hand and player_cards[0] == 'A':
            ev_stand = self.ev_stand(player_sum, dealer_dist)
            return {
                'recommended_action': 'Stand',
                'expected_value': ev_stand,
                'all_evs': {'Stand': ev_stand},
                'dealer_bust_prob': dealer_dist.get('Bust', 0.0),
                'deviation_note': 'Split Aces (No hit allowed)'
            }

        # Calculate base actions
        ev_stand = self.ev_stand(player_sum, dealer_dist)
        ev_h = self.ev_hit(player_sum, usable_aces, dealer_dist, current_remaining)
        
        # Calculate Double Down EV (Requires exactly two cards and checking DAS permissions)
        ev_double = float('-inf')
        if is_first_action and len(player_cards) == 2:
            if not is_split_hand or getattr(self.config, 'das', True):
                ev_double = 0.0
                total_cards = sum(current_remaining.values())
                if total_cards > 0:
                    for rank, count in current_remaining.items():
                        if count > 0:
                            prob = count / total_cards
                            new_s, new_a = self.calculate_new_sum(player_sum, rank, usable_aces)
                            if new_s > 21:
                                ev_double += prob * (-2.0) # Bet is doubled, loss is doubled
                            else:
                                stand_val = self.ev_stand(new_s, dealer_dist)
                                ev_double += prob * (stand_val * 2.0)

        # Calculate Split EV
        # Models splitting a pair into two independent hands and assumes optimal play on each.
        ev_split = float('-inf')
        if can_split and is_first_action and not is_split_hand:
            split_card_val = card_val(player_cards[0])
            s1 = split_card_val
            a1 = 1 if split_card_val == 11 else 0
            
            ev_split = 0.0
            total_cards = sum(current_remaining.values())
            if total_cards > 0:
                for rank, count in current_remaining.items():
                    if count > 0:
                        prob = count / total_cards
                        new_s, new_a = self.calculate_new_sum(s1, rank, a1)
                        if new_s > 21:
                            ev_split += prob * (-1.0)
                        else:
                            if split_card_val == 11:
                                # Splitting Aces forces an immediate stand
                                best_ev_for_draw = self.ev_stand(new_s, dealer_dist)
                            else:
                                current_remaining[rank] -= 1
                                stand_val = self.ev_stand(new_s, dealer_dist)
                                hit_val = self.ev_hit(new_s, new_a, dealer_dist, current_remaining)
                                evs_to_compare = [stand_val, hit_val]
                                
                                # Include Double Down branches if DAS is permitted
                                if getattr(self.config, 'das', True):
                                    double_val = 0.0
                                    sub_cards = sum(current_remaining.values())
                                    if sub_cards > 0:
                                        for r2, c2 in current_remaining.items():
                                            if c2 > 0:
                                                p2 = c2 / sub_cards
                                                ns2, na2 = self.calculate_new_sum(new_s, r2, new_a)
                                                if ns2 > 21: double_val += p2 * (-2.0)
                                                else: double_val += p2 * (self.ev_stand(ns2, dealer_dist) * 2.0)
                                        evs_to_compare.append(double_val)
                                        
                                current_remaining[rank] += 1
                                best_ev_for_draw = max(evs_to_compare)
                                
                            ev_split += prob * best_ev_for_draw
            ev_split *= 2.0 # The EV of one side of the split is mirrored for the second hand

        # Surrender EV (Guaranteed forfeiture of 50% of the bet)
        ev_surrender = float('-inf')
        if getattr(self.config, 'surrender', None) and is_first_action and not is_split_hand and len(player_cards) == 2:
            ev_surrender = -0.5

        # Consolidate viable actions and identify the mathematical maximum
        actions = {'Stand': ev_stand, 'Hit': ev_h}
        if ev_double != float('-inf'): actions['Double'] = ev_double
        if ev_split != float('-inf'): actions['Split'] = ev_split
        if ev_surrender != float('-inf'): actions['Surrender'] = ev_surrender

        best_action = max(actions, key=actions.get)
        best_ev = actions[best_action]
        
        note = self.get_deviation_note(player_cards, dealer_upcard, best_action)
        return {
            'recommended_action': best_action,
            'expected_value': best_ev,
            'all_evs': actions,
            'dealer_bust_prob': dealer_dist.get('Bust', 0.0),
            'deviation_note': note
        }

    def get_deviation_note(self, hand, dealer_up, recommended_action):
        """
        Cross-references the generated optimal action against standard card counting matrices 
        (Illustrious 18 and Fab 4). Adds an informational tag if a recognized deviation is executed.
        """
        tc = self.counter.true_count
        s17 = getattr(self.config, 's17', True)
        
        # Normalize all face cards to '10' for standardized index lookups
        dealer_val = '10' if dealer_up in ['J', 'Q', 'K'] else dealer_up
        
        value = 0
        aces = 0
        for card in hand:
            if card in ['J', 'Q', 'K']: value += 10
            elif card == 'A':
                value += 11
                aces += 1
            else: value += int(card)
            
        while value > 21 and aces > 0:
            value -= 10
            aces -= 1
            
        is_soft = aces > 0 and value <= 21
        is_pair = len(hand) == 2 and hand[0] == hand[1]
        
        if is_pair and value == 20 and recommended_action == 'Split':
            if dealer_val == '5' and tc >= 5: return "I18: Split 10s vs 5 (TC >= 5)"
            if dealer_val == '6' and tc >= 4: return "I18: Split 10s vs 6 (TC >= 4)"
            
        if not is_soft and not is_pair:
            if recommended_action == 'Surrender':
                if value == 15 and dealer_val == '10' and tc >= 0: return "Fab 4: Surrender 15 vs 10 (TC >= 0)"
                if value == 15 and dealer_val == '9' and tc >= 2: return "Fab 4: Surrender 15 vs 9 (TC >= 2)"
                if value == 15 and dealer_val == 'A':
                    if s17 and tc >= 1: return "Fab 4: Surrender 15 vs A (S17, TC >= 1)"
                    if not s17 and tc >= -1: return "Fab 4: Surrender 15 vs A (H17, TC >= -1)"
                if value == 14 and dealer_val == '10' and tc >= 3: return "Fab 4: Surrender 14 vs 10 (TC >= 3)"
                
            if value == 16 and dealer_val == '10':
                if recommended_action == 'Stand' and tc >= 0: return "I18: Stand 16 vs 10 (TC >= 0)"
                if recommended_action == 'Hit' and tc < 0: return "Deviation: Hit 16 vs 10 (TC < 0)"
                
            if value == 15 and dealer_val == '10' and recommended_action == 'Stand' and tc >= 4: return "I18: Stand 15 vs 10 (TC >= 4)"
            if value == 14 and dealer_val == '10' and recommended_action == 'Stand' and tc >= 3: return "I18: Stand 14 vs 10 (TC >= 3)"
            if value == 12 and dealer_val == '3' and recommended_action == 'Stand' and tc >= 2: return "I18: Stand 12 vs 3 (TC >= 2)"
            if value == 12 and dealer_val == '2' and recommended_action == 'Stand' and tc >= 3: return "I18: Stand 12 vs 2 (TC >= 3)"
            if value == 16 and dealer_val == '9' and recommended_action == 'Stand' and tc >= 4: return "I18: Stand 16 vs 9 (TC >= 4)"
            
            if value == 13 and dealer_val == '2' and recommended_action == 'Hit' and tc <= -1: return "Deviation: Hit 13 vs 2 (TC <= -1)"
            if value == 12 and dealer_val == '4' and recommended_action == 'Hit' and tc < 0: return "Deviation: Hit 12 vs 4 (TC < 0)"
            if value == 12 and dealer_val == '5' and recommended_action == 'Hit' and tc <= -2: return "Deviation: Hit 12 vs 5 (TC <= -2)"
            if value == 12 and dealer_val == '6' and recommended_action == 'Hit' and tc <= -1: return "Deviation: Hit 12 vs 6 (TC <= -1)"
            if value == 13 and dealer_val == '3' and recommended_action == 'Hit' and tc <= -2: return "Deviation: Hit 13 vs 3 (TC <= -2)"
            
            if recommended_action == 'Double':
                if value == 11 and dealer_val == 'A':
                    if s17 and tc >= 1: return "I18: Double 11 vs A (S17, TC >= 1)"
                    if not s17 and tc >= 0: return "Deviation: Double 11 vs A (H17, TC >= 0)"
                if value == 10 and dealer_val == '10' and tc >= 4: return "I18: Double 10 vs 10 (TC >= 4)"
                if value == 10 and dealer_val == 'A':
                    if s17 and tc >= 4: return "I18: Double 10 vs A (S17, TC >= 4)"
                    if not s17 and tc >= 3: return "Deviation: Double 10 vs A (H17, TC >= 3)"
                if value == 9 and dealer_val == '2' and tc >= 1: return "I18: Double 9 vs 2 (TC >= 1)"
                if value == 9 and dealer_val == '7' and tc >= 3: return "I18: Double 9 vs 7 (TC >= 3)"
        
        return ""

    def should_buy_insurance(self):
        """
        Evaluates if buying insurance is mathematically profitable (+EV).
        Profitable only when the ratio of tens to remaining cards is strictly > 33.3%.
        """
        current_remaining_cards = self.counter.get_remaining_cards()
        total_cards_left = sum(current_remaining_cards.values())

        if total_cards_left == 0:
            return False

        tens_remaining = sum(current_remaining_cards[rank] for rank in ['10', 'J', 'Q', 'K'])
        probability_of_tens = tens_remaining / total_cards_left

        return probability_of_tens > (1 / 3)