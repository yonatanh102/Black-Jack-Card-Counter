import pytest
import math
from Card import Card
from GameConfig import GameConfig
from Shoe import Shoe
from Counter import Counter
from StrategyEngine import StrategyEngine

# --- Fixtures ---
@pytest.fixture
def empty_counter_env():
    """Provides a fresh game environment before each test."""
    config = GameConfig(num_decks=6, s17=True)
    shoe = Shoe(config)
    counter = Counter(shoe)
    engine = StrategyEngine(counter)
    return shoe, counter, engine

# --- Tests for Shoe ---
def test_shoe_initialization(empty_counter_env):
    shoe, _, _ = empty_counter_env
    assert shoe.config.num_decks == 6
    assert len(shoe.cards) == 6 * 52
    assert shoe.decks_remaining() == 6.0

def test_draw_card_reduces_count(empty_counter_env):
    shoe, _, _ = empty_counter_env
    initial_count = len(shoe.cards)
    card = shoe.draw_card()
    assert card is not None
    assert len(shoe.cards) == initial_count - 1

def test_shoe_shuffle_exact_boundary(empty_counter_env):
    """Verifies the shuffle flag triggers exactly on the cut card threshold."""
    shoe, _, _ = empty_counter_env
    # Draw cards until exactly at the threshold
    while len(shoe.cards) > shoe.cut_card_threshold:
        shoe.draw_card()
    assert shoe.shuffle() is True

def test_shoe_draw_empty_returns_none(empty_counter_env):
    """Ensures drawing from an empty shoe returns None gracefully without crashing."""
    shoe, _, _ = empty_counter_env
    shoe.cards = [] 
    assert shoe.draw_card() is None

def test_shoe_decks_remaining_rounding(empty_counter_env):
    """Checks that boundary values (e.g., exactly 0.25 decks) round correctly to 0.5."""
    shoe, _, _ = empty_counter_env
    # 13 cards exactly = 0.25 decks
    shoe.cards = [Card('2', 'Hearts')] * 13 
    assert shoe.decks_remaining() == 0.5

# --- Tests for Counter ---
def test_running_count_updates_correctly(empty_counter_env):
    _, counter, _ = empty_counter_env

    for rank in ['2', '3', '4', '5', '6']:
        counter.update_count(rank)
    assert counter.running_count == 5

    for rank in ['7', '8', '9']:
        counter.update_count(rank)
    assert counter.running_count == 5

    for rank in ['10', 'J', 'Q', 'K', 'A']:
        counter.update_count(rank)
    assert counter.running_count == 0

def test_true_count_calculation(empty_counter_env):
    shoe, counter, _ = empty_counter_env
    for _ in range(52 * 3):
        shoe.draw_card()
    for _ in range(6):
        counter.update_count('5')
    assert counter.true_count == 2

def test_true_count_negative(empty_counter_env):
    """Verifies that a negative running count truncates towards zero (e.g., -0.5 -> 0)."""
    _, counter, _ = empty_counter_env
    for _ in range(3):
        counter.update_count('K')
    # Running count is -3, decks left is 6. True count = math.trunc(-3/6) = 0
    assert counter.true_count == 0

def test_exact_prob_exhausted_rank(empty_counter_env):
    """Ensures probability is 0.0 when all cards of a specific rank have been dealt."""
    shoe, counter, _ = empty_counter_env
    total_aces = 4 * shoe.config.num_decks
    for _ in range(total_aces):
        counter.update_count('A')
    assert counter.exact_probability_of_rank('A') == 0.0

def test_player_bust_probability_safe_hand(empty_counter_env):
    _, counter, _ = empty_counter_env
    assert counter.player_bust_probability(current_sum=11, usable_aces=0) == 0.0

def test_player_bust_probability_soft_hand(empty_counter_env):
    _, counter, _ = empty_counter_env
    assert counter.player_bust_probability(current_sum=16, usable_aces=1) == 0.0

def test_player_bust_probability_hard_16(empty_counter_env):
    _, counter, _ = empty_counter_env
    expected_prob = 192 / 312
    assert counter.player_bust_probability(current_sum=16, usable_aces=0) == pytest.approx(expected_prob)

def test_player_bust_probability_21(empty_counter_env):
    """A hard 21 will always bust on any drawn card."""
    _, counter, _ = empty_counter_env
    assert counter.player_bust_probability(current_sum=21, usable_aces=0) == 1.0

def test_counter_zero_cards_left_no_div_zero(empty_counter_env):
    """Prevents ZeroDivisionError in probability calculations when the shoe is completely empty."""
    shoe, counter, _ = empty_counter_env
    for rank in counter.seen:
        counter.seen[rank] = 4 * shoe.config.num_decks
        
    assert counter.exact_probability_of_rank('2') == 0.0
    assert counter.dealer_bust_probability('6') == 0.0
    assert counter.player_bust_probability(12, 0) == 0.0

def test_dealer_bust_prob_upcard_already_seen(empty_counter_env):
    """
    Regression Test: Ensures the dealer bust calculation doesn't crash 
    if the upcard was already processed by the screen reader into counter.seen.
    """
    _, counter, _ = empty_counter_env
    counter.update_count('6')
    prob = counter.dealer_bust_probability('6')
    assert 0.0 <= prob <= 1.0

# --- Tests for Strategy Engine ---
def test_calculate_new_sum(empty_counter_env):
    _, _, engine = empty_counter_env
    new_sum, aces = engine.calculate_new_sum(0, 'A', 0)
    assert new_sum == 11 and aces == 1

    new_sum, aces = engine.calculate_new_sum(new_sum, 'A', aces)
    assert new_sum == 12 and aces == 1

    new_sum, aces = engine.calculate_new_sum(new_sum, 'K', aces)
    assert new_sum == 12 and aces == 0

    new_sum, aces = engine.calculate_new_sum(new_sum, '7', aces)
    assert new_sum == 19 and aces == 0

def test_best_action_hard_20(empty_counter_env):
    _, _, engine = empty_counter_env
    result = engine.get_best_action(player_cards=['10', 'J'], dealer_upcard='6', is_first_action=True)
    assert result['recommended_action'] == 'Stand'
    assert result['expected_value'] > 0

def test_best_action_hard_11(empty_counter_env):
    _, _, engine = empty_counter_env
    result = engine.get_best_action(player_cards=['4', '7'], dealer_upcard='6', is_first_action=True)
    assert result['recommended_action'] == 'Double'
    assert result['expected_value'] > 0

def test_best_action_never_split_fives(empty_counter_env):
    """Splitting 5s is mathematically incorrect; engine should recommend Double or Hit."""
    _, _, engine = empty_counter_env
    result = engine.get_best_action(player_cards=['5', '5'], dealer_upcard='6')
    assert result['recommended_action'] != 'Split'

def test_best_action_never_split_tens(empty_counter_env):
    """Splitting 10s is generally incorrect; engine should recommend Stand."""
    _, _, engine = empty_counter_env
    result = engine.get_best_action(player_cards=['10', '10'], dealer_upcard='6')
    assert result['recommended_action'] == 'Stand'
    assert 'Split' in result['all_evs']

def test_natural_blackjack_payout(empty_counter_env):
    """Natural blackjack EV should reflect the configured bonus payout (e.g., 3:2 = 1.5)."""
    _, _, engine = empty_counter_env
    result = engine.get_best_action(player_cards=['A', 'K'], dealer_upcard='6')
    assert result['recommended_action'] == 'Stand'
    assert result['expected_value'] >= 1.4  # Approximate lower bound against a 6

def test_no_surrender_in_options(empty_counter_env):
    """If Surrender is disabled in GameConfig, it must not appear in the available actions."""
    shoe, _, engine = empty_counter_env
    shoe.config.surrender = None
    result = engine.get_best_action(player_cards=['10', '6'], dealer_upcard='10')
    assert 'Surrender' not in result['all_evs']

def test_not_first_action_options(empty_counter_env):
    """Double, Split, and Surrender are illegal after the first action (3+ cards)."""
    _, _, engine = empty_counter_env
    result = engine.get_best_action(player_cards=['2', '3', '5'], dealer_upcard='6', is_first_action=False)
    keys = result['all_evs'].keys()
    assert 'Double' not in keys
    assert 'Surrender' not in keys
    assert 'Split' not in keys

def test_ev_split_das_vs_no_das():
    """Expected Value of a Split should be lower if Double After Split (DAS) is disabled."""
    config_das = GameConfig(num_decks=1, das=True)
    engine_das = StrategyEngine(Counter(Shoe(config_das)))
    
    config_no_das = GameConfig(num_decks=1, das=False)
    engine_no_das = StrategyEngine(Counter(Shoe(config_no_das)))

    rem_das = engine_das.counter.get_remaining_cards()
    dist_das = engine_das.get_dealer_distribution('6', rem_das)
    ev_das = engine_das.ev_split('8', dist_das, rem_das)

    rem_no_das = engine_no_das.counter.get_remaining_cards()
    dist_no_das = engine_no_das.get_dealer_distribution('6', rem_no_das)
    ev_no_das = engine_no_das.ev_split('8', dist_no_das, rem_no_das)
    
    assert ev_das > ev_no_das

def test_ev_hit_no_mutation(empty_counter_env):
    """Validates that deep recursive EV calculations do not mutate the original counter state."""
    _, counter, engine = empty_counter_env
    original_seen = dict(counter.seen)

    rem = counter.get_remaining_cards()
    dist = engine.get_dealer_distribution('6', rem)
    engine.ev_hit(12, 0, dist, rem)
    
    assert counter.seen == original_seen

def test_insurance_exact_boundary(empty_counter_env):
    """Insurance requires strict inequality (> 1/3) to be profitable. Exactly 1/3 is a pass."""
    shoe, counter, engine = empty_counter_env
    for rank in counter.seen:
        counter.seen[rank] = 4 * shoe.config.num_decks
        
    counter.seen['10'] -= 1
    counter.seen['2'] -= 2
    assert engine.should_buy_insurance() is False

def test_s17_h17_dealer_distribution_ace():
    """H17 dealers hit on Soft 17, making them less likely to finish exactly on 17 compared to S17."""
    config_s17 = GameConfig(num_decks=1, s17=True)
    engine_s17 = StrategyEngine(Counter(Shoe(config_s17)))
    remaining_s17 =Counter(Shoe(config_s17)).get_remaining_cards()

    config_h17 = GameConfig(num_decks=1, s17=False)
    engine_h17 = StrategyEngine(Counter(Shoe(config_h17)))
    remaining_h17 =Counter(Shoe(config_h17)).get_remaining_cards()

    dist_s17 = engine_s17.get_dealer_distribution('A', remaining_s17)
    dist_h17 = engine_h17.get_dealer_distribution('A', remaining_h17)
    
    assert dist_h17[17] < dist_s17[17]

def test_dealer_distribution_accounts_for_upcard_removal(empty_counter_env):
    """
    Verifies that explicitly tracking a card in the counter alters the dealer's probability distribution,
    proving that the DFS tree dynamically reacts to the global state managed by the screen reader.
    """
    _, counter, engine = empty_counter_env
    rem_before = counter.get_remaining_cards()
    dist_before = engine.get_dealer_distribution('6', rem_before)

    # Simulate the screen reader spotting a '6' and feeding it to the Counter
    counter.update_count('6')
    rem_after = counter.get_remaining_cards()
    dist_after = engine.get_dealer_distribution('6', rem_after)

    assert dist_before != dist_after

def test_dealer_distribution_uses_memo_effectively():
    """
    Performance Test (Smoke Test): 
    Verifies that the memoization cache significantly reduces the DFS time complexity.
    An upcard of '2' creates the deepest possible recursion tree for the dealer.
    Without the memo cache, this calculation could take several seconds or crash.
    With the memo cache, it should complete in a fraction of a second.
    """
    import time
    
    config = GameConfig(num_decks=1, s17=True)
    counter = Counter(Shoe(config))
    engine = StrategyEngine(counter)
    rem = counter.get_remaining_cards()
    
    start_time = time.perf_counter()
    dist = engine.get_dealer_distribution('2', rem)
    elapsed_time = time.perf_counter() - start_time
    
    # Assert the calculation completes under a reasonable threshold (1.0 second).
    # Typically, with memoization, this resolves in < 0.1 seconds.
    assert elapsed_time < 1.0, f"Memoization failed: DFS took {elapsed_time} seconds!"
    
    # Verify the output is still a valid probability distribution summing to ~1.0
    assert math.isclose(sum(dist.values()), 1.0, rel_tol=1e-5)