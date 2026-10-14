import pytest
import math
from Card import Card
from GameConfig import GameConfig
from Shoe import Shoe
from Counter import Counter
from StrategyEngine import StrategyEngine
from TableManager import TableManager

# ==========================================
# Fixtures
# ==========================================

@pytest.fixture
def empty_counter_env():
    """
    Provides a fresh, isolated game environment before each test.
    Initializes a standard 6-deck shoe with S17, DAS, and dealer peek enabled.
    """
    config = GameConfig(num_decks=6, s17=True, das=True, peek=True)
    shoe = Shoe(config)
    counter = Counter(shoe)
    engine = StrategyEngine(counter)
    return shoe, counter, engine

# ==========================================
# Tests for Global State & Counter
# ==========================================

def test_running_count_updates_correctly(empty_counter_env):
    """Verifies that the Hi-Lo running count increments and decrements correctly based on card ranks."""
    _, counter, _ = empty_counter_env

    for rank in ['2', '3', '4', '5', '6']:
        counter.update_count(rank)
    assert counter.running_count == 5

    for rank in ['7', '8', '9']:
        counter.update_count(rank)
    assert counter.running_count == 5  # Neutral cards do not change the count

    for rank in ['10', 'J', 'Q', 'K', 'A']:
        counter.update_count(rank)
    assert counter.running_count == 0  # High cards decrement the count

def test_true_count_calculation(empty_counter_env):
    """Ensures True Count accurately divides the running count by the remaining decks (rounded to the nearest half)."""
    _, counter, _ = empty_counter_env
    # Simulate an exact distribution that represents exactly 3 decks remaining
    counter.seen.update({
        '7': 24, '8': 24, '9': 24,
        '2': 9, '3': 9, '4': 9, '5': 9, '6': 9,
        '10': 8, 'J': 8, 'Q': 8, 'K': 8, 'A': 7
    })
    assert counter.running_count == 6
    assert counter.true_count == 2

def test_true_count_negative(empty_counter_env):
    """Verifies that a negative running count truncates towards zero mathematically."""
    _, counter, _ = empty_counter_env
    for _ in range(3):
        counter.update_count('K')
    assert counter.true_count == 0

def test_exact_prob_exhausted_rank(empty_counter_env):
    """Ensures the probability of drawing a rank drops to 0.0 once all its instances are dealt."""
    shoe, counter, _ = empty_counter_env
    total_aces = 4 * shoe.config.num_decks
    for _ in range(total_aces):
        counter.update_count('A')
    assert counter.exact_probability_of_rank('A') == 0.0

def test_update_count_rejects_overflow(empty_counter_env):
    """Prevents the counter from registering more physical cards than structurally possible in the shoe."""
    shoe, counter, _ = empty_counter_env
    max_per_rank = 4 * shoe.config.num_decks
    for _ in range(max_per_rank):
        counter.update_count('5')
        
    result = counter.update_count('5')  # Attempt to exceed the physical limit
    assert result is False
    assert counter.seen['5'] == max_per_rank

# ==========================================
# Tests for Strategy Engine
# ==========================================

def test_calculate_new_sum(empty_counter_env):
    """Validates dynamic hand value calculations, especially soft ace downgrades."""
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
    """A hard 20 should overwhelmingly favor standing."""
    _, _, engine = empty_counter_env
    result = engine.get_best_action(player_cards=['10', 'J'], dealer_upcard='6', is_first_action=True)
    assert result['recommended_action'] == 'Stand'

def test_best_action_hard_11(empty_counter_env):
    """A hard 11 against a weak dealer upcard should trigger a Double Down."""
    _, _, engine = empty_counter_env
    result = engine.get_best_action(player_cards=['4', '7'], dealer_upcard='6', is_first_action=True)
    assert result['recommended_action'] == 'Double'

def test_best_action_never_split_fives(empty_counter_env):
    """Basic strategy explicitly forbids splitting 5s, favoring doubling or hitting."""
    _, _, engine = empty_counter_env
    result = engine.get_best_action(player_cards=['5', '5'], dealer_upcard='6')
    assert result['recommended_action'] != 'Split'

def test_best_action_never_split_tens(empty_counter_env):
    """Splitting 10s is generally mathematically inferior to standing on a guaranteed 20."""
    _, _, engine = empty_counter_env
    result = engine.get_best_action(player_cards=['10', '10'], dealer_upcard='6')
    assert result['recommended_action'] == 'Stand'
    assert 'Split' in result['all_evs']

def test_natural_blackjack_payout(empty_counter_env):
    """Natural blackjacks must reflect the configured bonus payout ratio (e.g., 3:2)."""
    _, _, engine = empty_counter_env
    result = engine.get_best_action(player_cards=['A', 'K'], dealer_upcard='6')
    assert result['recommended_action'] == 'Stand'
    assert result['expected_value'] >= 1.4  

def test_no_surrender_in_options(empty_counter_env):
    """If Surrender is disabled via GameConfig, it must be completely excluded from EV evaluations."""
    shoe, _, engine = empty_counter_env
    shoe.config.surrender = None
    result = engine.get_best_action(player_cards=['10', '6'], dealer_upcard='10')
    assert 'Surrender' not in result['all_evs']

def test_not_first_action_options(empty_counter_env):
    """Double Down, Split, and Surrender actions are strictly illegal after the first hit."""
    _, _, engine = empty_counter_env
    result = engine.get_best_action(player_cards=['2', '3', '5'], dealer_upcard='6', is_first_action=False)
    keys = result['all_evs'].keys()
    assert 'Double' not in keys
    assert 'Surrender' not in keys
    assert 'Split' not in keys

def test_ev_split_das_vs_no_das():
    """Splitting should yield a higher Expected Value when Double After Split (DAS) is permitted."""
    config_das = GameConfig(num_decks=1, das=True)
    engine_das = StrategyEngine(Counter(Shoe(config_das)))
    
    config_no_das = GameConfig(num_decks=1, das=False)
    engine_no_das = StrategyEngine(Counter(Shoe(config_no_das)))

    res_das = engine_das.get_best_action(['8', '8'], '6')
    res_no_das = engine_no_das.get_best_action(['8', '8'], '6')
    
    assert res_das['all_evs']['Split'] > res_no_das['all_evs']['Split']

def test_no_double_after_split_if_das_false():
    """Double Down must be blocked on a split hand if DAS is configured as False."""
    config_no_das = GameConfig(num_decks=1, das=False)
    engine = StrategyEngine(Counter(Shoe(config_no_das)))
    result = engine.get_best_action(['6', '5'], '6', is_first_action=True, is_split_hand=True)
    assert 'Double' not in result['all_evs']

def test_split_aces_force_stand(empty_counter_env):
    """Casino rules strictly enforce standing after splitting Aces and receiving exactly one card."""
    _, _, engine = empty_counter_env
    result = engine.get_best_action(player_cards=['A', '7'], dealer_upcard='6', is_first_action=True, is_split_hand=True)
    assert result['recommended_action'] == 'Stand'
    assert 'Double' not in result['all_evs']

def test_ev_hit_no_mutation(empty_counter_env):
    """Deep recursive EV tree evaluations must not mutate or alter the original counter state."""
    _, counter, engine = empty_counter_env
    original_seen = dict(counter.seen)

    rem = counter.get_remaining_cards()
    dist = engine.get_dealer_distribution('6', rem)
    engine.ev_hit(12, 0, dist, rem)
    
    assert counter.seen == original_seen

def test_insurance_exact_boundary(empty_counter_env):
    """Insurance is only mathematically viable when the probability of drawing a ten is strictly > 33.3%."""
    shoe, counter, engine = empty_counter_env
    # Exhaust all cards
    for rank in counter.seen:
        counter.seen[rank] = 4 * shoe.config.num_decks
        
    # Manually configure exact 1/3 probability boundary
    counter.seen['10'] -= 1
    counter.seen['2'] -= 2
    assert engine.should_buy_insurance() is False

def test_s17_h17_dealer_distribution_ace():
    """H17 dealers are less likely to finish exactly on 17 compared to S17 dealers due to forced hits on Soft 17."""
    config_s17 = GameConfig(num_decks=1, s17=True)
    engine_s17 = StrategyEngine(Counter(Shoe(config_s17)))
    
    config_h17 = GameConfig(num_decks=1, s17=False)
    engine_h17 = StrategyEngine(Counter(Shoe(config_h17)))

    dist_s17 = engine_s17.get_dealer_distribution('A')
    dist_h17 = engine_h17.get_dealer_distribution('A')
    
    assert dist_h17[17] < dist_s17[17]

def test_deviation_note_recognizes_face_cards_as_ten(empty_counter_env, monkeypatch):
    """Deviation logic should treat all face cards uniformly as '10' when looking up matrix indices."""
    _, counter, engine = empty_counter_env
    # Force True Count to 1 to trigger the specific deviation
    monkeypatch.setattr(type(counter), 'true_count', property(lambda self: 1))
    
    note_ten = engine.get_deviation_note(['10', '6'], '10', 'Stand')
    note_king = engine.get_deviation_note(['10', '6'], 'K', 'Stand')
    
    assert note_ten != ""
    assert note_ten == note_king

# ==========================================
# Tests for Table Manager & Debouncing
# ==========================================

def test_process_frame_empty_returns_last_result(empty_counter_env):
    """Momentary vision drops (empty frames) must return the cached output to prevent HUD flickering."""
    _, counter, engine = empty_counter_env
    manager = TableManager(counter, engine)
    
    mock_result = {'recommended_action': 'Hit'}
    manager._last_eval_result = mock_result
    
    res = manager.process_frame([], 1000, 1000)
    assert res == mock_result

def test_is_vision_stable_false_when_no_cards_detected(empty_counter_env):
    """Vision stability is inherently False if no cards are actively detected."""
    _, counter, engine = empty_counter_env
    manager = TableManager(counter, engine)
    manager._last_raw_state = (None, ())
    manager._raw_state_counter = 5
    assert manager.is_vision_stable is False

def test_is_vision_stable_true_after_real_detection(empty_counter_env):
    """Vision stability requires valid detection states spanning multiple consecutive frames."""
    _, counter, engine = empty_counter_env
    manager = TableManager(counter, engine)
    manager._last_raw_state = ('10', (('8', '3'),))
    manager._raw_state_counter = 3
    assert manager.is_vision_stable is True

def test_manager_skips_single_card_hands(empty_counter_env, mocker):
    """Ensures hands with fewer than 2 cards (e.g., during active dealing animations) do not trigger evaluations."""
    _, counter, engine = empty_counter_env
    manager = TableManager(counter, engine)
    spy = mocker.spy(engine, 'get_best_action')
    
    manager._stable_dealer = '10'
    manager._stable_hands = [['8']] # Single card hand
    manager._raw_state_counter = 3
    manager.counted_in_round = ['10', '8']
    
    # Mocking geometrical coordinates to simulate a live detection
    mock_boxes = [
        {'rank': '10', 'x': 570, 'y': 240},
        {'rank': '8', 'x': 550, 'y': 400} 
    ]
    
    manager.process_frame(mock_boxes, 1000, 1000)
    assert spy.call_count == 0 

def test_logger_not_called_repeatedly(empty_counter_env, mocker):
    """A specific hand composition should only be logged once per round to prevent CSV spam."""
    _, counter, engine = empty_counter_env
    manager = TableManager(counter, engine)
    mocker.patch.object(manager.logger, 'log_decision')
    
    manager._stable_dealer = '10'
    manager._stable_hands = [['8', '3']]
    manager._raw_state_counter = 3
    manager.counted_in_round = ['10', '8', '3']
    
    mock_boxes = [
        {'rank': '10', 'x': 570, 'y': 240},
        {'rank': '8', 'x': 550, 'y': 400},
        {'rank': '3', 'x': 570, 'y': 400}
    ]
    
    manager.process_frame(mock_boxes, 1000, 1000)
    assert manager.logger.log_decision.call_count == 1
    
    # Force re-evaluation without altering the physical hand
    manager._last_eval_key = None
    manager.process_frame(mock_boxes, 1000, 1000)
    assert manager.logger.log_decision.call_count == 1 

def test_logger_resets_between_rounds(empty_counter_env):
    """The logging registry must be cleared automatically when a round concludes (>30 empty frames)."""
    _, counter, engine = empty_counter_env
    manager = TableManager(counter, engine)
    
    manager._stable_dealer = '10'
    manager._stable_hands = [['8', '3']]
    manager._logged_hands_this_round.add(('10', 0, ('8', '3')))
    
    for _ in range(31): # 30 consecutive empty frames trigger the round reset
        manager.process_frame([], 1000, 1000)
        
    assert len(manager._logged_hands_this_round) == 0

def test_logger_no_reset_on_short_disconnect(empty_counter_env):
    """A short visual disconnect (<=30 frames) must not reset the round state."""
    _, counter, engine = empty_counter_env
    manager = TableManager(counter, engine)
    
    manager._stable_dealer = '10'
    manager._stable_hands = [['8', '3']]
    manager._logged_hands_this_round.add(('10', 0, ('8', '3')))
    
    for _ in range(16): # Only 16 empty frames, below the 30 frame threshold
        manager.process_frame([], 1000, 1000)
        
    assert len(manager._logged_hands_this_round) == 1

def test_partial_clear_no_double_count(empty_counter_env):
    """A partial clear of the table must not cause remaining cards to be double-counted after reset."""
    _, counter, engine = empty_counter_env
    manager = TableManager(counter, engine)
    
    # Initial state: 3 cards counted
    manager.counted_in_round = ['10', '8', '3']
    counter.seen['10'] = 1
    counter.seen['8'] = 1
    counter.seen['3'] = 1
    
    # Simulate sudden drop (only '10' remains) for >15 frames to trigger partial reset
    mock_boxes_partial = [{'rank': '10', 'x': 570, 'y': 240}]
    for _ in range(16):
        manager.process_frame(mock_boxes_partial, 1000, 1000)
        
    # The round resets but preserves '10' in counted_in_round.
    # We must ensure '10' isn't sent to the counter again.
    assert manager.counted_in_round == ['10']
    
    # Process the same remaining card again to ensure stability
    for _ in range(5):
        manager.process_frame(mock_boxes_partial, 1000, 1000)
        
    assert counter.seen['10'] == 1 # Still exactly 1, no double counting

def test_logger_handles_identical_split_hands(empty_counter_env, mocker):
    """If a player splits into two mathematically identical hands, both must be logged independently."""
    _, counter, engine = empty_counter_env
    manager = TableManager(counter, engine)
    mocker.patch.object(manager.logger, 'log_decision')
    
    manager._stable_dealer = '10'
    manager._stable_hands = [['8', '5'], ['8', '5']] 
    manager._raw_state_counter = 3
    manager.counted_in_round = ['10', '8', '5', '8', '5']
    
    mock_boxes = [
        {'rank': '10', 'x': 570, 'y': 240},
        {'rank': '8', 'x': 500, 'y': 400},
        {'rank': '5', 'x': 520, 'y': 400},
        {'rank': '8', 'x': 600, 'y': 400},
        {'rank': '5', 'x': 620, 'y': 400}
    ]
    
    manager.process_frame(mock_boxes, 1000, 1000)
    assert manager.logger.log_decision.call_count == 2

def test_recommendation_not_blocked_by_other_seats(empty_counter_env):
    """
    Ensures cards from external players (outside designated zones) do not lock the evaluation gate
    if the main player's hand and the dealer's upcard are already fully debounced.
    """
    _, counter, engine = empty_counter_env
    manager = TableManager(counter, engine)
    
    manager.counted_in_round = ['10', '8', '3'] 
    manager._stable_dealer = '10'
    manager._stable_hands = [['8', '3']]
    manager._raw_state_counter = 3
    
    mock_boxes = [
        {'rank': '10', 'x': 570, 'y': 240},  # Dealer zone
        {'rank': '8', 'x': 550, 'y': 400},   # Player zone
        {'rank': '3', 'x': 570, 'y': 400},   # Player zone
        {'rank': '7', 'x': 900, 'y': 400},   # External player (pending debounce)
    ]
    
    result = manager.process_frame(mock_boxes, 1000, 1000)
    
    assert result is not None
    assert result['dealer_upcard'] == '10'
    assert len(result['hands_actions']) == 1
    assert result['hands_actions'][0]['cards'] == ['8', '3']