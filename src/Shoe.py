class Shoe:
    """
    A lightweight container for the shoe configuration constraints,
    used to calculate total cards and cut-card penetration points.
    Does not manage physical card objects, as the Counter serves as the single source of truth.
    """
    def __init__(self, config):
        self.config = config
        self.total_initial_cards = self.config.get_total_cards()
        self.cut_card_threshold = self.config.get_cut_card_threshold()