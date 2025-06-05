from PSSimPy.utils.time_utils import add_minutes_to_time

class FirstBestSearch:

    def __init__(self, accounts: list, delta: float, gamma: float, phi: float, chi: float, start_time: str='08:00'):
        self.accounts = accounts # expected to be a list of tuples (id, balance, collateral_posted)
        self.account_states = {}
        for account in self.accounts:
            self.initialize_account_state(account)
        self.delta = delta
        self.gamma = gamma
        self.phi = phi
        self.chi = chi
        self.start_time = start_time

    def initialize_account_state(self, account: tuple):
        if account[2] > 0:
            has_collateral = True
        else:
            has_collateral = False

        initial_state = {
            "balance": account[1],        # cash balance
            "borrowed_trad": 0,  # amount borrowed traditionally (cost γ)
            "borrowed_claim": 0, # amount borrowed via pledged incoming (cost φ)
            "borrowed_unsecured": 0, # amount borrowed via unsecured loan (cost χ)
            "obligations": 0,    # outstanding obligations the focal player owes
            "claims": 0,
            "has_collateral": has_collateral
        }
        self.account_states[account[0]] = initial_state

    @staticmethod
    def _borrowing_choice(shortfall: int, state: dict, gamma: float, phi: float, chi: float):
        remaining_shortfall = shortfall
        add_borrowed_trad = 0
        add_borrowed_claim = 0
        add_borrowed_unsecured = 0
        if (phi < gamma or not state['has_collateral']) and phi < chi:
            avail_claim = max(0.0, state['claims'] - state['borrowed_claim'])
            add_borrowed_claim = avail_claim
            remaining_shortfall -= add_borrowed_claim
        if state['has_collateral'] and gamma < chi:
            add_borrowed_trad = remaining_shortfall
        else:
            add_borrowed_unsecured = remaining_shortfall

        borrowings = {'borrowed_trad': add_borrowed_trad, 'borrowed_claim': add_borrowed_claim, 'borrowed_unsecured': add_borrowed_unsecured}
        return borrowings

