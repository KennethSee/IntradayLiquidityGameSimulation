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

    # def _borrowing_choice()

