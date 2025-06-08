from PSSimPy.utils.time_utils import add_minutes_to_time
from .transaction_path import TransactionPath

class FirstBestSearch:

    def __init__(self, accounts: list, delta: float, gamma: float, phi: float, chi: float, transaction_path: TransactionPath, total_costs: int=0, outstanding_txns=[], start_time: str='08:00', end_time: str='10:30', account_states: dict=None):
        self.accounts = accounts # expected to be a list of tuples (id, balance, collateral_posted)
        self.account_ids = [x[0] for x in accounts]
        if account_states is None:
            self.account_states = {}
            for account in self.accounts:
                self.initialize_account_state(account)
        else:
            self.account_states = account_states
        self.delta = delta
        self.gamma = gamma
        self.phi = phi
        self.chi = chi
        self.transaction_path = transaction_path
        self.total_costs = total_costs
        self.start_time = start_time
        self.end_time = end_time
        self.outstanding_txns = outstanding_txns

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

    def transition_period(self, action_set: dict):
        # identify transactions that arrived in the period
        outstanding_txns = self.outstanding_txns.copy()
        txns = outstanding_txns.extend(self.transaction_path.retrieve_txns_by_time(self.start_time))

        # perform necessary borrowings
        account_states = self.account_states.copy()
        for acc in self.account_ids:
            if action_set[acc] == 1:
                obligations = sum([txn[3] for txn in txns if txn[1]==acc])
                shortfall = max(0, obligations - account_states[acc]['balance'])
                borrowings = self._borrowing_choice(shortfall, account_states[acc], self.gamma, self.phi, self.chi)
                account_states[acc]['borrowed_trad'] += borrowings['borrowed_trad']
                account_states[acc]['borrowed_claim'] += borrowings['borrowed_claim']
                account_states[acc]['borrowed_unsecured'] += borrowings['borrowed_unsecured']
                account_states[acc]['balance'] += (borrowings['borrowed_trad'] + borrowings['borrowed_claim'] + borrowings['borrowed_unsecured'])

        # execute actions based on action set
        revised_oustanding_txns = []
        for txn in txns:
            sender_acc = txn[1]
            recipient_acc = txn[2]
            amount = txn[3]
            if action_set[sender_acc] == 1: # pay
                account_states[sender_acc]['balance'] -= amount
                account_states[recipient_acc]['balance'] += amount
                account_states[sender_acc]['obligations'] = 0
            else:
                account_states[sender_acc]['obligations'] += amount
                revised_oustanding_txns.append(txn)

        # calculate costs for the period
        costs = [self._log_costs(account_states[acc], self.delta, self.gamma, self.phi, self.chi) for acc in self.account_ids]

        # return any spare liquidity to offset borrowing
        # TO-DO

        return costs

    # def transition_period_account(self, account_state: dict, obligations,)

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

    @staticmethod
    def _return_excess_liquidity(state: dict, gamma: float, phi: float, chi: float):
        # 1) Build dynamic cost map
        cost_map = {
            'borrowed_trad': gamma,
            'borrowed_claim': phi,
            'borrowed_unsecured': chi
        }
        # 2) Sort credit types by descending cost
        repay_order = sorted(cost_map, key=lambda k: cost_map[k], reverse=True)
        # 3) Decide how to repay
        for kind in repay_order:
            repayment_amount = min(state['balance'], state[kind])
            state['balance'] -= repayment_amount
            state[kind] -= repayment_amount

        return state
    
    @staticmethod
    def _log_costs(state: dict, delta, gamma, phi, chi):
        costs = state['obligations'] * delta + state['borrowed_trad'] * gamma + state['borrowed_claim'] * phi + state['borrowed_unsecured'] * chi
        return costs
