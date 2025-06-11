import numpy as np
import copy
from itertools import product

from .transaction_path import TransactionPath


def generate_action_sets(n_players: int)->list:
    return np.array(list(product([0, 1], repeat=n_players)))

class FirstBestSearch:

    def __init__(self, accounts: list, delta: float, gamma: float, phi: float, chi: float, transaction_path: TransactionPath, action_sets: np.array, total_costs: int=0, outstanding_txns=[], start_time: str='08:00', end_time: str='10:30', account_states: dict=None):
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
        self.action_sets = action_sets

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

    def transition_period(self, action_set: dict, outstanding_txns: list, current_time: str, account_states: dict=None):
        """
        Step the system forward one settlement tick at `current_time`.
        - action_set: dict of account_id→{0,1}
        - outstanding_txns: list of txns still awaiting settlement at the start of this tick
        - current_time: the time-string for this tick, e.g. "08:00"
        
        Returns:
        - costs: list of per-account costs this period
        - updated_states: dict of account_id→state_dict after payments/delays
        - revised_outstanding: list of txns still outstanding after this tick
        """
        # 1) bring in any newly arrived txns
        new_txns = self.transaction_path.retrieve_txns_by_time(current_time)
        txns = outstanding_txns + new_txns

        # 2) deep-copy account states for this branch
        if account_states is None:
            account_states = self.account_states
        account_states = copy.deepcopy(account_states)
        account_states = self._update_claims(account_states, txns)

        # 3) Borrowing phase for those who will pay
        for acc in self.account_ids:
            if action_set[acc] == 1:
                # sum obligations due from this account
                obligations = sum(txn[3] for txn in txns if txn[1] == acc)
                balance = account_states[acc]['balance']
                shortfall = max(0.0, obligations - balance)
                borrow = self._borrowing_choice(shortfall, account_states[acc], self.gamma, self.phi, self.chi)
                # update borrow records and balance
                for key in ('borrowed_trad', 'borrowed_claim', 'borrowed_unsecured'):
                    account_states[acc][key] += borrow[key]
                account_states[acc]['balance'] += sum(borrow.values())

        # 4) Settlement phase: pay or delay
        revised_outstanding = []
        for txn in txns:
            sender, recipient, amount = txn[1], txn[2], txn[3]
            if action_set[sender] == 1:
                # pay
                account_states[sender]['balance'] -= amount
                account_states[recipient]['balance'] += amount
                account_states[sender]['obligations'] = 0
            else:
                # delay
                account_states[sender]['obligations'] += amount
                revised_outstanding.append(txn)

        # 5) Compute costs this period
        costs = [
            self._log_costs(account_states[acc], self.delta, self.gamma, self.phi, self.chi)
            for acc in self.account_ids
        ]

        # 6) Repay excess liquidity
        for acc in self.account_ids:
            account_states[acc] = self._return_excess_liquidity(
                account_states[acc], self.gamma, self.phi, self.chi
            )

        return costs, account_states, revised_outstanding

    def _update_claims(self, account_states, oustanding_txns):
        for acc_id in self.account_ids:
            claims = sum([txn[3] for txn in oustanding_txns if txn[2]==acc_id])
            account_states[acc_id]['claims'] = claims
        return account_states
    
    def run_search(self):
        """
        Entry point for the full DP search.  Returns the minimum total cost
        over all periods and action sequences.
        """
        # 1) Build the sorted list of settlement times
        #    Assume transaction_path has a method `all_times()` returning all time‐strings in order.
        times = self.transaction_path.all_times(self.start_time, self.end_time)  
        # Or, if you only want between start_time and end_time:
        # times = [t for t in times if self.start_time <= t <= self.end_time]

        # 2) Initialize best_cost
        self.best_cost = float('inf')

        # 3) Start DFS from the first time‐index 0
        init_states = copy.deepcopy(self.account_states)
        init_outstanding = list(self.outstanding_txns)
        self._dfs(0, init_states, init_outstanding, 0.0, times)

        return self.best_cost
    
    def _dfs(self, t_idx, account_states, outstanding_txns, acc_cost, times):
        """
        Recursive DFS over all action_sets.
        """
        # base case
        if t_idx >= len(times):
            self.best_cost = min(self.best_cost, acc_cost)
            return

        time = times[t_idx]
        # for each possible joint action
        for action_row in self.action_sets:
            action_set = {
                acc_id: int(action_row[i]) 
                for i, acc_id in enumerate(self.account_ids)
            }
            # step one period
            costs, new_states, new_outstanding = self.transition_period(
                action_set, outstanding_txns, time, account_states
            )
            period_cost = sum(costs)
 
            # recurse
            self._dfs(
                t_idx + 1,
                new_states,
                new_outstanding,
                acc_cost + period_cost,
                times
            )

    @staticmethod
    def _borrowing_choice(shortfall: float, state: dict,
                        gamma: float, phi: float, chi: float):
        remaining = shortfall
        trad_used = claim_used = unsec_used = 0.0

        # 1) claim-backed if cheapest
        if (phi < gamma or not state['has_collateral']) and (phi < chi):
            avail = max(0.0, state['claims'] - state['borrowed_claim'])
            use  = min(avail, remaining)
            claim_used      = use
            remaining      -= use

        # 2) traditional collateral if available & cheapest
        if state['has_collateral'] and (gamma < chi):
            # optionally clamp by posted_collateral if you record that
            use = min(remaining, state.get('posted_collateral', remaining))
            trad_used       = use
            remaining      -= use

        # 3) unsecured for any remainder (now guaranteed ≥ 0)
        if remaining > 0:
            unsec_used      = remaining
            remaining      -= remaining

        return {
            'borrowed_trad': trad_used,
            'borrowed_claim': claim_used,
            'borrowed_unsecured': unsec_used
        }


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
