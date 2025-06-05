from functools import lru_cache
from itertools import combinations
import math

class FirstBestSearch:
    """
    Exact first‐best DP over a fixed transaction path.

    Parameters
    ----------
    n_players : int
        Number of banks (indexed 0..n_players-1).
    n_periods : int
        Total number of discrete settlement periods, T.
    delta : float
        Per‐unit delay cost.
    gamma : float
        Cost per unit for collateralized borrowing (traditional).
    phi : float
        Cost per unit for claim-backed borrowing.
    chi : float
        Cost per unit for unsecured borrowing.
    txn_path : dict or list of length (T+1)
        Mapping t -> list of (debtor_idx, creditor_idx, amount) for t=1..T.
        (Index 0 is unused or can be empty.)
    initial_collateral : sequence of length n_players
        Each bank i’s posted collateral amount available at period 1.
    """

    def __init__(self,
                 n_players: int,
                 n_periods: int,
                 delta: float,
                 gamma: float,
                 phi: float,
                 chi: float,
                 txn_path,
                 initial_collateral):
        self.n = n_players
        self.T = n_periods
        self.delta = delta
        self.gamma = gamma
        self.phi = phi
        self.chi = chi

        # txn_path[t] should be a list of triples (debtor, creditor, amount)
        # for t in 1..T. Example: txn_path[3] = [(0,2,1.0), (1,0,0.5), ...]
        self.txn_path = txn_path

        # Each bank’s posted collateral at the start of period 1
        if len(initial_collateral) != n_players:
            raise ValueError("initial_collateral must have length n_players")
        self.initial_collateral = tuple(initial_collateral)

        # Precompute for quick lookup: at each t, each bank’s total
        # obligation and the “who owes whom” mapping
        # obligations[t][i] = sum of amounts debtor i owes in period t
        # creditors[t][i]   = list of (creditor, amount) pairs for i’s obligations
        self.obligations = [ [0.0]*n_players for _ in range(self.T+1) ]
        self.creditors   = [ [[] for _ in range(n_players)] for _ in range(self.T+1) ]
        for t in range(1, self.T+1):
            for (deb, cred, amt) in self.txn_path[t]:
                self.obligations[t][deb] += amt
                self.creditors[t][deb].append((cred, amt))

        # We will memoize the DP in dp_min_cost(state, t).
        # State is a tuple of length n_players, each entry is a 4‐tuple:
        #   (balance, obligation, collateral, claims)
        # All floats, so round them (or use them exactly if deterministic).
        # We store state as a tuple of tuples:
        #   state = ((b0, w0, coll0, c0), (b1, w1, coll1, c1), ..., (b_{n-1}, w_{n-1}, coll_{n-1}, c_{n-1}))
        # where each bank i’s w_i = outstanding obligation at start of period t,
        #   coll_i = posted collateral still available,
        #   c_i = unpledged claim balance at start of period t,
        #   b_i = current cash balance at start of period t.
        # Initially, all b_i = 0, c_i = 0, w_i = obligations[1][i], coll_i = initial_collateral[i].

    def build_initial_state(self):
        """Return the period‐1 state as a tuple of (b_i, ω_i, coll_i, claim_i) for i=0..n-1."""
        state0 = []
        for i in range(self.n):
            b0 = 0.0
            w0 = self.obligations[1][i]
            coll0 = self.initial_collateral[i]
            claim0 = 0.0
            state0.append((b0, w0, coll0, claim0))
        return tuple(state0)

    def cheapest_feasible(self, bank_state):
        """
        Given a single bank’s state (b_i, w_i, coll_i, claim_i), return
        a sorted list of feasible channels:
          [ (ch_name, marginal_cost, capacity), ... ]
        in ascending order of marginal_cost, but only if capacity > 0.
        ch_name ∈ {"trad","claim","unsec","delay"}.
        """
        (_, w_i, coll_i, claim_i) = bank_state
        options = []

        # 1) Traditional collateral up to coll_i at cost gamma
        if coll_i > 0:
            options.append(("trad", self.gamma, coll_i))

        # 2) Claim-backed up to claim_i at cost phi
        if claim_i > 0:
            options.append(("claim", self.phi, claim_i))

        # 3) Unsecured is always possible, infinite capacity
        options.append(("unsec", self.chi, math.inf))

        # 4) Delay (always possible, infinite capacity) at cost delta
        options.append(("delay", self.delta, math.inf))

        # Sort by marginal cost then by channel‐name (for deterministic tie‐breaking)
        options.sort(key=lambda x: (x[1], x[0]))
        return options

    def allocate_payment(self, bank_state):
        """
        Given (b_i, w_i, coll_i, claim_i), return a small helper list of
        four values: (trad_used, claim_used, unsec_used, delay_flag),
        indicating how the bank would cover w_i if it must pay *now* under
        the cheapest‐feasible rule.  The planner uses this to compute cost.
        """
        (b_i, w_i, coll_i, claim_i) = bank_state
        if w_i <= 0:
            return (0.0, 0.0, 0.0, False)  # nothing to pay

        # Try each feasible channel in ascending cost order
        channels = self.cheapest_feasible(bank_state)
        remaining = w_i
        trad_used = claim_used = unsec_used = 0.0

        for (ch_name, cost, cap) in channels:
            if cost > self.delta:
                # Now cheaper to delay than to borrow at cost > delta
                # => must “delay” (so this bank cannot pay in this period)
                return (0.0, 0.0, 0.0, True)

            # Otherwise borrow from this channel up to min(cap, remaining)
            take = min(cap, remaining)
            if take > 0:
                if ch_name == "trad":
                    trad_used = take
                elif ch_name == "claim":
                    claim_used = take
                elif ch_name == "unsec":
                    unsec_used = take
                # reduce remaining
                remaining -= take

            if remaining <= 1e-12:
                break

        # At this point, if remaining > 0, it means all feasible channels
        # had cost <= delta, but total capacity from coll_i+claim_i+∞ exceeded w_i,
        # so we covered it.  No delay_flag needed.
        return (trad_used, claim_used, unsec_used, False)

    def period_transition(self, state_t, payers, period_t):
        """
        Given state_t = tuple of length n:  (b_i, w_i, coll_i, claim_i)
        and a chosen set of payers (list of bank‐indices), compute:

          (immediate_cost, next_state)

        1) immediate_cost = sum_{i in payers} [γ·trad_i + φ·claim_i + χ·unsec_i]
                         + sum_{i not in payers} [δ·ω_i]

        2) next_state for each i:
           - If i in payers:
             • new_balance = 0 (they spend exactly ω_i to pay)
             • new_obligation = 0   (they cleared it)
             • new_collateral = coll_i - trad_used
             • new_claims = claim_i - claim_used
           - If i not in payers:
             • new_balance = b_i (unchanged)
             • new_obligation = w_i + obligations[t+1][i]
             • new_collateral = coll_i (unchanged)
             • new_claims = claim_i (unchanged)

           Finally, after deciding who paid, we collect “payments_to[j]”
           from each i’s payment and add that amount to j’s “new_claims” at t+1.

        Returns:
          immediate_cost : float
          next_state : tuple of length n of (b_i', w_i', coll_i', claim_i')
        """
        n = self.n
        # Unpack state_t
        b = [0.0]*n
        w = [0.0]*n
        coll = [0.0]*n
        claim = [0.0]*n
        for i in range(n):
            b[i], w[i], coll[i], claim[i] = state_t[i]

        # 1) Compute immediate cost
        immediate_cost = 0.0
        # Track for each i how much they borrowed from each channel
        trad_used  = [0.0]*n
        claim_used = [0.0]*n
        unsec_used = [0.0]*n
        delay_flag = [False]*n

        # a) For each i in payers, allocate payment
        for i in payers:
            (t_u, c_u, u_u, d_flag) = self.allocate_payment((b[i], w[i], coll[i], claim[i]))
            if d_flag:
                # If they should have delayed (because delta < cheapest cost),
                # treat them as a delayer instead
                delay_flag[i] = True
                continue
            trad_used[i], claim_used[i], unsec_used[i] = (t_u, c_u, u_u)
            cost_i = self.gamma*t_u + self.phi*c_u + self.chi*u_u
            immediate_cost += cost_i

        # b) For each i not in payers OR those flagged to delay, pay delay cost
        for i in range(n):
            if i not in payers or delay_flag[i]:
                immediate_cost += self.delta * w[i]
                delay_flag[i] = True

        # 2) Build next_state entries and track payments_to[j]
        payments_to = [0.0]*n
        next_state = [None]*n

        # a) Payers who succeeded (delay_flag[i] == False) send payment w[i] to their creditor(s)
        for i in payers:
            if delay_flag[i]:
                # This payer is treated as delayer
                continue

            # They pay w[i] in full.  We need to know whom they owed.
            # Each obligation from i in period t gets credited to one or more creditors:
            # self.creditors[t][i] is a list of (creditor_idx, amount) pairs.
            for (j, amt) in self.creditors[period_t][i]:
                payments_to[j] += amt

            # Now construct new state for bank i
            new_b_i = 0.0
            new_w_i = 0.0
            new_coll_i = coll[i] - trad_used[i]
            new_claim_i = claim[i] - claim_used[i]
            next_state[i] = (new_b_i, new_w_i, new_coll_i, new_claim_i)

        # b) Delayers keep their prior balance and collateral/claims, but carry obligations forward
        for i in range(n):
            if not (i in payers and not delay_flag[i]):
                # i is a delayer
                new_b_i = b[i]
                new_w_i = w[i] + ( self.obligations[period_t+1][i] if period_t+1 <= self.T else 0.0 )
                new_coll_i = coll[i]
                new_claim_i = claim[i]
                next_state[i] = (new_b_i, new_w_i, new_coll_i, new_claim_i)

        # c) Distribute inbound payments_to[*] as new claims at t+1
        for j in range(n):
            (b_j, w_j, coll_j, claim_j) = next_state[j]
            new_claim_j = claim_j + payments_to[j]
            next_state[j] = (b_j, w_j, coll_j, new_claim_j)

        return immediate_cost, tuple(next_state)

    @lru_cache(maxsize=None)
    def dp_min_cost(self, state_t, t):
        """
        Recursively compute the minimum total cost from period t..T given
        that the current system state is `state_t` at the start of period t.
        Returns (min_cost, optimal_subset_of_payers).
        """
        # Base case
        if t > self.T:
            return 0.0, frozenset()

        n = self.n
        best_cost = float("inf")
        best_subset = frozenset()

        # Enumerate all 2^n subsets of banks as possible payers
        # Represent each subset as a bitmask from 0..(2^n - 1).
        # If bit i is set, then i is in the payer set S.
        for mask in range(1 << n):
            payers = [i for i in range(n) if (mask & (1 << i))]

            # Feasibility check: if any i in payers finds cheapest feasible cost > delta,
            # then i should have delayed, so skip this subset entirely.
            feasible = True
            for i in payers:
                (b_i, w_i, coll_i, claim_i) = state_t[i]
                if w_i <= 0:
                    continue  # no obligation to cover
                # find cheapest feasible cost
                cheapest = math.inf
                if coll_i > 0:
                    cheapest = min(cheapest, self.gamma)
                if claim_i > 0:
                    cheapest = min(cheapest, self.phi)
                cheapest = min(cheapest, self.chi)
                if cheapest > self.delta:
                    feasible = False
                    break
            if not feasible:
                continue

            # Compute immediate cost and next state
            c0, next_state = self.period_transition(state_t, payers, t)
            future_cost, _ = self.dp_min_cost(next_state, t+1)
            total_cost = c0 + future_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_subset = frozenset(payers)

        return best_cost, best_subset

    def compute_min_cost(self):
        """
        Build the initial state from period 1 and call the DP.  
        Returns the scalar minimum total cost from t=1..T.
        """
        initial_state = self.build_initial_state()
        min_cost, _ = self.dp_min_cost(initial_state, 1)
        return min_cost
