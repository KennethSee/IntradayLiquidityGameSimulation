from collections import namedtuple
import random

# ────────────────────────────────────────────────────────────────────
#  State definition: one borrowing bucket only (cost γ)
# ────────────────────────────────────────────────────────────────────
MDPStateTrad = namedtuple("MDPStateTrad", [
    "t",               # current period
    "balance",         # cash balance
    "borrowed",        # outstanding borrowed amount (cost γ)
    "obligations",     # payments the focal bank must make
    "claims",          # payments owed to the focal bank
    "expected_inbound" # aggregate expected inbound payment
])

# ────────────────────────────────────────────────────────────────────
#  MDP class for the traditional regime (no mechanism, one γ)
# ────────────────────────────────────────────────────────────────────
class TradMDPSearch:
    """
    MDP for an n‑bank intraday‑liquidity game with ONLY the traditional
    borrowing channel (cost γ). No pledge‑based borrowing, no unsecured χ.
    """

    def __init__(self,
                 n_players=40,
                 n_periods=40,
                 p_t=0.8,           # prob. an obligation arises per opponent
                 delta=0.20,        # per‑unit delay cost
                 gamma=0.10,        # borrowing cost
                 zeta=0.0,          # learning rate (0 = Rational Expectations)
                 seed=42):
        random.seed(seed)
        self.n_players   = n_players
        self.n_periods   = n_periods
        self.p_t         = p_t
        self.delta       = delta
        self.gamma       = gamma
        self.zeta        = zeta
        self.z_star      = 1.0        # expected settlement rate under RE

    # ───────────────────── initial state ───────────────────────────
    def initial_state(self):
        E0 = (self.n_players - 1) * self.p_t * self.z_star
        return MDPStateTrad(t=0,
                            balance=0.0,
                            borrowed=0.0,
                            obligations=0.0,
                            claims=0.0,
                            expected_inbound=E0)

    # ───────────────────── transition function ─────────────────────
    def transition_function(self, state: MDPStateTrad, action: int):
        """
        action 0 = DELAY,  1 = PAY
        Borrowing (if needed) uses ONLY γ.  Carry cost γ·borrowed each step.
        """
        if state.t >= self.n_periods:
            return [(state, 1.0, 0.0)]

        ω   = state.obligations
        bal = state.balance

        if action == 1:          # PAY
            shortfall     = max(0.0, ω - bal)
            new_borrowed  = state.borrowed + shortfall
            bal_after_pay = 0.0 if shortfall > 0 else bal - ω
        else:                    # DELAY
            shortfall     = 0.0
            new_borrowed  = state.borrowed      # nothing new borrowed
            bal_after_pay = bal                 # no payment made

        # Inbound flows arrive (claims realised in expectation)
        bal_after_in  = bal_after_pay + state.expected_inbound

        # Borrowing cost this period: γ times current borrowed stock
        carry_cost    = new_borrowed * self.gamma

        # Use any excess cash to repay borrowing, most expensive first
        # (only γ exists, so repay whatever is possible)
        repay_amt     = min(bal_after_in, new_borrowed)
        new_borrowed -= repay_amt
        bal_final     = bal_after_in - repay_amt

        # Delay cost if we chose to delay
        delay_cost = self.delta * ω if action == 0 else 0.0
        immediate_cost = carry_cost + delay_cost

        # Update obligations & claims for next period
        new_oblig = (ω if action == 0 else 0.0) + (self.n_players - 1) * self.p_t
        new_claim = state.claims + (self.n_players - 1) * self.p_t

        next_state = MDPStateTrad(
            t               = state.t + 1,
            balance         = bal_final,
            borrowed        = new_borrowed,
            obligations     = new_oblig,
            claims          = new_claim,
            expected_inbound= state.expected_inbound  # constant under RE
        )

        return [(next_state, 1.0, immediate_cost)]

    # ───────────────────── auxiliary helpers ───────────────────────
    def actions(self, state): return [0, 1]

    def state_to_key(self, s):
        return (s.t, round(s.balance,4), round(s.borrowed,4),
                round(s.obligations,4), round(s.claims,4),
                round(s.expected_inbound,4))

    def depth_limited_value(self, state, depth, memo=None):
        if memo is None: memo = {}
        if depth <= 0 or state.t >= self.n_periods:
            return (0.0, None)
        key = (self.state_to_key(state), depth)
        if key in memo: return memo[key]

        best_val, best_act = float('-inf'), None
        for a in self.actions(state):
            exp_val = 0.0
            for ns, p, cost in self.transition_function(state, a):
                r        = -cost
                fut_val, _ = self.depth_limited_value(ns, depth-1, memo)
                exp_val += p * (r + fut_val)
            if exp_val > best_val:
                best_val, best_act = exp_val, a
        memo[key] = (best_val, best_act)
        return memo[key]
