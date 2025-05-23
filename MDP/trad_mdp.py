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
    "expected_inbound" # aggregate expected inbound payment
])

# ────────────────────────────────────────────────────────────────────
#  MDP class for the traditional regime (no mechanism, one γ)
# ────────────────────────────────────────────────────────────────────
class TradMDPSearch:
    """
    Traditional‑only MDP (one borrowing cost γ).  The timing of events,
    borrowing, repayment, and expectation updating is *identical* to the
    multi‑channel `MechMDPSearch` you posted—just stripped down to one β.
    """

    # --------------- constructor -------------------------------
    def __init__(self,
                 n_players=40,
                 n_periods=40,
                 p_t=0.8,
                 delta=0.2,
                 gamma=0.1,
                 zeta=0.0,
                 seed=42):
        random.seed(seed)
        self.n_players = n_players
        self.n_periods = n_periods
        self.p_t       = p_t
        self.delta     = delta
        self.gamma     = gamma
        self.zeta      = zeta
        # dominant‑strategy belief of others (tie→pay)
        self.z_star    = 1.0   # always 1.0 in original mech

    # --------------- initial state ----------------------------
    def initial_state(self):
        E0 = (self.n_players - 1) * self.p_t * self.z_star
        return MDPStateTrad(0, 0.0, 0.0, 0.0, E0)

    # --------------- helper: repay ----------------------------
    def repay_outstanding_borrowings(self, s: MDPStateTrad) -> MDPStateTrad:
        """Use cash in s.balance to retire as much of s.borrowed as possible."""
        repay_amt = min(s.balance, s.borrowed)
        return s._replace(balance=s.balance - repay_amt,
                          borrowed=s.borrowed - repay_amt)

    # --------------- transition function ----------------------
    def transition_function(self, state: MDPStateTrad, action: int):
        """
        • obligations ω are already in the state when decision is made
        • PAY logic mirrors your original flow (borrow, pay, inbound, repay)
        • DELAY accumulates obligations
        """
        if state.t >= self.n_periods:
            return [(state, 1.0, 0.0)]

        ω          = state.obligations
        b_current  = state.balance          # cash before expected inbound

        if action == 1:  # = PAY
            shortfall = max(0.0, ω - b_current)
            add_borrow = shortfall          # all shortfall borrowed at γ
            new_borrowed = state.borrowed + add_borrow

            # pay obligations
            b_after_pay = 0.0 if shortfall > 0 else b_current - ω

            # inbound arrives
            b_after_in  = b_after_pay + state.expected_inbound

            # immediate borrowing cost (same formula as original logic)
            cost_borrow = self.gamma * new_borrowed if shortfall > 0 else 0.0

            # obligations cleared and reset for next tick
            ω_next = (self.n_players - 1) * self.p_t
        else:          # = DELAY
            add_borrow    = 0.0
            new_borrowed  = state.borrowed
            b_after_in    = b_current + state.expected_inbound
            cost_borrow   = 0.0
            ω_next        = ω + (self.n_players - 1) * self.p_t

        # build next‑state pre‑repayment
        next_state = MDPStateTrad(
            t               = state.t + 1,
            balance         = b_after_in,
            borrowed        = new_borrowed,
            obligations     = ω_next,
            expected_inbound= state.expected_inbound   # ζ = 0 → constant
        )

        # immediate repayment step (same place as in MechMDP)
        next_state = self.repay_outstanding_borrowings(next_state)

        immediate_cost = cost_borrow
        return [(next_state, 1.0, immediate_cost)]

    # --------------- action set -------------------------------
    def actions(self, _): return [0, 1]

    # --------------- memo key ---------------------------------
    def state_to_key(self, s):
        return (s.t, round(s.balance,4), round(s.borrowed,4),
                round(s.obligations,4), round(s.expected_inbound,4))

    # --------------- depth‑limited solver ---------------------
    def depth_limited_value(self, state, depth, memo=None):
        if memo is None: memo = {}
        if depth <= 0 or state.t >= self.n_periods:
            return (0.0, None)
        key = (self.state_to_key(state), depth)
        if key in memo: return memo[key]

        best_val, best_act = float('-inf'), None
        for a in self.actions(state):
            val = 0.0
            for ns, p, cost in self.transition_function(state, a):
                reward = -cost
                fut, _ = self.depth_limited_value(ns, depth-1, memo)
                val   += p * (reward + fut)
            if val > best_val:
                best_val, best_act = val, a
        memo[key] = (best_val, best_act)
        return memo[key]

    # --------------- run‑time state updater -------------------
    def update_current_state(self, cur: MDPStateTrad, last_action: int, obs: dict):
        """
        Mirrors MechMDP flow:
          • add realised inbound
          • add arrived obligations
          • if last action was PAY, reset obligations to arrivals
          • immediate repayment with any surplus cash
          • expectations updated via ζ‑rule
        """
        inbound   = obs.get("inbound_payments", 0.0)
        arrivals  = obs.get("arrived_obligations", 0)
        obs_E     = obs.get("observed_expected", cur.expected_inbound)

        b = cur.balance + inbound
        β = cur.borrowed

        # obligations update
        if last_action == 1:    # previous period paid
            ω = arrivals
        else:                   # delayed: carry + new
            ω = cur.obligations + arrivals

        # repay from surplus
        repay = min(b, β)
        β    -= repay
        b    -= repay

        # expectation update (ζ = 0 ⇒ stays constant)
        E_next = self.zeta * obs_E + (1 - self.zeta) * cur.expected_inbound

        return MDPStateTrad(cur.t + 1, b, β, ω, E_next)