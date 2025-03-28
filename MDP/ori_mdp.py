from collections import namedtuple
import random

# state: s = (t, b, β, ω, E)
MDPState = namedtuple("MDPState", [
    "t", 
    "balance", 
    "borrowed", 
    "obligations", 
    "expected_inbound"
])

class OriMDPSearch:
    """
    MDP formulation for the focal player's decision in an n-player Intraday Liquidity Game.
    
    State: s = (t, b, β, ω, E), where:
      - t: current period,
      - b: focal player's balance,
      - β: focal player's borrowed amount,
      - ω: focal player's outstanding obligations,
      - E: aggregate expected inbound payment from the n-1 opponents, 
           given by E_t = (n-1)p_tz^*.
    
    Here, p_t is the probability that an obligation arises (per opponent) in period t.
    If an obligation arises, opponents pay the focal player if their dominant strategy is to pay,
    which (with tie → pay) implies z^* = 1 when γ ≤ δ, and 0 when γ > δ.
    Under pure rational expectations (i.e., with ζ = 0), agents do not update their expectations.
    
    The focal player has two strategies:
      0 = Delay: incur delay cost δ per unit of obligation; obligations carry forward.
      1 = Pay: if b < ω, borrow the shortfall at cost γ per unit; clear obligations;
           any excess balance repays borrowing.
           
    The focal player also pays a carry cost of γ·β each period.
    
    We solve the MDP via depth-limited dynamic programming, reflecting bounded rationality.
    """
    
    def __init__(self, 
                 n_players=4,
                 n_periods=4, 
                 p_t=0.8,      # probability an obligation arises per opponent in a period
                 delta=0.2, 
                 gamma=0.1,
                 zeta=0.0,     # learning rate, set to 0 for pure RE
                 seed=42):
        """
        n_players   : total players (focal + opponents)
        n_periods   : horizon for the MDP
        p_t         : probability an obligation arises per opponent
        delta       : per-unit delay cost
        gamma       : cost for borrowing and carrying borrowed funds
        zeta        : learning rate for updating expectations (set to 0 for pure RE)
        seed        : random seed for reproducibility
        """
        random.seed(seed)
        self.n_players = n_players
        self.n_periods = n_periods
        self.p_t = p_t
        self.delta = delta
        self.gamma = gamma
        self.zeta = zeta  # with zeta=0, no updating occurs
        # Dominant strategy assumption: if γ ≤ δ then opponents pay, i.e., z^*=1; otherwise, z^*=0.
        self.z_star = 1.0 if self.gamma <= self.delta else 0.0

    def initial_state(self):
        """
        The initial state: the focal player's balance, borrowed, and obligations are zero.
        The initial aggregate expected inbound is:
           E₀ = (n_players - 1) * p_t * z_star.
        """
        E0 = (self.n_players - 1) * self.p_t * self.z_star
        return MDPState(
            t=0,
            balance=0.0,
            borrowed=0.0,
            obligations=0.0,
            expected_inbound=E0
        )

    def carry_cost(self, borrowed):
        """Cost for carrying borrowed funds over one period."""
        return self.gamma * borrowed if borrowed > 0 else 0.0

    def transition_function(self, state, action):
        """
        Given state s = (t, b, β, ω, E) and focal action a ∈ {0,1},
        return a list of (next_state, probability, immediate_cost).
        
        Here, we assume that in each period, every opponent independently generates an obligation 
        with probability p_t, so that on average, (n_players - 1)*p_t obligations arrive.
        For simplicity, we assume that the number of new obligations is exactly (n-1)p_t.
        
        The focal player's balance is increased by the aggregate expected inbound E,
        and then the chosen action is applied:
         - If a = 1 (Pay): if b + E < ω + (n-1)p_t, the focal borrows the shortfall (cost = γ*(shortfall)); 
           obligations clear.
         - If a = 0 (Delay): cost = δ times the total obligations.
        In both cases, a carry cost γβ is added.
        """
        if state.t >= self.n_periods:
            return [(state, 1.0, 0.0)]
        
        # Compute carry cost
        cost_carry = self.carry_cost(state.borrowed)
        # Inbound: focal's balance increases by E
        mid_balance = state.balance + state.expected_inbound
        # New obligations: exactly (n-1)p_t arrive
        new_arrivals = (self.n_players - 1) * self.p_t
        new_oblig = state.obligations + new_arrivals

        if action == 1:  # Pay
            shortfall = max(0.0, new_oblig - mid_balance)
            cost_borrow = self.gamma * shortfall if shortfall > 0 else 0.0

            new_balance = mid_balance
            new_borrowed = state.borrowed
            if shortfall > 0:
                new_balance = 0.0
                new_borrowed += shortfall
                new_oblig_after = 0.0
            else:
                new_balance = mid_balance - new_oblig
                new_oblig_after = 0.0

            if new_balance > 0 and new_borrowed > 0:
                repay = min(new_balance, new_borrowed)
                new_borrowed -= repay
                new_balance -= repay

            immediate_cost = cost_carry + cost_borrow
            next_state = MDPState(
                t=state.t + 1,
                balance=new_balance,
                borrowed=new_borrowed,
                obligations=new_oblig_after,
                expected_inbound=state.expected_inbound  # under pure RE, remains constant
            )
            return [(next_state, 1.0, immediate_cost)]
        else:  # Delay
            cost_delay = self.delta * new_oblig
            immediate_cost = cost_carry + cost_delay
            next_state = MDPState(
                t=state.t + 1,
                balance=mid_balance,
                borrowed=state.borrowed,
                obligations=new_oblig,
                expected_inbound=state.expected_inbound
            )
            return [(next_state, 1.0, immediate_cost)]

    def actions(self, state):
        """Return the focal player's actions: 0 = Delay, 1 = Pay."""
        return [0, 1]

    def state_to_key(self, state):
        """Convert state into a hashable tuple for memoization."""
        return (state.t, round(state.balance,4), round(state.borrowed,4),
                round(state.obligations,4), round(state.expected_inbound,4))

    def depth_limited_value(self, state, depth, memo=None):
        """
        Depth-limited lookahead from state up to 'depth' periods.
        Returns (best_value, best_action) where best_value is the maximum expected reward 
        (i.e., negative total cost) and best_action ∈ {0,1}.
        
        This recursive algorithm uses memoization. The limited depth represents the 
        bounded rationality of the focal player in forming its rational expectation.
        """
        if memo is None:
            memo = {}
        if depth <= 0 or state.t >= self.n_periods:
            return (0.0, None)
        key = (self.state_to_key(state), depth)
        if key in memo:
            return memo[key]
        best_value = float('-inf')
        best_action = None
        for a in self.actions(state):
            transitions = self.transition_function(state, a)
            total_val = 0.0
            for (ns, prob, cost) in transitions:
                immediate_reward = -cost  # cost is a negative reward
                future_val, _ = self.depth_limited_value(ns, depth - 1, memo)
                total_val += prob * (immediate_reward + future_val)
            if total_val > best_value:
                best_value = total_val
                best_action = a
        memo[key] = (best_value, best_action)
        return memo[key]

    def update_current_state(self, current_state, focal_action, partial_observations):
        """
        Update the current state based on the focal player's chosen action and partial observations.
        
        partial_observations is a dictionary containing:
          - "inbound_payments": actual inbound payment received,
          - "arrived_obligations": number of new obligations that actually arrived,
          - "observed_expected": observed aggregate inbound (e.g., from opponents).
        
        Under our pure rational expectations assumption (with ζ = 0), the aggregate expectation
        remains fixed as E_t = (n-1)p_tz^*. However, to allow for potential deviations, we include 
        an update rule:
        
          E' = ζ · (observed_expected) + (1 - ζ) · E.
          
        Setting ζ = 0 recovers the pure RE case.
        
        Returns the updated state s' = (t+1, b', β', ω', E').
        """
        observed_inbound = partial_observations.get("inbound_payments", 0.0)
        new_balance_pre = current_state.balance + observed_inbound
        arrived = partial_observations.get("arrived_obligations", 0)
        new_obligations = current_state.obligations + arrived
        observed_expected = partial_observations.get("observed_expected", current_state.expected_inbound)
        new_expected = self.zeta * observed_expected + (1 - self.zeta) * current_state.expected_inbound

        if focal_action == 1:  # Pay
            shortfall = max(0.0, new_obligations - new_balance_pre)
            new_borrowed = current_state.borrowed + shortfall
            if shortfall > 0:
                new_balance = 0.0
                new_oblig_after = 0.0
            else:
                new_balance = new_balance_pre - new_obligations
                new_oblig_after = 0.0
            if new_balance > 0 and new_borrowed > 0:
                repay = min(new_balance, new_borrowed)
                new_borrowed -= repay
                new_balance -= repay
            next_state = MDPState(
                t=current_state.t + 1,
                balance=new_balance,
                borrowed=new_borrowed,
                obligations=new_oblig_after,
                expected_inbound=new_expected
            )
            return next_state
        else:  # Delay
            next_state = MDPState(
                t=current_state.t + 1,
                balance=new_balance_pre,
                borrowed=current_state.borrowed,
                obligations=new_obligations,
                expected_inbound=new_expected
            )
            return next_state