from collections import namedtuple
import random

# Extended state with separate borrowed balances and tracking of claims.
MDPStateExt = namedtuple("MDPStateExt", [
    "t",              # current period
    "balance",        # cash balance
    "borrowed_trad",  # amount borrowed traditionally (cost γ)
    "borrowed_claim", # amount borrowed via pledged incoming (cost φ)
    "borrowed_unsecured", # amount borrowed via unsecured loan (cost χ)
    "obligations",    # outstanding obligations the focal player owes
    "claims",         # outstanding claims (money owed to the focal player)
    "expected_inbound"  # aggregated expected inbound payment from others
])

class MechMDPSearch:
    """
    MDP formulation for the focal player's decision in an n-player Intraday Liquidity Game with a mechanism.
    
    Extended State: 
      s = (t, b, β_trad, β_claim, β_unsecured, ω, C, E),
    where:
      - t: current period,
      - b: focal player's cash balance,
      - β_trad: funds borrowed via traditional channel (cost γ),
      - β_claim: funds borrowed via pledged transactions (cost φ),
      - β_unsecured: funds borrowed via unsecured loans (cost χ),
      - ω: outstanding obligations (already learned for the current period),
      - C: outstanding claims (already learned),
      - E: aggregate expected inbound payment from the n-1 opponents, given by E = (n-1)*p_t (with z* = 1 under pure RE).
    
    The focal player has two strategies:
      0 = Delay: incur delay cost δ per unit (plus additional cost δ′) on outstanding obligations, which then accumulate.
      1 = Pay: attempt to settle all obligations using its current balance. If insufficient,
           borrow the shortfall using the cheaper channel (traditional if γ < φ, or pledged otherwise; if collateral is unavailable, use unsecured).
           After payment, the expected inbound is added to the balance, and any excess cash is used to repay outstanding borrowing.
           
    A carry cost is incurred on borrowed funds (γ for traditional, φ for pledged) each period.
    
    We solve the MDP via depth-limited dynamic programming.
    """
    
    def __init__(self, 
                 n_players=40,
                 n_periods=40, 
                 has_collateral=True,
                 p_t=0.8,       # probability an obligation arises per opponent per period
                 delta=0.2,     # per-unit delay cost
                 delta_prime=0.15,  # additional delay cost imposed by the mechanism
                 gamma=0.1,     # traditional borrowing cost
                 phi=0.05,      # pledged-collateral borrowing cost
                 chi=0.3,       # unsecured borrowing cost
                 zeta=0.0,      # learning rate (set to 0 for pure RE)
                 seed=42):
        random.seed(seed)
        self.n_players = n_players
        self.n_periods = n_periods
        self.p_t = p_t
        self.delta = delta
        self.delta_prime = delta_prime
        self.gamma = gamma
        self.phi = phi
        self.chi = chi
        self.zeta = zeta  # under pure RE, no updating
        self.has_collateral = has_collateral
        # Under our mechanism, we set z* = 1.
        self.z_star = 1.0

    def initial_state(self):
        """
        The initial state: balance, borrowed amounts, obligations, and claims are zero.
        Expected inbound: E₀ = (n_players - 1) * p_t * z_star.
        """
        E0 = (self.n_players - 1) * self.p_t * self.z_star
        return MDPStateExt(
            t=0,
            balance=0.0,
            borrowed_trad=0.0,
            borrowed_claim=0.0,
            borrowed_unsecured=0.0,
            obligations=0.0,
            claims=0.0,
            expected_inbound=E0
        )

    def transition_function(self, state, action):
        """
        Transition function for one period.
        
        We assume that at the decision point, the state has already been updated with:
          - New obligations arrived (ω reflects the current period's obligations).
          - Any updates from the previous period (repayments, claim collections) have been applied.
        
        Then the focal player makes a decision using its current balance b, obligations ω, and claims C.
        
        If action == PAY:
          - Compute shortfall = max(0, ω - b).
          - Borrow the shortfall using the cheaper channel:
               if has_collateral: use pledged if φ < γ and φ < χ; else traditional;
               if no collateral: use unsecured.
          - Payment reduces the balance by ω (or sets balance to 0 if borrowing occurred).
        Then, the expected inbound E is received and added to the balance.
        Borrowing costs and carry costs are computed.
        Any excess cash is used to repay outstanding borrowing.
        Finally, if PAY, obligations are cleared; if DELAY, obligations remain.
        """
        if state.t >= self.n_periods:
            return [(state, 1.0, 0.0)]
        
        # Use the obligations already in state (they have been learned).
        total_oblig = state.obligations
        current_balance = state.balance  # current cash available (before receiving E)
        
        if action == 1:  # PAY
            shortfall = max(0.0, total_oblig - current_balance)
            # Borrowing decision:
            if shortfall > 0:
                remaining_shortfall = shortfall
                add_borrowed_claim = 0.0
                add_borrowed_trad = 0.0
                add_borrowed_unsecured = 0.0
                if (self.phi < self.gamma or not self.has_collateral) and self.phi < self.chi: # this looks wrong, check condition
                    avail_claim = max(0.0, state.claims - state.borrowed_claim)
                    add_borrowed_claim = min(avail_claim, remaining_shortfall)
                    remaining_shortfall -= add_borrowed_claim
                    print(f'remaining shortfall: {remaining_shortfall}')
                if self.has_collateral and self.gamma < self.chi:
                    add_borrowed_trad = remaining_shortfall
                else:
                    add_borrowed_unsecured = remaining_shortfall
            else:
                add_borrowed_trad = 0.0
                add_borrowed_claim = 0.0
                add_borrowed_unsecured = 0.0
            
            new_borrowed_trad = state.borrowed_trad + add_borrowed_trad
            new_borrowed_claim = state.borrowed_claim + add_borrowed_claim
            new_borrowed_unsecured = state.borrowed_unsecured + add_borrowed_unsecured
            
            # Execute payment: if shortfall > 0, balance goes to 0; otherwise, reduce by total_oblig.
            if shortfall > 0:
                balance_after_payment = 0.0
            else:
                balance_after_payment = current_balance - total_oblig
            
            # Now, the expected inbound arrives.
            updated_balance = balance_after_payment + state.expected_inbound
            
            # Immediate cost: borrowing cost (for shortfall) plus carry cost.
            cost_borrow = (new_borrowed_trad * self.gamma + new_borrowed_claim * self.phi + new_borrowed_unsecured * self.chi) if shortfall > 0 else 0.0
            immediate_cost = cost_borrow
            
            # REPAYMENT ASSUMED TO BE EOD
            # Repayment: use any excess cash to repay borrowed funds.
            # total_borrowed = new_borrowed_trad + new_borrowed_claim + new_borrowed_unsecured
            # if updated_balance > 0 and total_borrowed > 0:
            #     repay_amount = min(updated_balance, total_borrowed)
            #     if self.gamma >= self.phi:
            #         repay_trad = min(repay_amount, new_borrowed_trad)
            #         new_borrowed_trad -= repay_trad
            #         repay_amount -= repay_trad
            #         repay_claim = min(repay_amount, new_borrowed_claim)
            #         new_borrowed_claim -= repay_claim
            #         repay_unsecured = min(repay_amount, new_borrowed_unsecured)
            #         new_borrowed_unsecured -= repay_unsecured
            #         updated_balance -= (repay_trad + repay_claim + repay_unsecured)
            #     else:
            #         repay_claim = min(repay_amount, new_borrowed_claim)
            #         new_borrowed_claim -= repay_claim
            #         repay_amount -= repay_claim
            #         repay_trad = min(repay_amount, new_borrowed_trad)
            #         new_borrowed_trad -= repay_trad
            #         repay_unsecured = min(repay_amount, new_borrowed_unsecured)
            #         new_borrowed_unsecured -= repay_unsecured
            #         updated_balance -= (repay_claim + repay_trad + repay_unsecured)
            
            # If PAY, obligations are cleared.
            new_oblig_after = 0.0 + (self.n_players - 1) * self.p_t
            
            next_state = MDPStateExt(
                t = state.t + 1,
                balance = updated_balance,
                borrowed_trad = new_borrowed_trad,
                borrowed_claim = new_borrowed_claim,
                borrowed_unsecured = new_borrowed_unsecured,
                obligations = new_oblig_after,
                claims = state.claims + (self.n_players - 1) * self.p_t,  
                expected_inbound = state.expected_inbound
            )
            return [(next_state, 1.0, immediate_cost)]
        
        else:  # DELAY
            # Under DELAY, obligations remain (accumulate).
            updated_balance = current_balance + state.expected_inbound
            # Delay cost is δ + δ′ per unit.
            cost_delay = (self.delta + self.delta_prime) * total_oblig
            immediate_cost = cost_delay
            new_oblig_after = total_oblig + (self.n_players - 1) * self.p_t
            
            next_state = MDPStateExt(
                t = state.t + 1,
                balance = updated_balance,
                borrowed_trad = state.borrowed_trad,
                borrowed_claim = state.borrowed_claim,
                borrowed_unsecured = state.borrowed_unsecured,
                obligations = new_oblig_after,
                claims = state.claims + (self.n_players - 1) * self.p_t,
                expected_inbound = state.expected_inbound
            )
            return [(next_state, 1.0, immediate_cost)]
    
    def actions(self, state):
        """Return the focal player's actions: 0 = Delay, 1 = Pay."""
        return [0, 1]
    
    def state_to_key(self, state):
        """Convert state into a hashable tuple for memoization."""
        return (state.t, round(state.balance, 4), round(state.borrowed_trad, 4),
                round(state.borrowed_claim, 4), round(state.borrowed_unsecured, 4),
                round(state.obligations, 4), round(state.claims, 4), round(state.expected_inbound, 4))
    
    def depth_limited_value(self, state, depth, memo=None):
        """
        Depth-limited lookahead from state up to 'depth' periods.
        Returns (best_value, best_action) where best_value is the maximum expected reward (i.e., negative total cost)
        and best_action ∈ {0,1}.
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
            print(a, transitions)
            total_val = 0.0
            for (ns, prob, cost) in transitions:
                immediate_reward = -cost  # cost is negative reward
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
          - "arrived_obligations": number of new obligations that arrived,
          - "observed_claims": amount of claims received,
          - "observed_expected": observed aggregate inbound (not used when ζ = 0).
        
        Under pure RE (ζ = 0), expected_inbound remains constant.
        
        If the previous action was PAY, obligations are reset to the newly arrived obligations.
        In addition, if the action was PAY, update borrowed amounts by checking for shortfall and repayment.
        """
        observed_inbound = partial_observations.get("inbound_payments", 0.0)
        observed_claims = partial_observations.get("observed_claims", 0.0)
        new_balance_pre = current_state.balance + observed_inbound
        # Update claim balance: claims accumulate, but if inbound payments cover some unleveraged claims, subtract.
        new_claims_pre = current_state.claims + observed_claims - min(observed_inbound, current_state.claims - current_state.borrowed_claim)
        
        new_borrowed_trad = current_state.borrowed_trad
        new_borrowed_claim = current_state.borrowed_claim
        new_borrowed_unsecured = current_state.borrowed_unsecured
        
        arrived = partial_observations.get("arrived_obligations", 0)
        new_obligations = current_state.obligations + arrived
        observed_expected = partial_observations.get("observed_expected", current_state.expected_inbound)
        new_expected = self.zeta * observed_expected + (1 - self.zeta) * current_state.expected_inbound
        
        # REPAYMENT ASSUMED TO BE EOD
        # Repayment logic: use available cash to repay borrowed amounts.
        # repay_from_cash = new_balance_pre
        # repay_b_claim = min(new_borrowed_claim, repay_from_cash)
        # new_borrowed_claim -= repay_b_claim
        # repay_from_cash -= repay_b_claim
        # repay_b_trad = min(new_borrowed_trad, repay_from_cash)
        # new_borrowed_trad -= repay_b_trad
        # repay_from_cash -= repay_b_trad
        # repay_b_unsecured = min(new_borrowed_unsecured, repay_from_cash)
        # new_borrowed_unsecured -= repay_b_unsecured
        # repay_from_cash -= repay_b_unsecured
        # new_balance_pre = new_balance_pre - (repay_b_claim + repay_b_trad + repay_b_unsecured)
        
        if focal_action == 1:  # PAY
            shortfall = max(0.0, current_state.obligations - new_balance_pre)
            if shortfall > 0:
                remaining_shortfall = shortfall
                add_borrowed_claim = 0.0
                add_borrowed_trad = 0.0
                add_borrowed_unsecured = 0.0
                if (self.phi < self.gamma or not self.has_collateral) and self.phi < self.chi:
                    avail_claim = max(0.0, new_claims_pre - current_state.borrowed_claim)
                    add_borrowed_claim = avail_claim
                    remaining_shortfall -= add_borrowed_claim
                if self.has_collateral and self.gamma < self.chi:
                    add_borrowed_trad = remaining_shortfall
                else:
                    add_borrowed_unsecured = remaining_shortfall
                
                new_borrowed_claim += add_borrowed_claim
                new_borrowed_trad += add_borrowed_trad
                new_borrowed_unsecured += add_borrowed_unsecured
                
                new_balance = 0.0
                new_oblig_after = arrived
            else:
                new_balance = new_balance_pre - current_state.obligations
                new_oblig_after = arrived
            
            next_state = MDPStateExt(
                t = current_state.t + 1,
                balance = new_balance,
                borrowed_trad = new_borrowed_trad,
                borrowed_claim = new_borrowed_claim,
                borrowed_unsecured = new_borrowed_unsecured,
                obligations = new_oblig_after,
                claims = new_claims_pre,
                expected_inbound = new_expected
            )
            return next_state
        
        else:  # DELAY
            next_state = MDPStateExt(
                t = current_state.t + 1,
                balance = new_balance_pre,
                borrowed_trad = new_borrowed_trad,
                borrowed_claim = new_borrowed_claim,
                borrowed_unsecured = new_borrowed_unsecured,
                obligations = new_obligations,
                claims = new_claims_pre,
                expected_inbound = new_expected
            )
            return next_state