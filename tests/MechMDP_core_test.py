import unittest
from MDP.mech_mdp import MechMDPSearch, MDPStateExt

class TestYourFunctionality(unittest.TestCase):
    
    def setUp(self):
        """Runs before each test method."""
        # Optional setup like input data, test objects
        self.n_players = 3
        self.n_periods = 10
        self.has_collateral = True
        self.p_t = 0.8
        self.delta=0.0
        self.delta_prime = 0.15
        self.gamma = 0.1
        self.phi = 0.05
        self.chi = 0.3
        self.zeta = 0.0
        self.seed = 42

        self.mech_mdp = MechMDPSearch(self.n_players, self.n_periods, self.has_collateral, self.p_t, self.delta, self.delta_prime, self.gamma, self.phi, self.chi, self.zeta, self.seed)

    def test_initial_state(self):
        initial_state = self.mech_mdp.initial_state()
        expected_state = MDPStateExt(0, 0, 0, 0, 0, 0, 0, 1.6)

        self.assertEqual(initial_state, expected_state)  # or whatever is expected
    
    def test_update_current_state(self):
        """Tests that the update_current_state function works"""
        action = 0
        partial_observations_1 = {
                "inbound_payments": 0,
                "arrived_obligations": 2,
                "observed_claims": 1,
                "observed_expected": 0.75  # not used when ζ = 0
        }
        expected_next_state_1 = MDPStateExt(1, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.6)

        initial_state = self.mech_mdp.initial_state()
        state_1 = self.mech_mdp.update_current_state(initial_state, action, partial_observations_1)
        self.assertEqual(state_1, expected_next_state_1)

        action = 1
        partial_observations_2 = {
                "inbound_payments": 0,
                "arrived_obligations": 2,
                "observed_claims": 1,
                "observed_expected": 0.75  # not used when ζ = 0
        }
        expected_next_state_2 = MDPStateExt(2, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 1.6)

        state_2 = self.mech_mdp.update_current_state(state_1, action, partial_observations_2)
        self.assertEqual(state_2, expected_next_state_2)

        action = 0
        partial_observations_3 = {
                "inbound_payments": 2,
                "arrived_obligations": 0,
                "observed_claims": 1,
                "observed_expected": 0.75  # not used when ζ = 0
        }
        expected_next_state_3 = MDPStateExt(3, 2.0, 0.0, 2.0, 0.0, 2.0, 3.0, 1.6)

        state_3 = self.mech_mdp.update_current_state(state_2, action, partial_observations_3)
        self.assertEqual(state_3, expected_next_state_3)

    def test_update_current_status_alternate_unsecured(self):
        """Test case where it should be preferable to borrow unsecured credit"""
        this_mech_mdp = MechMDPSearch(self.n_players, self.n_periods, self.has_collateral, self.p_t, self.delta, self.delta_prime, self.gamma, self.phi, 0.025, self.zeta, self.seed)
        action = 0
        partial_observations_1 = {
                "inbound_payments": 0,
                "arrived_obligations": 2,
                "observed_claims": 1,
                "observed_expected": 0.75  # not used when ζ = 0
        }
        expected_next_state_1 = MDPStateExt(1, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.6)

        initial_state = this_mech_mdp.initial_state()
        state_1 = this_mech_mdp.update_current_state(initial_state, action, partial_observations_1)

        action = 1
        partial_observations_2 = {
                "inbound_payments": 0,
                "arrived_obligations": 2,
                "observed_claims": 1,
                "observed_expected": 0.75  # not used when ζ = 0
        }
        expected_next_state_2 = MDPStateExt(2, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 1.6)

        state_2 = this_mech_mdp.update_current_state(state_1, action, partial_observations_2)
        self.assertEqual(state_2, expected_next_state_2)

    def test_update_current_status_alternate_trad(self):
        """Test case where it should be preferable to borrow traditional credit"""
        this_mech_mdp = MechMDPSearch(self.n_players, self.n_periods, self.has_collateral, self.p_t, self.delta, self.delta_prime, self.gamma, 0.2, self.chi, self.zeta, self.seed)
        action = 0
        partial_observations_1 = {
                "inbound_payments": 0,
                "arrived_obligations": 2,
                "observed_claims": 1,
                "observed_expected": 0.75  # not used when ζ = 0
        }
        expected_next_state_1 = MDPStateExt(1, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.6)

        initial_state = this_mech_mdp.initial_state()
        state_1 = this_mech_mdp.update_current_state(initial_state, action, partial_observations_1)

        action = 1
        partial_observations_2 = {
                "inbound_payments": 0,
                "arrived_obligations": 2,
                "observed_claims": 1,
                "observed_expected": 0.75  # not used when ζ = 0
        }
        expected_next_state_2 = MDPStateExt(2, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 1.6)

        state_2 = this_mech_mdp.update_current_state(state_1, action, partial_observations_2)
        self.assertEqual(state_2, expected_next_state_2)

    def test_transition_function(self):
        initial_state = self.mech_mdp.initial_state()
        partial_observations = {
                "inbound_payments": 0,
                "arrived_obligations": 2,
                "observed_claims": 1,
                "observed_expected": 0.75  # not used when ζ = 0
        }
        next_state = self.mech_mdp.update_current_state(initial_state, 0, partial_observations)

        # test if action = 1
        new_state, _, cost = self.mech_mdp.transition_function(next_state, action=1)[0]
        expected_new_state = MDPStateExt(2, 1.6, 1.0, 1.0, 0.0, 0.0, 1.0, 1.6)
        self.assertEqual(new_state, expected_new_state)
        self.assertEqual(cost, 1 * self.gamma + 1 * self.phi)

        # test if action = 0
        new_state, _, cost = self.mech_mdp.transition_function(next_state, action=0)[0]
        expected_new_state = MDPStateExt(2, 1.6, 0.0, 0.0, 0.0, 2.0, 1.0, 1.6)
        self.assertEqual(new_state, expected_new_state)
        self.assertEqual(cost, 2 * (self.delta + self.delta_prime))
    
    # def tearDown(self):
    #     """Runs after each test method."""
    #     # Clean up actions if needed
    #     pass

if __name__ == '__main__':
    unittest.main()
