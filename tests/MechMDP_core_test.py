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

    # def test_your_function_type_error(self):
    #     """Test that function raises a TypeError when passed a wrong type."""
    #     with self.assertRaises(TypeError):
    #         your_function("invalid input")
    
    # def test_class_method_behavior(self):
    #     """Test a method from a class."""
    #     obj = YourClass(param=5)
    #     result = obj.method_name()
    #     self.assertTrue(result)  # or use assertEqual/assertAlmostEqual etc.
    
    # def tearDown(self):
    #     """Runs after each test method."""
    #     # Clean up actions if needed
    #     pass

if __name__ == '__main__':
    unittest.main()
