import os
import glob
import unittest
import numpy as np
import pandas as pd
from FirstBest.first_best_search import FirstBestSearch, generate_action_sets
from SimClasses.abm_sim_new import ABMSim

class TestFirstBestSearch(unittest.TestCase):
    
    def setUp(self):
        """Runs before each test method."""
        self.accounts = [('acc1', 0, 100), ('acc2', 0, 0), ('acc3', 0, 100)]
        self.start_time = '08:00'

    def test_generate_action_sets(self):
        expected = np.array([[0], [1]])
        actual = generate_action_sets(1)
        np.testing.assert_array_equal(actual, expected)

        expected = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        actual = generate_action_sets(2)
        np.testing.assert_array_equal(actual, expected)

        expected = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1]
        ])
        actual = generate_action_sets(3)
        np.testing.assert_array_equal(actual, expected)
    
    def test_initial_state(self):
        fbs = FirstBestSearch(self.accounts, 0.0, 0.0, 0.0, 0.0, self.start_time)
        self.assertEqual(fbs.account_states['acc1']['balance'], 0.0)
        self.assertEqual(fbs.account_states['acc1']['has_collateral'], True)
        self.assertEqual(fbs.account_states['acc2']['has_collateral'], False)

    def test_borrowing_choice(self):
        shortfall = 2
        state = {'balance': 0, 'borrowed_trad': 0, 'borrowed_claim': 1, 'borrowed_unsecured': 0, 'obligations': 0, 'claims': 2, 'has_collateral': True}
        borrowings1 = FirstBestSearch._borrowing_choice(shortfall, state, gamma=0.2, phi=0.3, chi=0.5)
        self.assertEqual(borrowings1['borrowed_trad'], 2)
        self.assertEqual(borrowings1['borrowed_claim'], 0)
        self.assertEqual(borrowings1['borrowed_unsecured'], 0)

        borrowings2 = FirstBestSearch._borrowing_choice(shortfall, state, gamma=0.4, phi=0.3, chi=0.5)
        self.assertEqual(borrowings2['borrowed_trad'], 1)
        self.assertEqual(borrowings2['borrowed_claim'], 1)
        self.assertEqual(borrowings2['borrowed_unsecured'], 0)

        borrowings3 = FirstBestSearch._borrowing_choice(shortfall, state, gamma=0.4, phi=0.3, chi=0.1)
        self.assertEqual(borrowings3['borrowed_trad'], 0)
        self.assertEqual(borrowings3['borrowed_claim'], 0)
        self.assertEqual(borrowings3['borrowed_unsecured'], 2)

        state = {'balance': 0, 'borrowed_trad': 0, 'borrowed_claim': 1, 'borrowed_unsecured': 0, 'obligations': 0, 'claims': 2, 'has_collateral': False}
        borrowings4 = FirstBestSearch._borrowing_choice(shortfall, state, gamma=0.2, phi=0.3, chi=0.5)
        self.assertEqual(borrowings4['borrowed_trad'], 0)
        self.assertEqual(borrowings4['borrowed_claim'], 1)
        self.assertEqual(borrowings4['borrowed_unsecured'], 1)

    def test_log_costs(self):
        state = {'balance': 0, 'borrowed_trad': 1, 'borrowed_claim': 2, 'borrowed_unsecured': 3, 'obligations': 4, 'claims': 2, 'has_collateral': True}
        costs = FirstBestSearch._log_costs(state, delta=0.2, gamma=0.3, phi=0.2, chi=0.5)
        self.assertEqual(costs, 3.0)

    def test_return_excess_liquidity(self):
        state = {'balance': 4, 'borrowed_trad': 1, 'borrowed_claim': 2, 'borrowed_unsecured': 3, 'obligations': 4, 'claims': 2, 'has_collateral': True}
        new_state_1 = FirstBestSearch._return_excess_liquidity(state, gamma=0.3, phi=0.2, chi=0.5)
        self.assertEqual(new_state_1['balance'], 0)
        self.assertEqual(new_state_1['borrowed_trad'], 0)
        self.assertEqual(new_state_1['borrowed_claim'], 2)
        self.assertEqual(new_state_1['borrowed_unsecured'], 0)

        state = {'balance': 4, 'borrowed_trad': 1, 'borrowed_claim': 2, 'borrowed_unsecured': 3, 'obligations': 4, 'claims': 2, 'has_collateral': True}
        new_state_2 = FirstBestSearch._return_excess_liquidity(state, gamma=0.2, phi=0.3, chi=0.5)
        self.assertEqual(new_state_2['balance'], 0)
        self.assertEqual(new_state_2['borrowed_trad'], 1)
        self.assertEqual(new_state_2['borrowed_claim'], 1)
        self.assertEqual(new_state_2['borrowed_unsecured'], 0)

        state = {'balance': 4, 'borrowed_trad': 1, 'borrowed_claim': 2, 'borrowed_unsecured': 3, 'obligations': 4, 'claims': 2, 'has_collateral': True}
        new_state_3 = FirstBestSearch._return_excess_liquidity(state, gamma=0.3, phi=0.2, chi=0.1)
        self.assertEqual(new_state_3['balance'], 0)
        self.assertEqual(new_state_3['borrowed_trad'], 0)
        self.assertEqual(new_state_3['borrowed_claim'], 0)
        self.assertEqual(new_state_3['borrowed_unsecured'], 2)

        state = {'balance': 10, 'borrowed_trad': 1, 'borrowed_claim': 2, 'borrowed_unsecured': 3, 'obligations': 4, 'claims': 2, 'has_collateral': True}
        new_state_4 = FirstBestSearch._return_excess_liquidity(state, gamma=0.3, phi=0.2, chi=0.1)
        self.assertEqual(new_state_4['balance'], 4)
        self.assertEqual(new_state_4['borrowed_trad'], 0)
        self.assertEqual(new_state_4['borrowed_claim'], 0)
        self.assertEqual(new_state_4['borrowed_unsecured'], 0)

        state = {'balance': 1, 'borrowed_trad': 1, 'borrowed_claim': 2, 'borrowed_unsecured': 3, 'obligations': 4, 'claims': 2, 'has_collateral': True}
        new_state_5 = FirstBestSearch._return_excess_liquidity(state, gamma=0.3, phi=0.2, chi=0.1)
        self.assertEqual(new_state_5['balance'], 0)
        self.assertEqual(new_state_5['borrowed_trad'], 0)
        self.assertEqual(new_state_5['borrowed_claim'], 2)
        self.assertEqual(new_state_5['borrowed_unsecured'], 3)

    def tearDown(self):
        """Runs after each test method."""
        # Remove each .csv file
        csv_files = glob.glob("*.csv")
        for file in csv_files:
            os.remove(file)

if __name__ == '__main__':
    unittest.main()