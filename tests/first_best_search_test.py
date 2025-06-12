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
        fbs = FirstBestSearch(self.accounts, 0.0, 0.0, 0.0, 0.0, None, None, self.start_time)
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

    def test_updates_claims(self):
        state1 = {'balance': 1, 'borrowed_trad': 1, 'borrowed_claim': 2, 'borrowed_unsecured': 3, 'obligations': 4, 'claims': 2, 'has_collateral': True}
        state2 = {'balance': 1, 'borrowed_trad': 1, 'borrowed_claim': 2, 'borrowed_unsecured': 3, 'obligations': 4, 'claims': 2, 'has_collateral': True}
        states = {'acc1': state1, 'acc2': state2}
        outstanding_txns = [('08:00', 'acc1', 'acc2', 1), ('08:15', 'acc1', 'acc2', 1), ('08:30', 'acc1', 'acc2', 1)]
        fbs = FirstBestSearch([('acc1', 0, 0), ('acc2', 0, 0)], 0.0, 0.0, 0.0, 0.0, None, None, self.start_time)
        new_state = fbs._update_claims(states, outstanding_txns)
        self.assertEqual(new_state['acc1']['claims'], 0)
        self.assertEqual(new_state['acc2']['claims'], 3)

    def test_run_search(self):
        class DummyTxnPath:
            """A fake transaction path for testing."""
            def __init__(self, times, txns_by_time):
                # times: list of time strings
                # txns_by_time: dict time_str → list of (time, from_acc, to_acc, amount)
                self._times = times
                self._map   = txns_by_time

            def all_times(self, start_time, end_time):
                return list(self._times)

            def retrieve_txns_by_time(self, t):
                return list(self._map.get(t, []))
            
        action_sets = generate_action_sets(3)
        
        # test no obligations
        times = ['08:00', '08:15']
        txns_by_time = {
            '08:00': [],
            '08:15': []
        }
        txn_path_no_oblig = DummyTxnPath(times, txns_by_time)

        fbs = FirstBestSearch(
            accounts        = self.accounts,
            delta           = 0.3,
            gamma           = 0.1,
            phi             = 0.1,
            chi             = 0.1,
            transaction_path= txn_path_no_oblig,
            action_sets     = action_sets,
            outstanding_txns= []
        )
        best = fbs.run_search()
        self.assertEqual(round(best, 1), 0.0)

        # # test with obligations
        times = ['08:00', '08:15']
        txns_by_time = {
            '08:00': [('08:00', 'acc1', 'acc2', 1)],
            '08:15': [('08:15', 'acc3', 'acc2', 1)]
        }
        txn_path_with_oblig = DummyTxnPath(times, txns_by_time)
        fbs = FirstBestSearch(
            accounts        = self.accounts,
            delta           = 0.3,
            gamma           = 0.0,
            phi             = 0.0,
            chi             = 0.0,
            transaction_path= txn_path_with_oblig,
            action_sets     = action_sets,
            outstanding_txns= []
        )
        best = fbs.run_search()
        self.assertEqual(round(best, 1), 0.0)

        txn_path_with_oblig = DummyTxnPath(times, txns_by_time)
        fbs = FirstBestSearch(
            accounts        = self.accounts,
            delta           = 0.3,
            gamma           = 0.1,
            phi             = 0.0,
            chi             = 0.2,
            transaction_path= txn_path_with_oblig,
            action_sets     = action_sets,
            outstanding_txns= []
        )
        best = fbs.run_search()
        self.assertEqual(round(best, 1), 0.3)

        txns_by_time = {
            '08:00': [('08:00', 'acc1', 'acc2', 1), ('08:00', 'acc2', 'acc1', 1)],
            '08:15': [('08:15', 'acc3', 'acc2', 1)]
        }
        txn_path_with_oblig = DummyTxnPath(times, txns_by_time)
        fbs = FirstBestSearch(
            accounts        = self.accounts,
            delta           = 0.3,
            gamma           = 0.2,
            phi             = 0.4,
            chi             = 0.5,
            transaction_path= txn_path_with_oblig,
            action_sets     = action_sets,
            outstanding_txns= []
        )
        best = fbs.run_search()
        self.assertEqual(round(best, 1), 0.8)

    def tearDown(self):
        """Runs after each test method."""
        # Remove each .csv file
        csv_files = glob.glob("*.csv")
        for file in csv_files:
            os.remove(file)

if __name__ == '__main__':
    unittest.main()