import os
import glob
import unittest
import pandas as pd
from FirstBest.transaction_path import TransactionPath
from SimClasses.abm_sim_new import ABMSim

class TestTransactionPath(unittest.TestCase):
    
    def setUp(self):
        """Runs before each test method."""
        self.transaction_path = TransactionPath()
        self.banks = {'name': ['b1', 'b2', 'b3', 'b4']}
        self.accounts = {'id': ['acc1', 'acc2', 'acc3', 'acc4'], 'owner': ['b1', 'b2', 'b3', 'b4'], 'balance': [0, 0, 0, 0]}
        self.transactions = {'sender_account': ['acc1', 'acc2', 'acc3', 'acc4'], 'recipient_account': ['acc2', 'acc3', 'acc4', 'acc1'], 'amount': [10, 10, 10, 10], 'time': ['08:00', '08:00', '08:30', '08:30']}

    def test_initial_state(self):
        self.assertEqual(self.transaction_path.txns_list, [])
        self.assertEqual(self.transaction_path.txns_by_time, {})

    def test_txn_building(self):
        sim_name = 'test_abm_sim'
        sim = ABMSim(sim_name, banks=self.banks, accounts=self.accounts, transactions=self.transactions, open_time='08:00', close_time='09:00')
        sim.run()

        expected_txns_list = [('08:00', 'acc1', 'acc2', 10), ('08:00', 'acc2', 'acc3', 10), ('08:30', 'acc3', 'acc4', 10), ('08:30', 'acc4', 'acc1', 10)]
        expected_txns_dict = {'08:00': [('08:00', 'acc1', 'acc2', 10), ('08:00', 'acc2', 'acc3', 10)], '08:30': [('08:30', 'acc3', 'acc4', 10), ('08:30', 'acc4', 'acc1', 10)]}

        # check that transaction path is generated correctly
        df = pd.read_csv(f'{sim_name}-processed_transactions.csv')
        self.transaction_path.extract_txns_from_df(df)
        self.assertCountEqual(self.transaction_path.txns_list, expected_txns_list)
        self.assertCountEqual(self.transaction_path.txns_by_time['08:00'], expected_txns_dict['08:00'])

    def test_generated_txn_reading(self):
        sim_name = 'test_abm_randtxn_sim'
        sim = ABMSim(sim_name, banks=self.banks, accounts=self.accounts, open_time='08:00', close_time='09:00', txn_amount_range=(1,1), txn_arrival_prob=0.5)
        sim.run()

        # check that transaction path is able to read randomly-generated transactions
        df = pd.read_csv(f'{sim_name}-transactions_arrival.csv')
        self.transaction_path.extract_txns_from_df(df)
        self.assertGreater(len(self.transaction_path.txns_list), 0)

    def tearDown(self):
            """Runs after each test method."""
            # clear transaction path
            self.transaction_path = TransactionPath()

            # Remove each .csv file
            csv_files = glob.glob("*.csv")
            for file in csv_files:
                os.remove(file)

if __name__ == '__main__':
    unittest.main()