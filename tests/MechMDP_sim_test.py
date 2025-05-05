import unittest
import pandas as pd
from PSSimPy import Bank, Transaction, Account
from PSSimPy.simulator import ABMSim
from PSSimPy.utils import add_minutes_to_time

from MDP.mech_mdp import MechMDPSearch, MDPStateExt

class TestYourFunctionality(unittest.TestCase):
    
    def setUp(self):
        """Runs before each test method."""
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

        # dummy banks
        self.b1 = Bank('b1', 'MechStrategic')
        self.b2 = Bank('b2', 'MechStrategic')
        self.b3 = Bank('b3', 'MechStrategic')

        # dummy accounts
        self.acc1 = Account('acc1', self.b1, balance=0, posted_collateral=1000)
        self.acc2 = Account('acc2', self.b2, balance=0, posted_collateral=1000)
        self.acc3 = Account('acc3', self.b3, balance=0, posted_collateral=0)

        self.mech_mdp = MechMDPSearch(self.n_players, self.n_periods, self.has_collateral, self.p_t, self.delta, self.delta_prime, self.gamma, self.phi, self.chi, self.zeta, self.seed)

    def create_strategic_bank(self, bank_name, n_players, n_periods, has_collateral, p_t, delta, delta_prime, gamma, phi, chi, zeta, seed):
        mdp = MechMDPSearch(n_players, n_periods, has_collateral, p_t, delta, delta_prime, gamma, phi, chi, zeta, seed)
        class MechStrategicBank(Bank):
            def __init__(self, name, strategy_type='MechStrategic', **kwargs):
                super().__init__(name, strategy_type, **kwargs)
                self.mdp_state = mdp.initial_state() # mdp needs to be redefined before each simulation run
                self.mdp_previous_action = 0
                self.n_periods = 10
            
            # overwrite strategy
            def strategy(self, txns_to_settle: set, all_outstanding_transactions: set, sim_name: str, day: int, current_time: str, queue) -> set:
                if len(txns_to_settle) == 0:
                    return set()
                else:
                    # we assume 1:1 mapping of bank to account so we can just extract any txn and use that account
                    txn = txns_to_settle.copy().pop()
                    bank_account = txn.sender_account

                # calculate amount of obligations that arrived in this period
                arrived_obligations = sum([txn.amount for txn in txns_to_settle if txn.arrival_time == current_time])
                # calculate the amount of claims that arrived in this current period
                observed_claims = sum([txn.amount for txn in all_outstanding_transactions if txn.arrival_time == current_time and txn.recipient_account.owner == self.name])

                if current_time == "08:00":
                    partial_obs = {
                        "inbound_payments": 0,
                        "arrived_obligations": arrived_obligations,
                        "observed_claims": observed_claims,
                        "observed_expected": 0.75  # not used when ζ = 0
                    }

                    self.mdp_state = mdp.update_current_state(self.mdp_state, self.mdp_previous_action, partial_obs)
                else:
                    # calculate actual inbound payments from previous period
                    previous_time = add_minutes_to_time(current_time, -15)
                    df_processed_txns = pd.read_csv(f'{sim_name}-processed_transactions.csv')
                    filtered_df = df_processed_txns[(df_processed_txns['to_account'] == bank_account) & 
                            (df_processed_txns['settlement_time'] == previous_time)]
                    inbound_payments = filtered_df['amount'].sum()

                    partial_obs = {
                        "inbound_payments": inbound_payments,
                        "arrived_obligations": arrived_obligations,
                        "observed_claims": observed_claims,
                        "observed_expected": 0.75  # not used when ζ = 0
                    }

                    self.mdp_state = mdp.update_current_state(self.mdp_state, self.mdp_previous_action, partial_obs)

                _, best_act = mdp.depth_limited_value(self.mdp_state, depth=self.n_periods)
                self.n_periods -= 1
                self.mdp_previous_action = best_act

                if best_act == 1:
                    return txns_to_settle
                else:
                    return set()
        return MechStrategicBank(bank_name)

    def test_strategic_bank(self):
        """
        Tests if a bank implemented to use the MDP Search behaves in a strategic manner as expected.
        """
        # scenario 1: A transaction arrives for the test bank without any incoming transactions. Since it has no collateral, it can only borrow unsecured credit. This is more expensive than delaying so we should see no transactions returned for settlement
        test_bank = self.create_strategic_bank('test bank', 2, 3, False, self.p_t, self.delta, self.delta_prime, self.gamma, self.phi, self.chi, self.zeta, self.seed)
        test_account = Account('test_account', test_bank)
        t1 = Transaction(test_account, Account('N/A', None), 1, time='08:00')
        txns_to_settle = test_bank.strategy({t1}, set(), 'N/A', 1, '08:00', None)
        self.assertEqual(len(txns_to_settle), 0, 'Expected no transactions to be settled')

        # scenario 2: Similar to scenario 1, but since there is collateral, it is cheaper to pay rather than delay so there should be a transaction that is sent for settlement.
        test_bank = self.create_strategic_bank('test bank', 2, 3, True, self.p_t, self.delta, self.delta_prime, self.gamma, self.phi, self.chi, self.zeta, self.seed)
        test_account = Account('test_account', test_bank, posted_collateral=100)
        t2 = Transaction(test_account, Account('N/A', None), 1, time='08:00')
        txns_to_settle = test_bank.strategy({t2}, set(), 'N/A', 1, '08:00', None)
        self.assertEqual(len(txns_to_settle), 1, 'Expected 1 transaction to be settled')
    
    
    # def tearDown(self):
    #     """Runs after each test method."""
    #     # Clean up actions if needed
    #     pass

if __name__ == '__main__':
    unittest.main()
