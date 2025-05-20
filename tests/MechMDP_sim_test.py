import os
import glob
import unittest
import pandas as pd
from PSSimPy import Bank, Transaction, Account
from PSSimPy.simulator import ABMSim
from PSSimPy.credit_facilities import AbstractCreditFacility
from PSSimPy.utils import add_minutes_to_time
from typing import List

from MDP.mech_mdp import MechMDPSearch, MDPStateExt

class TestYourFunctionality(unittest.TestCase):
    
    def setUp(self):
        """Runs before each test method."""
        self.n_players = 3
        self.n_periods = 10
        self.has_collateral = True
        self.p_t = 1.0
        self.delta=0.0
        self.delta_prime = 0.15
        self.gamma = 0.1
        self.phi = 0.05
        self.chi = 0.3
        self.zeta = 0.0
        self.seed = 42

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
                observed_claims = sum([txn.amount for txn in all_outstanding_transactions if txn.arrival_time == current_time and txn.recipient_account.owner.name == self.name])

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

                best_value, best_act = mdp.depth_limited_value(self.mdp_state, depth=self.n_periods)
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
        self.assertEqual(len(txns_to_settle), 0, 'Scenario 1: Expected no transactions to be settled')

        # scenario 2: Similar to scenario 1, but since there is collateral, it is cheaper to pay rather than delay so there should be a transaction that is sent for settlement.
        test_bank = self.create_strategic_bank('test bank', 2, 3, True, self.p_t, self.delta, self.delta_prime, self.gamma, self.phi, self.chi, self.zeta, self.seed)
        test_account = Account('test_account', test_bank, posted_collateral=100)
        t2 = Transaction(test_account, Account('N/A', None), 1, time='08:00')
        txns_to_settle = test_bank.strategy({t2}, set(), 'N/A', 1, '08:00', None)
        self.assertEqual(len(txns_to_settle), 1, 'Scenario 2: Expected 1 transaction to be settled')

        # scenario 3: A transaction arrives for the test bank, with no collateral, with another incoming transactions. Since borrowing costs is still cheaper than delaying, there should be a transaction sent for settlement.
        test_bank = self.create_strategic_bank('test bank', 2, 3, False, self.p_t, self.delta, self.delta_prime, self.gamma, self.phi, self.chi, self.zeta, self.seed)
        test_bank_other = self.create_strategic_bank('test bank other', 2, 3, False, self.p_t, self.delta, self.delta_prime, self.gamma, self.phi, self.chi, self.zeta, self.seed)
        test_account = Account('test_account', test_bank, posted_collateral=0)
        test_account_other = Account('test_account_other', test_bank_other, posted_collateral=0)
        t3 = Transaction(test_account, test_account_other, 1, time='08:00')
        t4 = Transaction(test_account_other, test_account, 1, time='08:00')
        txns_to_settle = test_bank.strategy({t3}, {t3, t4}, 'N/A', 1, '08:00', None)
        self.assertEqual(len(txns_to_settle), 1, 'Scenario 3: Expected 1 transaction to be settled')

        # scenario 4: Similar to scenrio 3 but with two obligations and two incoming transactions. There should be two transactions sent for settlement.
        test_bank = self.create_strategic_bank('test bank', 2, 3, False, self.p_t, self.delta, self.delta_prime, self.gamma, self.phi, self.chi, self.zeta, self.seed)
        test_bank_other_1 = self.create_strategic_bank('test bank other 1', 2, 3, False, self.p_t, self.delta, self.delta_prime, self.gamma, self.phi, self.chi, self.zeta, self.seed)
        test_bank_other_2 = self.create_strategic_bank('test bank other 2', 2, 3, False, self.p_t, self.delta, self.delta_prime, self.gamma, self.phi, self.chi, self.zeta, self.seed)
        test_account = Account('test_account', test_bank, posted_collateral=0)
        test_account_other_1 = Account('test_account_other_1', test_bank_other_1, posted_collateral=0)
        test_account_other_2 = Account('test_account_other_2', test_bank_other_2, posted_collateral=0)
        t5 = Transaction(test_account, test_account_other_1, 1, time='08:00')
        t6 = Transaction(test_account, test_account_other_2, 1, time='08:00')
        t7 = Transaction(test_account_other_1, test_account, 1, time='08:00')
        t8 = Transaction(test_account_other_2, test_account, 1, time='08:00')
        txns_to_settle = test_bank.strategy({t5, t6}, {t5, t6, t7, t8}, 'N/A', 1, '08:00', None)
        self.assertEqual(len(txns_to_settle), 2, 'Scenario 4: Expected 2 transactions to be settled')

        # scenario 5: Multiple periods and multiple transactions
        # test_bank = self.create_strategic_bank('test bank', 3, 3, False, self.p_t, self.delta, self.delta_prime, self.gamma, self.phi, self.chi, self.zeta, self.seed)
        # test_bank_other_1 = self.create_strategic_bank('test bank other 1', 2, 3, False, self.p_t, self.delta, self.delta_prime, self.gamma, self.phi, self.chi, self.zeta, self.seed)
        # test_bank_other_2 = self.create_strategic_bank('test bank other 2', 2, 3, False, self.p_t, self.delta, self.delta_prime, self.gamma, self.phi, self.chi, self.zeta, self.seed)
        # test_account = Account('test_account', test_bank, posted_collateral=0)
        # test_account_other_1 = Account('test_account_other_1', test_bank_other_1, posted_collateral=0)
        # test_account_other_2 = Account('test_account_other_2', test_bank_other_2, posted_collateral=0)
        # t9 = Transaction(test_account, test_account_other_1, 1, time='08:00')
        # t10 = Transaction(test_account, test_account_other_2, 1, time='08:00')
        # t11 = Transaction(test_account_other_1, test_account, 1, time='08:00')
        # t12 = Transaction(test_account_other_2, test_account, 1, time='08:00')
        # txns_to_settle = test_bank.strategy({t9, t10}, {t9, t10, t11, t12}, 'N/A', 1, '08:00', None)
        # self.assertEqual(len(txns_to_settle), 2, 'Expected 2 transactions to be settled')
        # t13 = Transaction(test_account, test_account_other_1, 1, time='08:15')
        # t14 = Transaction(test_account, test_account_other_2, 1, time='08:15')
        # t15 = Transaction(test_account_other_1, test_account, 1, time='08:15')
        # t16 = Transaction(test_account_other_2, test_account, 1, time='08:15')
        # txns_to_settle = test_bank.strategy({t13, t14}, {t13, t14, t15, t16}, 'N/A', 1, '08:15', None)
        # self.assertEqual(len(txns_to_settle), 2, 'Scenario 5: Expected 2 transactions to be settled')

    # def test_sim_delay(self):
    #     """
    #     Tests if expected delayed transactions are correctly being delayed in simulation.
    #     """
    #     mdp = MechMDPSearch(3, 3, False, 1, 0.0, 0.1, 0.4, 0.4, 0.5, 1, seed=42)
    #     class MechStrategicBank(Bank):
    #         def __init__(self, name, strategy_type='MechStrategic', **kwargs):
    #             super().__init__(name, strategy_type, **kwargs)
    #             self.mdp_state = mdp.initial_state() # mdp needs to be redefined before each simulation run
    #             self.mdp_previous_action = 0
    #             self.n_periods = 10
            
    #         # overwrite strategy
    #         def strategy(self, txns_to_settle: set, all_outstanding_transactions: set, sim_name: str, day: int, current_time: str, queue) -> set:
    #             if len(txns_to_settle) == 0:
    #                 return set()
    #             else:
    #                 # we assume 1:1 mapping of bank to account so we can just extract any txn and use that account
    #                 txn = txns_to_settle.copy().pop()
    #                 bank_account = txn.sender_account

    #             # calculate amount of obligations that arrived in this period
    #             arrived_obligations = sum([txn.amount for txn in txns_to_settle if txn.arrival_time == current_time])
    #             # calculate the amount of claims that arrived in this current period
    #             observed_claims = sum([txn.amount for txn in all_outstanding_transactions if txn.arrival_time == current_time and txn.recipient_account.owner == self.name])

    #             if current_time == "08:00":
    #                 partial_obs = {
    #                     "inbound_payments": 0,
    #                     "arrived_obligations": arrived_obligations,
    #                     "observed_claims": observed_claims,
    #                     "observed_expected": 0.75  # not used when ζ = 0
    #                 }

    #                 self.mdp_state = mdp.update_current_state(self.mdp_state, self.mdp_previous_action, partial_obs)
    #             else:
    #                 # calculate actual inbound payments from previous period
    #                 previous_time = add_minutes_to_time(current_time, -15)
    #                 df_processed_txns = pd.read_csv(f'{sim_name}-processed_transactions.csv')
    #                 filtered_df = df_processed_txns[(df_processed_txns['to_account'] == bank_account) & 
    #                         (df_processed_txns['settlement_time'] == previous_time)]
    #                 inbound_payments = filtered_df['amount'].sum()

    #                 partial_obs = {
    #                     "inbound_payments": inbound_payments,
    #                     "arrived_obligations": arrived_obligations,
    #                     "observed_claims": observed_claims,
    #                     "observed_expected": 0.75  # not used when ζ = 0
    #                 }

    #                 self.mdp_state = mdp.update_current_state(self.mdp_state, self.mdp_previous_action, partial_obs)

    #             _, best_act = mdp.depth_limited_value(self.mdp_state, depth=self.n_periods)
    #             self.n_periods -= 1
    #             self.mdp_previous_action = best_act

    #             if best_act == 1:
    #                 return txns_to_settle
    #             else:
    #                 return set()

    #     # simulation
    #     banks = {'name': ['b1', 'b2', 'b3'], 'strategy_type': ['MechStrategic', 'MechStrategic', 'MechStrategic']}
    #     accounts = {'id': ['acc1', 'acc2', 'acc3'], 'owner': ['b1', 'b2', 'b3'], 'balance': [0, 0, 0]}
    #     transactions = {
    #         'sender_account': ['acc1', 'acc1', 'acc2', 'acc2', 'acc3', 'acc3', 'acc1', 'acc1', 'acc2', 'acc2', 'acc3', 'acc3'], 
    #         'recipient_account': ['acc2', 'acc3', 'acc1', 'acc3', 'acc1', 'acc2', 'acc2', 'acc3', 'acc1', 'acc3', 'acc1', 'acc2'], 
    #         'amount': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
    #         'time': ['08:00', '08:00', '08:00', '08:00', '08:00', '08:00', '08:15', '08:15', '08:15', '08:15', '08:15', '08:15']
    #     }
    #     sim = ABMSim('test_delay', banks, accounts, transactions, strategy_mapping={'MechStrategic': MechStrategicBank}, open_time='08:00', close_time='08:30', eod_force_settlement=True)
    #     sim.run()

    #     # calculate number of delays
    #     df_processed_txns = pd.read_csv('./test_delay-processed_transactions.csv')
    #     num_txns_delayed = len(df_processed_txns[df_processed_txns['time'] != df_processed_txns['submission_time']])

    #     self.assertEqual(num_txns_delayed, 12, 'All 12 transactions expected to be delayed')

    def test_sim_borrow_costs(self):
        """
        Tests if expected borrowings are correctly being captured in simulation, and costs correctly calculated.
        """
        mdp = MechMDPSearch(3, 3, False, 1.0, self.delta, self.delta_prime, self.gamma, self.phi, self.chi, self.zeta, self.seed)
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
                observed_claims = sum([txn.amount for txn in all_outstanding_transactions if txn.arrival_time == current_time and txn.recipient_account.owner.name == self.name])

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

                best_value, best_act = mdp.depth_limited_value(self.mdp_state, depth=self.n_periods)
                self.n_periods -= 1
                self.mdp_previous_action = best_act

                if best_act == 1:
                    return txns_to_settle
                else:
                    return set()
                
        class CollateralizedCreditFacility(AbstractCreditFacility):
            def __init__(self, collateralized_transactions=None, gamma=0.6, phi=0.5, chi=0.75):
                AbstractCreditFacility.__init__(self)
                self.collateralized_transactions = collateralized_transactions if collateralized_transactions is not None else {}
                self.gamma = gamma  # traditional borrowing cost
                self.phi = phi      # pledged-collateral borrowing cost
                self.chi = chi      # unsecured borrowing cost
                # Track credit issued: account.id -> list of tuples (credit_type, amount)
                self.used_credit = {}
                self.history = {}

            def lend_credit(self, account, amount: float) -> None:
                """
                Issue credit to an account using one of three methods based on the following rules:
                
                1. If χ is the lowest among γ, φ, and χ, always issue unsecured credit.
                
                2. If the account has sufficient posted collateral (i.e. account.posted_collateral >= amount):
                - If φ < χ and there exists a valid incoming transaction (valid_txn) that can be used as collateral,
                    issue credit via collateralized transactions (using that valid_txn).
                - Else, issue credit via posted collateral.
                
                3. If the account has insufficient posted collateral (i.e. account.posted_collateral < amount):
                - If χ < φ, issue unsecured credit.
                - Else, if there is a valid incoming transaction, issue credit via collateralized transactions.
                - Otherwise, do not provide credit.
                
                The type of credit issued is recorded in self.used_credit for later cost calculations.
                """
                # THIS IS WRONG. NEED TO AMEND TO ENSURE THAT MORE THAN 1 TXNS CAN BE USED TO GET CREDIT. ALSO ENSURE THAT THERE CAN BE MULTIPLE SOURCES OF CREDIT IF REQUIRED
                # Rule 1: If unsecured borrowing cost is lowest, issue unsecured credit.
                if self.chi < self.gamma and self.chi < self.phi:
                    self.used_credit.setdefault(account.id, []).append(('unsecured', amount))
                    account.balance += amount
                    return

                # Helper: find a valid incoming transaction for collateral.
                def find_valid_txns():
                    lowest_valid_amt = float('inf')
                    lowest_valid_amt_txn = None
                    valid_txns = {txn for txn in account.txn_in if txn.status_code == 0 and 
                                txn not in self.collateralized_transactions.get(account.id, set())}
                    for txn in valid_txns:
                        if txn.amount >= amount and (txn.amount < lowest_valid_amt or lowest_valid_amt is None):
                            lowest_valid_amt = txn.amount
                            lowest_valid_amt_txn = txn
                    return lowest_valid_amt_txn

                # Check if account has sufficient posted collateral.
                if account.posted_collateral >= amount:
                    # Rule 2: Sufficient posted collateral.
                    if self.phi < self.chi:
                        # Try to use a valid incoming transaction as collateral.
                        valid_txn = find_valid_txns()
                        if valid_txn is not None:
                            self.collateralized_transactions.setdefault(account.id, set()).add(valid_txn)
                            self.used_credit.setdefault(account.id, []).append(('collateralized_txn', amount))
                            account.balance += amount
                            return
                    # Fallback: issue credit using posted collateral.
                    self.used_credit.setdefault(account.id, []).append(('collateralized_posted', amount))
                    account.balance += amount
                    account.posted_collateral -= amount
                    return
                else:
                    # Rule 3: Insufficient posted collateral.
                    if self.chi < self.phi:
                        # Unsecured credit is cheaper.
                        self.used_credit.setdefault(account.id, []).append(('unsecured', amount))
                        account.balance += amount
                        return
                    else:
                        # Try to use incoming transaction as collateral.
                        valid_txn = find_valid_txns()
                        if valid_txn is not None:
                            self.collateralized_transactions.setdefault(account.id, set()).add(valid_txn)
                            self.used_credit.setdefault(account.id, []).append(('collateralized_txn', amount))
                            account.balance += amount
                            return
                        else:
                            # No option available: do not issue credit.
                            print('unable to obtain credit')
                            return

            def collect_all_repayment(self, day: int, accounts: List[Account]) -> None:
                """
                Collect repayments from all accounts.
                """
                for account in accounts:
                    self.history.setdefault(account.id, []).append(
                        (day, self.get_total_credit(account), self.get_total_fee(account))
                    )
                    self.collect_repayment(account)

            def calculate_fee(self, credit_amount, credit_type) -> float:
                """
                Calculate the fee for a given credit amount based on its type.
                
                - For 'collateralized_posted', fee = gamma * credit_amount.
                - For 'collateralized_txn', fee = phi * credit_amount.
                - For 'unsecured', fee = chi * credit_amount.
                """
                if credit_type == 'collateralized_posted':
                    return self.gamma * credit_amount
                elif credit_type == 'collateralized_txn':
                    return self.phi * credit_amount
                elif credit_type == 'unsecured':
                    return self.chi * credit_amount
                else:
                    return 0.0


            def collect_repayment(self, account) -> None:
                # Not implemented for now.
                pass

            def get_total_credit(self, account: Account) -> float:
                """
                Obtain the total amount of credit issued to an account.
                """
                return sum([amt for (ctype, amt) in self.used_credit.get(account.id, [])])

            def get_total_fee(self, account: Account) -> float:
                """
                Obtain the total fee amount for an account based on the types of credit issued.
                """
                total_fee = 0.0
                for credit_type, amount in self.used_credit.get(account.id, []):
                    total_fee += self.calculate_fee(credit_amount=amount, credit_type=credit_type)
                return total_fee

            
            def get_total_credit_and_fee(self, account: Account) -> float:
                """
                Obtain the total credit and fee for an account.
                """
                return self.get_total_credit(account) + self.get_total_fee(account)

        # simulation
        banks = {'name': ['b1', 'b2', 'b3'], 'strategy_type': ['MechStrategic', 'MechStrategic', 'MechStrategic']}
        accounts = {'id': ['acc1', 'acc2', 'acc3'], 'owner': ['b1', 'b2', 'b3'], 'balance': [0, 0, 0]}
        transactions = {
            'sender_account': ['acc1', 'acc1', 'acc2', 'acc2', 'acc3', 'acc3', 'acc1', 'acc1', 'acc2', 'acc2', 'acc3', 'acc3'], 
            'recipient_account': ['acc2', 'acc3', 'acc1', 'acc3', 'acc1', 'acc2', 'acc2', 'acc3', 'acc1', 'acc3', 'acc1', 'acc2'], 
            'amount': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 
            'time': ['08:00', '08:00', '08:00', '08:00', '08:00', '08:00', '08:15', '08:15', '08:15', '08:15', '08:15', '08:15']
        }
        collateralized_credit_facility = CollateralizedCreditFacility(gamma=self.gamma, phi=self.phi, chi=self.chi)
        sim = ABMSim('test_borrowing', banks, accounts, transactions, strategy_mapping={'MechStrategic': MechStrategicBank}, open_time='08:00', close_time='08:30', credit_facility=collateralized_credit_facility, eod_force_settlement=True)
        print('SIM TEST START')
        sim.run()
        print('SIM TEST END')

        # calculate number of delays
        df_processed_txns = pd.read_csv('./test_borrowing-processed_transactions.csv')
        num_txns_delayed = len(df_processed_txns[df_processed_txns['time'] != df_processed_txns['submission_time']])
        self.assertEqual(num_txns_delayed, 0, 'No transaction expected to be delayed')

        # calculate borrowing costs
        expected_borrowing_costs = 12 * self.phi
        df_credit_facility = pd.read_csv('./test_borrowing-credit_facility.csv')
        actual_borrowing_costs = df_credit_facility['total_fee'].sum()
        self.assertEqual(actual_borrowing_costs, expected_borrowing_costs, 'Borrowing costs are incorrect')
    
    
    # def tearDown(self):
    #     """Runs after each test method."""
    #     # Remove each .csv file
    #     csv_files = glob.glob("*.csv")
    #     for file in csv_files:
    #         os.remove(file)

if __name__ == '__main__':
    unittest.main()
