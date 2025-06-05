import pandas as pd
from datetime import datetime
from collections import defaultdict

class TransactionPath:

    def __init__(self):
        self.txns_list = []

    def extract_txns_from_df(self, df):
        """
        Extracts tuples of (time, from_account, to_account, amount) from the DataFrame,
        sorted by time in ascending order.
        """
        # Create a list of tuples from the DataFrame rows
        txns_list = list(df[['time', 'from_account', 'to_account', 'amount']].itertuples(index=False, name=None))
        
        # Sort the list using the time column (HH:MM format)
        txns_list.sort(key=lambda x: datetime.strptime(x[0], "%H:%M").time())
        
        self.txns_list = txns_list

    @staticmethod
    def _group_txns_by_time(txns_list):
        """
        Given a list of transaction tuples (time, from_account, to_account, amount),
        group them into a dictionary with 'time' as the key and a list of transactions as the value.

        Parameters:
        - txns_list: List of tuples (time, from_account, to_account, amount)

        Returns:
        - Dictionary where each key is a time string and the value is a list of transactions.
        """
        time_dict = defaultdict(list)
        
        for txn in txns_list:
            time = txn[0]
            time_dict[time].append(txn)
        
        return dict(time_dict)