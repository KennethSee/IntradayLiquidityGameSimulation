import pandas as pd
from datetime import datetime
from collections import defaultdict
from PSSimPy.utils.time_utils import add_minutes_to_time

class TransactionPath:

    def __init__(self):
        self.txns_list = []
        self.txns_by_time = {}

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
        self.txns_by_time = self._group_txns_by_time(txns_list)

    def retrieve_txns_by_time(self, time: str):
        return self.txns_by_time[time]

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
    
    @staticmethod
    def all_times(start_time: str, end_time: str) -> list:
        """
        Generate 15-minute intervals from start_time up to but not including end_time.
        E.g. all_times('08:00','09:00') → ['08:00','08:15','08:30','08:45'].
        """
        times = []
        current = start_time
        while True:
            # stop before appending if we've reached or passed end_time
            if datetime.strptime(current, '%H:%M') >= datetime.strptime(end_time, '%H:%M'):
                break
            times.append(current)
            current = add_minutes_to_time(current, 15)
        return times