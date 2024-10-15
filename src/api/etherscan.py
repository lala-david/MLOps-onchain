import asyncio
import httpx
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
import logging

load_dotenv()

api_key = os.getenv('ETHERSCAN_API_KEY')
data = pd.read_csv('eth_mlops.csv')
addresses = data['address']
flags = data['flag']

output_file = 'eth_data_output.csv'
columns = ['address', 'AvgRecTnxInterval', 'ActivityDays', 'TotalTransactions', 'AvgValSent',
           'NumUniqRecAddress', 'NumUniqSetAddress', 'TotalEtherBalance', 'TxFreq',
           'MaxValueSent', 'MaxValueReceived', 'maxTimeBetweenRecTnx', 'avgTransactionValue',
           'receivedTransactions', 'sentTransactions', 'AvgSentTnxInterval', 'AvgRecTnxValue', 'flag']

with open(output_file, 'w', encoding='utf-8') as f:
    pd.DataFrame(columns=columns).to_csv(f, index=False)

logging.basicConfig(level=logging.INFO)

async def fetch_transactions(client, address):
    url = f"https://api.etherscan.io/api?module=account&action=txlist&address={address}&startblock=0&endblock=99999999&sort=asc&apikey={api_key}"
    try:
        response = await client.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == '1':
                transactions = []
                for tx in data['result']:
                    from_address = tx['from']
                    to_address = tx['to']
                    amount = int(tx['value']) / 10**18
                    timestamp = int(tx['timeStamp'])
                    gas = int(tx['gasUsed'])
                    transactions.append([from_address, to_address, amount, timestamp, gas])
                return transactions
            else:
                logging.warning(f"No transactions found for address {address}")
                return []
        else:
            logging.warning(f"Failed to fetch transactions for address {address}: {response.status_code}")
            return None
    except Exception as e:
        logging.warning(f"Failed to fetch transactions for address {address}: {e}")
        return None

async def process_address(client, address, flag):
    transactions = await fetch_transactions(client, address)
    if not transactions:
        return None

    try:
        tx_array = np.array(transactions)

        total_transactions = int(tx_array.shape[0])
        received_transactions = int(np.sum(tx_array[:, 1] == address))
        sent_transactions = int(np.sum(tx_array[:, 0] == address))

        if received_transactions > 0:
            received_times = tx_array[tx_array[:, 1] == address][:, 3].astype(np.int64)
            received_values = tx_array[tx_array[:, 1] == address][:, 2].astype(np.float64)
            avg_rec_interval = np.diff(received_times).mean().item() if len(received_times) > 1 else None
            avg_rec_value = received_values.mean().item()
            max_time_between_rec = np.diff(received_times).max().item() if len(received_times) > 1 else None
            max_value_received = received_values.max().item()
        else:
            avg_rec_interval = None
            avg_rec_value = None
            max_time_between_rec = None
            max_value_received = None

        if sent_transactions > 0:
            sent_times = tx_array[tx_array[:, 0] == address][:, 3].astype(np.int64)
            sent_values = tx_array[tx_array[:, 0] == address][:, 2].astype(np.float64)
            avg_sent_interval = np.diff(sent_times).mean().item() if len(sent_times) > 1 else None
            avg_sent_value = sent_values.mean().item()
            max_value_sent = sent_values.max().item()
        else:
            avg_sent_interval = None
            avg_sent_value = None
            max_value_sent = None

        total_ether_balance = tx_array[:, 2].astype(np.float64).sum().item()
        num_uniq_rec_address = len(np.unique(tx_array[tx_array[:, 1] == address][:, 0]))
        num_uniq_set_address = len(np.unique(tx_array[tx_array[:, 0] == address][:, 1]))

        if tx_array[:, 3].astype(np.int64).max() != tx_array[:, 3].astype(np.int64).min():
            tx_freq = (total_transactions / (tx_array[:, 3].astype(np.int64).max() - tx_array[:, 3].astype(np.int64).min())).item()
        else:
            tx_freq = None

        activity_days = ((tx_array[:, 3].astype(np.int64).max() - tx_array[:, 3].astype(np.int64).min()) / (60 * 60 * 24)).item()
        avg_tx_value = tx_array[:, 2].astype(np.float64).mean().item()

        return (address, avg_rec_interval, activity_days, total_transactions, avg_sent_value,
                num_uniq_rec_address, num_uniq_set_address, total_ether_balance, tx_freq,
                max_value_sent, max_value_received, max_time_between_rec, avg_tx_value,
                received_transactions, sent_transactions, avg_sent_interval, avg_rec_value, flag)
    except Exception as e:
        logging.warning(f"Error processing address {address}: {e}")
        return None

async def main():
    async with httpx.AsyncClient() as client:
        batch_size = 50
        for i in tqdm(range(0, len(addresses), batch_size), desc="Processing addresses", colour="blue", dynamic_ncols=True):
            addresses_batch = addresses[i:i+batch_size]
            flags_batch = flags[i+i+batch_size]
            tasks = [process_address(client, address, flag) for address, flag in zip(addresses_batch, flags_batch)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            valid_results = [res for res in results if res is not None]
            if valid_results:
                with open(output_file, 'a', encoding='utf-8') as f:
                    pd.DataFrame(valid_results, columns=columns).to_csv(f, header=False, index=False)

            await asyncio.sleep(1)  

if __name__ == "__main__":
    asyncio.run(main())