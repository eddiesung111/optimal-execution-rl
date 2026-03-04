# src/data_loader.py
import pandas as pd
import numpy as np
from pathlib import Path

def process_lobster_dataset(ticker, orderbook_file, message_file, output_file):
    
    print(f"[{ticker}] Loading files...")

    msg_cols = ['time', 'type', 'order_id', 'size', 'price', 'direction']
    ob_cols = [
        'ap1', 'av1', 'bp1', 'bv1', 
        'ap2', 'av2', 'bp2', 'bv2', 
        'ap3', 'av3', 'bp3', 'bv3', 
        'ap4', 'av4', 'bp4', 'bv4', 
        'ap5', 'av5', 'bp5', 'bv5'
    ]
    

    df_msg = pd.read_csv(message_file, header=None, names=msg_cols)
    df_ob = pd.read_csv(orderbook_file, header=None, names=ob_cols)

    df_ob['time_seconds'] = df_msg['time']
    df_ob['type'] = df_msg['type']
    
    # Dropping columns with type 7, since the trading halts.
    df_ob = df_ob[df_ob['type'] != 7].copy()

    # Convert prices from integer to float (divide by 10,000)
    price_columns = ['ap1', 'ap2', 'ap3', 'ap4', 'ap5', 'bp1', 'bp2', 'bp3', 'bp4', 'bp5']
    for col in price_columns:
        df_ob[col] = df_ob[col] / 10000.0
    
    # Create a "time bucket" column by flooring the time_seconds to the nearest integer
    df_ob['time_bucket'] = np.floor(df_ob['time_seconds']).astype(int)
    df_snapshot = df_ob.groupby('time_bucket').last()
    
    # Market Open is 34200 (9:30 AM), Market Close is 57600 (4:00 PM)
    master_clock = np.arange(34200, 57601)
    df_snapshot = df_snapshot.reindex(master_clock, method='ffill')
    df_snapshot = df_snapshot.bfill().reset_index()
    df_snapshot = df_snapshot.rename(columns={'index': 'time_bucket'})
    df_snapshot = df_snapshot.drop(columns=['type', 'time_seconds'])


    df_snapshot['Spread'] = df_snapshot['ap1'] - df_snapshot['bp1']
    df_snapshot['spn'] = df_snapshot['Spread'].rolling(window = 960, min_periods=240).rank(pct=True)
    df_snapshot['vpn'] = df_snapshot['av1'].rolling(window = 960, min_periods=240).rank(pct=True)

    df_snapshot['p_mid'] = (df_snapshot['ap1'] + df_snapshot['bp1']) / 2.0
    df_snapshot['v_ask_total'] = df_snapshot[['av1', 'av2', 'av3', 'av4', 'av5']].sum(axis=1)
    df_snapshot['v_bid_total'] = df_snapshot[['bv1', 'bv2', 'bv3', 'bv4', 'bv5']].sum(axis=1)
    df_snapshot['v_total'] = (df_snapshot['v_bid_total'] + df_snapshot['v_ask_total'])

    df_snapshot['imbalance'] = (df_snapshot['v_bid_total'] - df_snapshot['v_ask_total']) / df_snapshot['v_total']
    df_snapshot['log_return'] = np.log(df_snapshot['p_mid'] / df_snapshot['p_mid'].shift(1))
    df_snapshot['auto_corr'] = df_snapshot['log_return'].rolling(window=300, min_periods=60).corr(df_snapshot['log_return'].shift(60)).fillna(0)

    df_snapshot['auto_corr_mean'] = df_snapshot['auto_corr'].rolling(window = 960, min_periods=150).mean().fillna(0)
    df_snapshot['auto_corr_std'] = df_snapshot['auto_corr'].rolling(window = 960, min_periods=150).std().fillna(1e-6)
    df_snapshot.dropna(subset=['spn', 'vpn'], inplace=True)
    df_snapshot.to_csv(output_file, index=False)

    display_path = f"root/data/{ticker}/{ticker}_clean.csv"
    print(f"[{ticker}] Success! Saved exactly {len(df_snapshot)} 1-second snapshots to {display_path}\n")


def main():
    tickers = ['MSFT', 'AAPL', 'AMZN', 'GOOG', 'INTC']

    data_dir = Path(__file__).resolve().parent.parent / 'data'

    for ticker in tickers:
        display_path = f"root/data/{ticker}/{ticker}_clean.csv"
        message_files = data_dir / ticker / 'message.csv'
        orderbook_files = data_dir / ticker / 'orderbook.csv'

        if message_files.exists() and orderbook_files.exists():
            out_file = data_dir / ticker / f"{ticker}_clean.csv"
            process_lobster_dataset(ticker, str(orderbook_files), str(message_files), str(out_file))
        else:
            print(f"Files for {ticker} not found in {display_path}. Skipping...")

if __name__ == "__main__":
    main()