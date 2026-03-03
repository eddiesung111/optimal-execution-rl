# src/environment.py
import numpy as np
import pandas as pd
import random

class ExecutionEnvironment:
    def __init__(self, dfs_dict, total_shares_dict, T=8, tau_seconds=60, 
                 num_buckets_I=16, num_buckets_S=5, num_buckets_V=5, num_buckets_M=10, is_ddqn=True):
        
        self.dfs_dict = dfs_dict
        self.total_shares_dict = total_shares_dict
        self.tickers = list(dfs_dict.keys())
        self.is_ddqn = is_ddqn
        
        self.T = T
        self.tau_seconds = tau_seconds
        
        self.num_buckets_I = num_buckets_I
        self.num_buckets_S = num_buckets_S
        self.num_buckets_V = num_buckets_V
        self.num_buckets_M = num_buckets_M
        
        self.action_space = np.linspace(0.5, 1.5, 11)
        
        self.current_ticker = None
        self.df_lob = None
        
        self.current_t = 0
        self.current_idx = 0 
        self.total_shares = 0
        self.inventory_left = 0
        self.total_ideal_cost = 0.0
        self.total_actual_cost = 0.0
        self.prev_mid_price = 0.0

    def reset(self, ticker=None, start_idx=None):
        if ticker is None:
            self.current_ticker = random.choice(self.tickers)
        else:
            self.current_ticker = ticker
        
        # Load the LOB DataFrame and AC trajectory for the selected ticker
        self.df_lob = self.dfs_dict[self.current_ticker]
        self.total_shares = self.total_shares_dict[self.current_ticker]
        self.inventory_left = self.total_shares

        self.current_t = 0
        self.total_actual_cost = 0.0 

        # Randomly select a valid starting index for the episode
        rows_needed_for_episode = self.T * self.tau_seconds
        max_start_idx = len(self.df_lob) - rows_needed_for_episode - 1
        if start_idx is None:
            self.current_idx = np.random.randint(0, max_start_idx)
        else:
            self.current_idx = start_idx
        
        # Calculate the arrival price and total ideal cost based on the starting index
        row = self.df_lob.iloc[self.current_idx]
        self.arrival_price = (row['ap1'] + row['bp1']) / 2.0

        self.prev_mid_price = self.arrival_price 
        self.total_ideal_cost = self.total_shares * self.arrival_price
        
        return self._get_state()


    def step(self, action_idx):
        beta = self.action_space[action_idx]
        baseline_trade = self.total_shares / self.T

        shares_to_buy = baseline_trade * beta
        shares_to_buy = min(shares_to_buy, self.inventory_left)

        if self.current_t == self.T - 1:
            shares_to_buy = self.inventory_left
        
        row = self.df_lob.iloc[self.current_idx]
        current_mid_price = (row['ap1'] + row['bp1']) / 2.0

        actual_cost = self._walk_the_book(row, shares_to_buy)
        self.total_actual_cost += actual_cost
        
        if shares_to_buy > 0:
            step_mid_cost = shares_to_buy * current_mid_price
            slippage_penalty = actual_cost - step_mid_cost
        else:
            slippage_penalty = 0

        new_inventory_left = self.inventory_left - shares_to_buy
        price_drift = current_mid_price - self.prev_mid_price
        inventory_penalty = new_inventory_left * price_drift

        raw_step_cost = slippage_penalty + inventory_penalty
        execution_reward = -(raw_step_cost / self.total_ideal_cost) * 10000.0

        self.inventory_left = new_inventory_left
        self.current_t += 1
        self.current_idx += self.tau_seconds
        self.prev_mid_price = current_mid_price

        done = (self.current_t >= self.T) or (self.inventory_left <= 1e-5)
        
        next_state = self._get_state()
        
        info = {
            "step_actual_cost": actual_cost,
            "slippage_penalty": slippage_penalty,
            "inventory_penalty": inventory_penalty,
            "shares_bought": shares_to_buy,
            "ticker": self.current_ticker
        }
        
        return next_state, execution_reward, done, info


    def _walk_the_book(self, row, shares_to_buy):
        if shares_to_buy == 0:
            return 0.0
            
        shares_remaining = shares_to_buy
        total_cost = 0.0

        # Walk through levels 1 to 5
        for level in range(1, 6): 
            ask_price = row[f'ap{level}']
            ask_vol = row[f'av{level}']
            
            if ask_vol >= shares_remaining:
                total_cost += shares_remaining * ask_price
                shares_remaining = 0
                break
            else:
                total_cost += ask_vol * ask_price
                shares_remaining -= ask_vol

        if shares_remaining > 0:
            penalty_multiplier = 1.001
            penalty_price = row['ap5'] * penalty_multiplier
            total_cost += shares_remaining * penalty_price
            
        return total_cost

    
    def _get_state(self):
        if self.is_ddqn:
            return self._get_state_ddqn()
        else:
            return self._get_state_tabular()
        
    def _get_state_tabular(self):
        row = self.df_lob.iloc[self.current_idx]
        t = self.current_t

        # Inventory Bucket
        if self.total_shares > 0:
            i_ratio = self.inventory_left / self.total_shares
            i_bucket = min(int(i_ratio * self.num_buckets_I), self.num_buckets_I - 1)
        else:
            i_bucket = 0
        
        # Spread Bucket (spn is 0.0 to 1.0)
        s_val = float(row['spn']) * self.num_buckets_S
        s_bucket = max(0, min(int(s_val), self.num_buckets_S - 1))
        
        # Ask Volume Bucket (vpn is 0.0 to 1.0)
        v_val = float(row['vpn']) * self.num_buckets_V
        v_bucket = max(0, min(int(v_val), self.num_buckets_V - 1))

        # Imbalance Bucket (Imbalance is -1.0 to 1.0)
        imbalance_raw = float(row['imbalance'])
        imbalance_normalized = (imbalance_raw + 1.0) / 2.0 
        
        m_val = imbalance_normalized * self.num_buckets_M
        m_bucket = max(0, min(int(m_val), self.num_buckets_M - 1))

        return (t, i_bucket, s_bucket, v_bucket, m_bucket)
    

    def _get_state_ddqn(self):
        row = self.df_lob.iloc[self.current_idx]
        
        # Normalize all the features into [-1,1] for better training
        # Time
        time_ratio = self.current_t / (self.T - 1)
        time_pct = (time_ratio * 2.0) - 1.0 
        
        # Inventory
        if self.total_shares > 0:
            inv_ratio = self.inventory_left / self.total_shares
            inv_pct = (inv_ratio * 2.0) - 1.0
        else:
            inv_pct = -1.0
        
        # Spread (spn)
        spread_raw = row['spn']
        spread_scaled = (spread_raw * 2.0) - 1.0
        
        # Ask Volume (vpn)
        vol_raw = row['vpn']
        vol_scaled = (vol_raw * 2.0) - 1.0

        # Order Book Imbalance
        imbalance = row['imbalance']

        # Autocorrelation Z-score
        raw_ac = row['auto_corr']
        ac_mean = row['auto_corr_mean']
        ac_std = row['auto_corr_std']

        if ac_std < 1e-6:
            norm_ac = 0.0
        else:
            norm_ac = (raw_ac - ac_mean) / ac_std
        
        # Autocorrelation feature
        norm_ac = np.clip(norm_ac, -3.0, 3.0) / 3.0

        # Return all 6 dimensions as a float32 numpy array
        return np.array([time_pct, inv_pct, spread_scaled, vol_scaled, imbalance, norm_ac], dtype=np.float32)
    

    def get_state_dim(self):
        """Returns the size of the DDQN state vector."""
        return 6

    def get_action_dim(self):
        """Returns the number of discrete actions in the action space."""
        return len(self.action_space)
