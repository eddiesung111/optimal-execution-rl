# main.py
import argparse
import torch
import numpy as np
import pickle
import pandas as pd

from src import DDQNAgent, ExecutionEnvironment, AlmgrenChrissModel, QAgent

def load_market_data(tickers, T=8, tau=60, eta=0.1):
    print("--- Loading Market Data and Calculating AC Baselines ---")
    dfs_dict = {}
    ac_trajectories_dict = {}
    total_shares_dict = {}
    global_risk_target = 0.00001

    for ticker in tickers:
        data_path = f"data/{ticker}/{ticker}_clean.csv"
        df_lob = pd.read_csv(data_path)
        
        whole_book_vol = df_lob['av1'].mean() + df_lob['av2'].mean() + df_lob['av3'].mean() + df_lob['av4'].mean() + df_lob['av5'].mean()
        target_step_volume = whole_book_vol * 0.4
        V = target_step_volume * T
        
        price_diffs = df_lob['ap1'].diff().dropna()
        sigma = np.std(price_diffs)
        dynamic_lam = global_risk_target / (sigma ** 2)
        print(f"[{ticker}] Loaded, V = {V:<5.0f} shares | Sigma = {sigma:.6f}")
        
        ac_model = AlmgrenChrissModel(V=V, T=T, tau=tau, lam=dynamic_lam, sigma=sigma, eta=eta)
        
        dfs_dict[ticker] = df_lob
        ac_trajectories_dict[ticker] = ac_model.generate_trajectory()
        total_shares_dict[ticker] = V # Store V for TWAP
        
    print("------------------------------------------------------\n")
    return dfs_dict, ac_trajectories_dict, total_shares_dict


def train_tabular(dfs_dict, ac_trajectories_dict, total_shares_dict, episodes=500):
    print(f"--- Starting Training Phase for Tabular DP ---")
    env = ExecutionEnvironment(dfs_dict, total_shares_dict, is_ddqn=False)
    agent = QAgent(
        T=env.T, 
        num_I=env.num_buckets_I, 
        num_S=env.num_buckets_S, 
        num_V=env.num_buckets_V, 
        num_M=env.num_buckets_M,
        actions=env.action_space
    )
    
    T = env.T
    I = env.num_buckets_I
    A = len(env.action_space)
    
    reward_history = []
    
    for episode in range(episodes):
        env.reset() 
        start_ticker = env.current_ticker
        start_idx = env.current_idx
        V = env.total_shares
        
        for t in reversed(range(T)):
            for i_bucket in range(I): 
                for a_idx in range(A):
                    env.current_t = t
                    env.current_idx = start_idx + (t * env.tau_seconds)
                    env.inventory_left = (i_bucket / max(1, I - 1)) * V
                    
                    state_x = env._get_state_tabular()
                    if state_x[0] >= T:
                        continue
                        
                    state_y, reward, done, info = env.step(a_idx)
                    agent.update(state_x, a_idx, reward, state_y)
                    
        # Perform forward test for debugging
        env.reset(ticker=start_ticker, start_idx=start_idx)
        episode_reward = 0
        done = False
        while not done:
            state = env._get_state_tabular()
            action_idx = agent.get_action(state)
            next_state, reward, done, info = env.step(action_idx)
            episode_reward += reward
            
        reward_history.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(reward_history[-100:])
            print(f"Episode: {episode + 1}/{episodes} | "
                  f"Recent Avg Reward: {avg_reward:.2f}")

    print("--- Training Complete! Saving models... ---")
    np.save("models/tabular_model.npy", agent.q_table)
    print("Model saved to 'models/tabular_model.npy'.")
    return agent


def train_ddqn(dfs_dict, ac_trajectories_dict, total_shares_dict, episodes=50000):
    print(f"--- Starting Training Phase for DDQN ---")
    env = ExecutionEnvironment(dfs_dict, total_shares_dict, is_ddqn=True)
    agent = DDQNAgent(state_dim=6, action_dim=len(env.action_space)) 

    # Track rewards
    reward_history = []

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action_idx = agent.get_action(state)
            next_state, reward, done, info = env.step(action_idx)
            agent.store_transition(state, action_idx, reward, next_state, done)
            agent.update()

            state = next_state
            episode_reward += reward

        agent.decay_epsilon()
        reward_history.append(episode_reward)

        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(reward_history[-1000:])
            print(f"Episode: {episode + 1}/{episodes} | Epsilon: {agent.epsilon:.3f} | Recent Avg Reward: {avg_reward:.2f}")

    print("--- Training Complete! Saving models... ---")
    torch.save(agent.policy_net.state_dict(), "models/ddqn_model.pth")
    print("Model saved to 'models/ddqn_model.pth'.")


def test_tabular(dfs_dict, ac_trajectories_dict, total_shares_dict, ticker, start_indices):
    episodes = len(start_indices)
    print(f"\n--- Testing Tabular DP on {ticker} ({episodes} episodes) ---")
    
    env = ExecutionEnvironment(dfs_dict, total_shares_dict, is_ddqn=False)
    agent = QAgent(
        T=env.T, 
        num_I=env.num_buckets_I, 
        num_S=env.num_buckets_S, 
        num_V=env.num_buckets_V, 
        num_M=env.num_buckets_M,
        actions=env.action_space
    )
    agent.q_table = np.load("models/tabular_model.npy")
    
    rl_bps_history, twap_bps_history, ac_bps_history = [], [], []
    
    for episode, start_idx in enumerate(start_indices):
        print_trajectory = (episode == 0)
        
        if print_trajectory:
            print(f"--- Trajectory & State Inspection: {ticker} ---")
            print(f"Step | TWAP Shares | RL Shares | Mult  | [Time, Inv, Spread, Vol, Mom]") 
            print("-" * 75)
        
        # RL Agent
        env.reset(ticker=ticker, start_idx=start_idx)
        while env.current_t < env.T:
            state = env._get_state_tabular()
            action_idx = agent.get_action(state)
            next_state, reward, done, info = env.step(action_idx)

            if print_trajectory:
                t = env.current_t - 1
                baseline_shares = env.total_shares / env.T
                rl_shares = info['shares_bought']
                action_mult = env.action_space[action_idx]
                state_str = f"[{state[0]:2d}, {state[1]:2d}, {state[2]:2d}, {state[3]:2d}, {state[4]:2d}]"
                print(f"{t:4d} | {baseline_shares:11.1f} | {rl_shares:9.1f} | {action_mult:4.2f}x | {state_str}")
            
        rl_bps_history.append(((env.total_ideal_cost - env.total_actual_cost) / env.total_ideal_cost) * 10000)
        
        # TWAP Baseline
        env.reset(ticker=ticker, start_idx=start_idx)
        while env.current_t < env.T:
            env.step(5) # Index 5 is the 1.0x neutral multiplier
            
        twap_bps_history.append(((env.total_ideal_cost - env.total_actual_cost) / env.total_ideal_cost) * 10000)

        # AC Baseline
        ac_actual_cost = 0.0
        ac_traj = ac_trajectories_dict[ticker]
        for t in range(env.T):
            idx = start_idx + (t * env.tau_seconds)
            row = dfs_dict[ticker].iloc[idx]
            ac_actual_cost += env._walk_the_book(row, ac_traj[t])
            
        ac_bps_history.append(((env.total_ideal_cost - ac_actual_cost) / env.total_ideal_cost) * 10000)

    return rl_bps_history, twap_bps_history, ac_bps_history


def test_ddqn(dfs_dict, ac_trajectories_dict, total_shares_dict, ticker, start_indices):
    episodes = len(start_indices)
    print(f"\n--- Testing DDQN on {ticker} ({episodes} episodes) ---")
    
    env = ExecutionEnvironment(dfs_dict, total_shares_dict, is_ddqn=True)
    agent = DDQNAgent(state_dim=6, action_dim=len(env.action_space))
    agent.policy_net.load_state_dict(torch.load("models/ddqn_model.pth"))
    agent.policy_net.eval()
    agent.epsilon = 0.0 
    
    rl_bps_history, twap_bps_history, ac_bps_history = [], [], []
    neutral_action_idx = len(env.action_space) // 2 

    for episode, start_idx in enumerate(start_indices):
        print_trajectory = (episode == 0)
        
        if print_trajectory:
            print(f"--- Trajectory & State Inspection: {ticker} ---")
            print(f"Step | TWAP Shares | RL Shares | Mult  | [Time , Inv  , spn  , vpn  , Imbal, AutoC]")
            print("-" * 92)
        
        # RL Agent
        state = env.reset(ticker=ticker, start_idx=start_idx)
        while env.current_t < env.T:
            action_idx = agent.get_action(state)
            next_state, reward, done, info = env.step(action_idx)

            if print_trajectory:
                t = env.current_t - 1
                twap_shares = env.total_shares / env.T
                rl_shares = info['shares_bought']
                action_mult = env.action_space[action_idx]
                state_str = f"[{state[0]:+5.2f}, {state[1]:+5.2f}, {state[2]:+5.2f}, {state[3]:+5.2f}, {state[4]:+5.2f}, {state[5]:+5.2f}]"
                print(f"{t:4d} | {twap_shares:9.1f} | {rl_shares:9.1f} | {action_mult:4.2f}x | {state_str}")
            state = next_state
            
        rl_bps_history.append(((env.total_ideal_cost - env.total_actual_cost) / env.total_ideal_cost) * 10000)
        
        # TWAP Baseline
        env.reset(ticker=ticker, start_idx=start_idx)
        while env.current_t < env.T:
            env.step(neutral_action_idx) 
            
        twap_bps_history.append(((env.total_ideal_cost - env.total_actual_cost) / env.total_ideal_cost) * 10000)

        # AC Baseline
        ac_actual_cost = 0.0
        ac_traj = ac_trajectories_dict[ticker]
        for t in range(env.T):
            idx = start_idx + (t * env.tau_seconds)
            row = dfs_dict[ticker].iloc[idx]
            ac_actual_cost += env._walk_the_book(row, ac_traj[t])
            
        ac_bps_history.append(((env.total_ideal_cost - ac_actual_cost) / env.total_ideal_cost) * 10000)

    return rl_bps_history, twap_bps_history, ac_bps_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Limit Order Book RL Trading System")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'])  
    parser.add_argument('--agent', type=str, required=True, choices=['ddqn', 'tabular'])
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    STOCK = ["AAPL", "AMZN", "GOOG", "INTC", "MSFT"]
    dfs, ac_trajs, total_shares = load_market_data(STOCK)

    print(f"Initializing {args.agent.upper()} agent in {args.mode.upper()} mode...")
    
    if args.mode == 'train':
        if args.agent == 'tabular':
            train_tabular(dfs, ac_trajs, total_shares, episodes=200 if args.debug else 15000)
        elif args.agent == 'ddqn':
            train_ddqn(dfs, ac_trajs, total_shares, episodes=2000 if args.debug else 50000)

    elif args.mode == 'test':
        np.random.seed(16112001) # !!!MY BIRTHDAY!!!
        
        all_histories = {}
        test_eps = 10 if args.debug else 500

        warmup_seconds = 960
        episode_length = 480
        
        for ticker in STOCK:
            df_len = len(dfs[ticker])
            valid_start_min = warmup_seconds
            valid_start_max = df_len - episode_length
            
            start_indices = np.random.randint(valid_start_min, valid_start_max, size=test_eps).tolist()
            
            if args.agent == 'tabular':
                rl_h, twap_h, ac_h = test_tabular(dfs, ac_trajs, total_shares, ticker, start_indices)
            elif args.agent == 'ddqn':
                rl_h, twap_h, ac_h = test_ddqn(dfs, ac_trajs, total_shares, ticker, start_indices)
            
            all_histories[ticker] = {
                'rl': rl_h,
                'twap': twap_h,
                'ac': ac_h
            }
        
        save_path = f"results/test_history_{args.agent}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(all_histories, f)
            
        print(f"\nRaw test data saved to: {save_path}")
        print("Run `python utils.py` to generate the CSV and comparison graphs.")