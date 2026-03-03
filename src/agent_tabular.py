import numpy as np

class QAgent:
    def __init__(self, T=8, num_I=16, num_S=5, num_V=5, num_M=10, actions=None, alpha_0=1.0, gamma=1.0):
        self.T = T
        self.num_I = num_I
        self.num_S = num_S
        self.num_V = num_V
        self.num_M = num_M

        self.actions = actions
            
        self.alpha_0 = alpha_0
        self.gamma = gamma
        
        # Initialize the Q-table and Visit-table
        shape = (self.T, self.num_I, self.num_S, self.num_V, self.num_M, len(self.actions))
        self.q_table = np.zeros(shape)
        self.n_table = np.zeros(shape)

    def get_action(self, state):
        t, i, s, v, m = state
        q_values = self.q_table[t, i, s, v, m, :]
        max_q = np.max(q_values)

        # Handles the ssame q-values situation
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)

    def update(self, state, action_idx, reward, next_state):
        t, i, s, v, m = state
        
        # Update visit counts
        self.n_table[t, i, s, v, m, action_idx] += 1
        visits = self.n_table[t, i, s, v, m, action_idx]
        
        # Decay the learning rate based on state-action visits
        alpha = self.alpha_0 / visits

        current_q = self.q_table[t, i, s, v, m, action_idx]
        
        if next_state is not None:
            next_t, next_i, next_s, next_v, next_m = next_state
            
            # Terminal state check
            if next_t >= self.T:
                max_future_q = 0.0
            else:
                max_future_q = np.max(self.q_table[next_t, next_i, next_s, next_v, next_m, :])
            
            # Standard Q-learning Temporal Difference (TD) target
            u_n = reward + (self.gamma * max_future_q) - current_q
            self.q_table[t, i, s, v, m, action_idx] = current_q + (alpha * u_n)