import numpy as np
from scipy.stats import skewnorm

# -----------------------------
# 1. Ratchet calculation
# -----------------------------
def get_ratchet_ratio(levels, ratios, current_frac, method='linear'):
    for i in range(len(levels)-1):
        if current_frac >= levels[i] and current_frac < levels[i+1]:
            if method == 'linear':
                ratio = ratios[i] + (ratios[i+1]-ratios[i])*(current_frac - levels[i])/(levels[i+1]-levels[i])
            elif method == 'step':
                ratio = ratios[i]
            else:
                raise ValueError("method must be 'linear' or 'step'")
            return ratio
    return ratios[-1]

def dp_gas_storage_stochastic(forward_prices, I0, I_max, U_max, W_max,
                              inject_levels, inject_ratios,
                              withdraw_levels, withdraw_ratios,
                              switching_cost=0,
                              ratchet_method='linear',
                              n_inventory_steps=101,
                              r_annual=0.05,
                              n_paths=1000,
                              sigma=0.1, dt=1,
                              I_terminal=None, terminal_tolerance=0.1, terminal_penalty=1e3):
    """
    Stochastic DP gas storage solver:
    - Accounts for optionality using Monte Carlo price paths
    - Maximizes expected NPV
    - Respects ratchets, max rates, switching costs
    - Can enforce a realistic terminal inventory (I_terminal)
    """
    T = len(forward_prices)
    inventory_grid = np.linspace(0, I_max, n_inventory_steps)
    discount_per_day = 1 / ((1 + r_annual) ** dt)
    
    # Monte Carlo paths around forward curve
    F = np.array(forward_prices)
    paths = np.zeros((n_paths, T))
    paths[:, 0] = F[0]
    for t in range(1, T):
        alpha = 10
        z = skewnorm.rvs(a=alpha, size=n_paths)
        paths[:, t] = F[t] * np.exp(sigma * np.sqrt(dt) * z)
    
    # DP table
    V = np.zeros((T+1, n_inventory_steps))
    policy = np.zeros((T, n_inventory_steps))
    
    # Terminal condition
    if I_terminal is not None:
        # Penalize deviation from target inventory
        V[T, :] = -terminal_penalty * np.maximum(np.abs(inventory_grid - I_terminal) - terminal_tolerance*I_max, 0)
    else:
        # Default: must be empty
        V[T, :] = -1e9 * (inventory_grid != 0)
    
    # Backward induction
    for t in reversed(range(T)):
        for i, I in enumerate(inventory_grid):
            best_value = -np.inf
            best_delta = 0
            current_frac = I / I_max
            U_eff = U_max * get_ratchet_ratio(inject_levels, inject_ratios, current_frac, ratchet_method)
            W_eff = W_max * get_ratchet_ratio(withdraw_levels, withdraw_ratios, current_frac, ratchet_method)
            
            delta_range = np.linspace(-min(W_eff, I), min(U_eff, I_max - I), 11)
            
            for delta in delta_range:
                I_next = I + delta
                if I_next < 0 or I_next > I_max:
                    continue
                u = max(delta, 0)
                w = max(-delta, 0)
                
                # Expected NPV over all paths
                expected_value = 0
                for path in paths:
                    P = path[t]
                    immediate_profit = discount_per_day**t * (P*w - P*u - switching_cost)
                    future_value = np.interp(I_next, inventory_grid, V[t+1, :])
                    expected_value += immediate_profit + future_value
                expected_value /= n_paths
                
                if expected_value > best_value:
                    best_value = expected_value
                    best_delta = delta
            
            V[t, i] = best_value
            policy[t, i] = best_delta
    
    # Forward simulation using expected forward prices
    inventory = np.zeros(T+1)
    inventory[0] = I0
    inject = np.zeros(T)
    withdraw = np.zeros(T)
    profit = np.zeros(T)
    
    for t in range(T):
        delta = np.interp(inventory[t], inventory_grid, policy[t, :])
        u = max(delta, 0)
        w = max(-delta, 0)
        inventory[t+1] = inventory[t] + u - w
        inject[t] = u
        withdraw[t] = w
        profit[t] = forward_prices[t]*w - forward_prices[t]*u
    
    return {
        'inventory': inventory,
        'inject': inject,
        'withdraw': withdraw,
        'profit': profit,
        'cumulative_npv': np.cumsum(profit * (discount_per_day ** np.arange(T)))
    }