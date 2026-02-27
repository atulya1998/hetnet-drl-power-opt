"""
test_environment.py — Unit tests for HetNet environment
test_constraints.py — Unit tests for constraint handler
"""

# ======================================================================
# test_environment.py
# ======================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest

DEFAULT_CFG = {
    'n_ue': 10, 'n_sbs': 4, 'n_mbs': 1,
    'mbs_coverage': 500, 'bh_link_range_min': 150,
    'bh_link_range_max': 200, 'n_clusters': 2,
    'bandwidth_an': 20e6, 'bandwidth_bh': 200e6,
    'n_prb': 100, 'carrier_freq': 2e9,
    'T_AN_X_mbs': 8, 'T_AN_X_sbs': 8,
    'T_AN_max_mbs': 130.0, 'T_AN_max_sbs': 1.0,
    'p_AN_o_mbs': 130.0,   'p_AN_o_sbs': 6.8,
    'delta_AN_mbs': 4.7,   'delta_AN_sbs': 4.0,
    'T_BH_X': 8, 'p_BH_o': 3.9, 'delta_BH': 0.0631,
    'Gamma_max': 100,
    'G_rx_vband': 36, 'G_rx_eband': 43,
    'L_rx': 5.0, 'N_f': 30.0, 'L_M': 15.0,
    'T_n': -174.0, 'EIRP_max': 85.0,
}

REWARD_CFG = {
    'lambda1': 0.5, 'lambda2': 1.0,
    'lambda3': 0.8, 'lambda4': 0.6,
    'P_ref': 5000.0
}


def make_env():
    from src.environment.hetnet_env import HetNetEnv
    return HetNetEnv(DEFAULT_CFG, REWARD_CFG)


def test_reset_returns_correct_shape():
    env   = make_env()
    state = env.reset()
    assert state.shape == (env.state_dim,), \
        f"Expected ({env.state_dim},), got {state.shape}"


def test_step_returns_correct_types():
    env   = make_env()
    env.reset()
    action = env.action_space.sample()
    next_s, reward, done, info = env.step(action)
    assert isinstance(reward, float)
    assert isinstance(done,   bool)
    assert 'P_Net' in info


def test_power_nonnegative():
    env = make_env()
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        _, _, _, info = env.step(action)
        assert info['P_Net'] >= 0, "P_Net must be non-negative"
        assert info['P_AN']  >= 0, "P_AN must be non-negative"
        assert info['P_BH']  >= 0, "P_BH must be non-negative"


def test_state_finite():
    env   = make_env()
    state = env.reset()
    assert np.all(np.isfinite(state)), "State contains inf/nan"


# ======================================================================
# test_constraints.py
# ======================================================================

def make_handler():
    from src.environment.constraint_handler import ConstraintHandler
    return ConstraintHandler(DEFAULT_CFG, REWARD_CFG)


def test_c3_no_bidirectional():
    handler = make_handler()
    X_BH = np.array([[0,1,0],[1,0,0],[0,0,0]], dtype=float)
    probs = np.array([[0,0.9,0],[0.4,0,0],[0,0,0]], dtype=float)
    fixed = handler.enforce_c3(X_BH, probs)
    for i in range(3):
        for j in range(3):
            assert not (fixed[i,j] == 1 and fixed[j,i] == 1), \
                f"Bidirectional link ({i},{j}) found after C3 enforcement"


def test_c5_penalty_zero_when_feasible():
    handler    = make_handler()
    n_ue, n_bs = 5, 3
    ue_assoc   = np.zeros((n_ue, n_bs))
    prb_req    = np.ones((n_ue, n_bs)) * 10  # 10 PRBs each, well below 100
    for u in range(n_ue):
        ue_assoc[u, u % n_bs] = 1
    penalty = handler.penalty_c5(ue_assoc, prb_req)
    assert penalty == 0.0, f"Expected 0 penalty, got {penalty}"


def test_c5_penalty_positive_when_violated():
    handler  = make_handler()
    n_ue, n_bs = 5, 1
    ue_assoc = np.ones((n_ue, n_bs))
    prb_req  = np.ones((n_ue, n_bs)) * 30   # 150 total > 100 limit
    penalty  = handler.penalty_c5(ue_assoc, prb_req)
    assert penalty > 0.0, "Expected positive penalty for PRB violation"


def test_c6_clipping():
    from src.environment.constraint_handler import ConstraintHandler
    handler  = ConstraintHandler(DEFAULT_CFG, REWARD_CFG)
    p_BH_d   = np.array([50.0, 100.0, 200.0])
    T_max    = 80.0
    clipped  = handler.enforce_c6(p_BH_d, T_max)
    assert np.all(clipped <= T_max), "C6 clipping failed"
    assert np.all(clipped >= 0.0),   "C6 clipping produced negative values"


if __name__ == '__main__':
    # Run tests manually
    print("Testing environment...")
    test_reset_returns_correct_shape()
    test_step_returns_correct_types()
    test_power_nonnegative()
    test_state_finite()
    print("  ✓ Environment tests passed")

    print("Testing constraints...")
    test_c3_no_bidirectional()
    test_c5_penalty_zero_when_feasible()
    test_c5_penalty_positive_when_violated()
    test_c6_clipping()
    print("  ✓ Constraint tests passed")

    print("\nAll tests passed!")
