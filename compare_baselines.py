"""
compare_baselines.py
--------------------
Compares DRL agent against all baselines from the paper:
  - Max-SINR
  - Min-PL
  - JUBSM
  - RUBSM
  - Proposed Heuristic (IBSBS)
  - DRL (ours)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.environment.hetnet_env import HetNetEnv


# ======================================================================
# Baseline algorithms
# ======================================================================

class MaxSINRBaseline:
    """UE associates to BS with maximum received SINR (Ref [17])."""
    def select_action(self, env):
        ue_assoc = np.zeros((env.n_ue, env.n_bs))
        for u in range(env.n_ue):
            best = np.argmax(env.sinr_matrix[u])
            ue_assoc[u, best] = 1.0
        return _build_action(env, ue_assoc, sleep_none=True)


class MinPLBaseline:
    """UE associates to nearest BS (minimum path loss, Ref [19])."""
    def select_action(self, env):
        ue_assoc = np.zeros((env.n_ue, env.n_bs))
        for u in range(env.n_ue):
            dists = np.linalg.norm(
                env.ue_positions[u] - env.bs_positions, axis=1
            )
            best = np.argmin(dists)
            ue_assoc[u, best] = 1.0
        return _build_action(env, ue_assoc, sleep_none=True)


class JUBSMBaseline:
    """Joint UE association, BH routing and switch-off (Ref [23])."""
    def select_action(self, env):
        ue_assoc = np.zeros((env.n_ue, env.n_bs))
        for u in range(env.n_ue):
            for i in range(env.n_bs):
                prb = env.prb_required[u, i]
                if prb <= env.Gamma_max:
                    ue_assoc[u, i] = 1.0
                    break
        # Switch off least-loaded s-BS (simplified)
        loads = np.sum(ue_assoc[:, 1:], axis=0)
        sleep = (loads == 0).astype(float)
        return _build_action(env, ue_assoc, sleep_sbs=sleep)


class RUBSMBaseline:
    """Robust UE association, BH routing and switch-off (Ref [24])."""
    def select_action(self, env):
        ue_assoc = np.zeros((env.n_ue, env.n_bs))
        for u in range(env.n_ue):
            # Use nominal + max deviation (scaled SINR)
            scores = env.sinr_matrix[u] - env.prb_required[u] * 0.01
            best   = np.argmax(scores)
            ue_assoc[u, best] = 1.0
        loads = np.sum(ue_assoc[:, 1:], axis=0)
        sleep = (loads < 1).astype(float)
        return _build_action(env, ue_assoc, sleep_sbs=sleep)


def _build_action(env, ue_assoc, sleep_none=False, sleep_sbs=None):
    """Helper to build full action vector from UE association."""
    n_bh  = env.n_sbs * env.n_bs
    bh    = np.random.uniform(0.4, 0.6, n_bh)  # neutral BH routing

    if sleep_none or sleep_sbs is None:
        sleep_sbs_v = np.zeros(env.n_sbs)       # all active
    else:
        sleep_sbs_v = sleep_sbs

    sleep_bh_v = np.zeros(env.n_sbs)            # all BH active

    return np.concatenate([
        ue_assoc.flatten(),
        bh,
        sleep_sbs_v,
        sleep_bh_v
    ]).astype(np.float32)


# ======================================================================
# Evaluation function
# ======================================================================

def evaluate_agent(agent_or_baseline, env, n_episodes=100, is_drl=False):
    """Run agent for n_episodes, return mean metrics."""
    powers, ees, active_sbs, active_bh = [], [], [], []

    for _ in range(n_episodes):
        state = env.reset()
        ep_powers = []

        for _ in range(50):  # short eval rollout
            if is_drl:
                action = agent_or_baseline.select_action(state, explore=False)
            else:
                action = agent_or_baseline.select_action(env)

            next_state, _, _, info = env.step(action)
            ep_powers.append(info['P_Net'])
            active_sbs.append(info['n_active_sbs'])
            active_bh.append(info['n_active_bh'])
            ees.append(info['EE'])
            state = next_state

        powers.append(np.mean(ep_powers))

    return {
        'power'     : np.mean(powers),
        'power_std' : np.std(powers),
        'EE'        : np.mean(ees),
        'active_sbs': np.mean(active_sbs),
        'active_bh' : np.mean(active_bh),
    }


# ======================================================================
# Plot comparison
# ======================================================================

def plot_comparison(results_by_ue, ue_counts, save_dir):
    """Bar and line charts comparing all algorithms."""
    os.makedirs(save_dir, exist_ok=True)
    algos  = list(next(iter(results_by_ue.values())).keys())
    colors = ['#e74c3c','#e67e22','#9b59b6','#3498db','#1abc9c','#2ecc71']

    # ── Power vs UEs ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (ax, metric, ylabel) in enumerate(zip(
        axes,
        ['power', 'EE', 'active_sbs'],
        ['Total Power (W)', 'Energy Efficiency (bits/J)', 'Active s-BSs']
    )):
        for algo, color in zip(algos, colors):
            vals = [results_by_ue[n][algo][metric] for n in ue_counts]
            ax.plot(ue_counts, vals, marker='o', label=algo,
                    color=color, linewidth=2)
        ax.set_xlabel('Number of UEs', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'{ylabel} vs UEs', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'baseline_comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {path}")

    # ── Print table ───────────────────────────────────────────────────
    print(f"\n{'Algorithm':<12} " +
          " ".join(f"UE={n:<6}" for n in ue_counts))
    print("-" * (12 + 9 * len(ue_counts)))
    for algo in algos:
        row = f"{algo:<12} "
        for n in ue_counts:
            p = results_by_ue[n][algo]['power']
            row += f"{p:<9.1f}"
        print(row)


# ======================================================================
# Main
# ======================================================================

def main():
    import yaml, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--model',  default='results/best_model.pth')
    parser.add_argument('--output', default='results/')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ue_counts = [20, 30, 40, 50, 60]
    results_by_ue = {}

    baselines = {
        'Max-SINR': MaxSINRBaseline(),
        'Min-PL'  : MinPLBaseline(),
        'JUBSM'   : JUBSMBaseline(),
        'RUBSM'   : RUBSMBaseline(),
    }

    # Load DRL agent if available
    drl_agent = None
    if os.path.exists(args.model):
        from src.agents.ddpg_agent import DDPGAgent
        # Temp env to get dims
        tmp_env  = HetNetEnv(config['network'], config['reward'])
        drl_agent = DDPGAgent(tmp_env.state_dim, tmp_env.action_dim)
        drl_agent.load(args.model)

    for n_ue in ue_counts:
        cfg = dict(config['network'])
        cfg['n_ue'] = n_ue
        env = HetNetEnv(cfg, config['reward'])

        results_by_ue[n_ue] = {}
        for name, baseline in baselines.items():
            r = evaluate_agent(baseline, env, n_episodes=50, is_drl=False)
            results_by_ue[n_ue][name] = r
            print(f"UE={n_ue} | {name:<10} | Power={r['power']:.1f}W")

        if drl_agent:
            r = evaluate_agent(drl_agent, env, n_episodes=50, is_drl=True)
            results_by_ue[n_ue]['DRL (ours)'] = r
            print(f"UE={n_ue} | {'DRL (ours)':<10} | Power={r['power']:.1f}W")

    plot_comparison(results_by_ue, ue_counts, args.output)


if __name__ == '__main__':
    main()
