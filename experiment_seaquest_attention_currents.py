"""Experiment S4 (Seaquest): object-attention agent (S3) trained with random currents (S2).

Combines the two previous changes: S3's object-centric attention policy with potential-based
surfacing shaping (`experiment_seaquest_object_attention.py`), trained in base Seaquest with S2's
random per-game currents that displace the submarine by 1 px per step in one of 8 directions
(`experiment_seaquest_currents.py`). Evaluation during training and checkpoint selection use the
plain base game. The held-out `gravity` mod is only evaluated afterwards with `evaluate_gravity.py`.
"""

import experiment_seaquest_currents as s2
import experiment_seaquest_object_attention as s3
import experiment_seaquest_ppo as s1
import seaquest_common as sc

AC, ACT_SAMPLE, ACT_GREEDY = s3.AC, s3.ACT_SAMPLE, s3.ACT_GREEDY


def get_parser():
    ap = s3.get_parser(name="seaquest_attention_currents")
    ap.add_argument("--p_none", type=float, default=0.25, help="probability of a game without current")
    ap.add_argument("--q_min", type=float, default=0.0)
    ap.add_argument("--q_max", type=float, default=1.0)
    return ap


def main():
    cfg = get_parser().parse_args()
    s1.main(cfg, make_train_env=lambda c: s2.CurrentWrapper(sc.make_env(), c),
            experiment="experiment_seaquest_attention_currents", agent=s3.AGENT, reward_fn=s3.reward_fn)


if __name__ == "__main__":
    main()
