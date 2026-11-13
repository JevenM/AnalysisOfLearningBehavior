"""
simulate_oulad.py
-----------------
Generates a synthetic dataset that mirrors the structure and statistical
properties of the OULAD (Open University Learning Analytics Dataset).

Real OULAD is available at:
https://analyse.kmi.open.ac.uk/open_dataset

We simulate it here so the project runs without manual download steps.
All distributions are based on published OULAD descriptive statistics.
"""

import numpy as np
import pandas as pd

SEED = 42
N = 400  # number of students


def simulate_oulad(n: int = N, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # 1. Latent learner type (ground truth — never fed to the model)
    #    0 = Independent Deep Learner
    #    1 = Guided Learner
    #    2 = Exam-Driven Learner
    #    3 = At-Risk / Disengaged
    # ------------------------------------------------------------------ #
    types = rng.choice([0, 1, 2, 3], size=n, p=[0.25, 0.30, 0.25, 0.20])

    records = []
    for t in types:
        if t == 0:   # Independent Deep Learner
            records.append({
                "total_clicks"           : rng.integers(600, 1400),
                "active_days"            : rng.integers(50, 90),
                "avg_session_min"        : rng.uniform(35, 70),
                "forum_posts"            : rng.integers(8, 25),
                "resource_clicks"        : rng.integers(150, 400),   # PDF / articles
                "video_clicks"           : rng.integers(50, 150),
                "quiz_attempts"          : rng.integers(1, 3),
                "avg_quiz_score"         : rng.uniform(72, 95),
                "assignment_submit_lag"  : rng.uniform(1, 5),        # days before deadline
                "help_requests"          : rng.integers(0, 4),
                "login_freq_per_week"    : rng.uniform(4, 7),
            })
        elif t == 1:  # Guided Learner
            records.append({
                "total_clicks"           : rng.integers(400, 900),
                "active_days"            : rng.integers(35, 70),
                "avg_session_min"        : rng.uniform(20, 45),
                "forum_posts"            : rng.integers(3, 12),
                "resource_clicks"        : rng.integers(60, 180),
                "video_clicks"           : rng.integers(120, 350),   # prefers video
                "quiz_attempts"          : rng.integers(2, 5),
                "avg_quiz_score"         : rng.uniform(60, 80),
                "assignment_submit_lag"  : rng.uniform(0, 3),
                "help_requests"          : rng.integers(4, 12),
                "login_freq_per_week"    : rng.uniform(3, 6),
            })
        elif t == 2:  # Exam-Driven Learner
            records.append({
                "total_clicks"           : rng.integers(200, 600),
                "active_days"            : rng.integers(20, 45),
                "avg_session_min"        : rng.uniform(15, 35),
                "forum_posts"            : rng.integers(0, 5),
                "resource_clicks"        : rng.integers(30, 100),
                "video_clicks"           : rng.integers(20, 80),
                "quiz_attempts"          : rng.integers(4, 9),       # heavy quiz use
                "avg_quiz_score"         : rng.uniform(55, 78),
                "assignment_submit_lag"  : rng.uniform(-1, 1),       # last minute
                "help_requests"          : rng.integers(1, 6),
                "login_freq_per_week"    : rng.uniform(1, 4),
            })
        else:         # At-Risk / Disengaged
            records.append({
                "total_clicks"           : rng.integers(10, 200),
                "active_days"            : rng.integers(3, 20),
                "avg_session_min"        : rng.uniform(3, 18),
                "forum_posts"            : rng.integers(0, 2),
                "resource_clicks"        : rng.integers(5, 40),
                "video_clicks"           : rng.integers(2, 30),
                "quiz_attempts"          : rng.integers(0, 3),
                "avg_quiz_score"         : rng.uniform(25, 58),
                "assignment_submit_lag"  : rng.uniform(-5, 0),       # late or missing
                "help_requests"          : rng.integers(0, 3),
                "login_freq_per_week"    : rng.uniform(0, 2),
            })

    df = pd.DataFrame(records)
    df["latent_type"] = types   # kept only for evaluation, not used in clustering
    df.index.name = "student_id"
    df.index = ["S" + str(i).zfill(4) for i in range(n)]
    return df


if __name__ == "__main__":
    df = simulate_oulad()
    df.to_csv("oulad_synthetic.csv")
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(df["latent_type"].value_counts())
