"""
learner_profiling.py
--------------------
Behavioral clustering of online students using K-Means.
Identifies four learner personas to support adaptive content design.

Pipeline:
  1. Load & inspect data
  2. Preprocess (scale)
  3. Determine optimal K (Elbow + Silhouette)
  4. Fit K-Means (K=4)
  5. Visualise clusters
  6. Interpret & label personas
  7. Export results
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

from simulate_oulad import simulate_oulad

#------------------------------------------
# 0. Style
#------------------------------------------
PALETTE = ["#4C72B0", "#55A868", "#C44E52", "#CCB974"]
PERSONA_NAMES = {
    0: "Independent\nDeep Learner",
    1: "Guided\nLearner",
    2: "Exam-Driven\nLearner",
    3: "At-Risk /\nDisengaged",
}
PERSONA_COLORS = {i: PALETTE[i] for i in range(4)}

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})


#------------------------------------------
# 1. Load data
#------------------------------------------
print("  Generating synthetic OULAD-like dataset …")
df_raw = simulate_oulad(n=400)

FEATURES = [
    "total_clicks", "active_days", "avg_session_min",
    "forum_posts", "resource_clicks", "video_clicks",
    "quiz_attempts", "avg_quiz_score",
    "assignment_submit_lag", "help_requests", "login_freq_per_week",
]
X = df_raw[FEATURES].copy()
y_true = df_raw["latent_type"].values   # only for evaluation


#------------------------------------------
# 2. Preprocess
#------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#------------------------------------------
# 3. Elbow + Silhouette
#------------------------------------------
print("  Computing Elbow & Silhouette scores …")
K_range = range(2, 10)
inertias, sil_scores = [], []

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Selecting Optimal Number of Clusters", fontsize=14, fontweight="bold")

axes[0].plot(K_range, inertias, "o-", color=PALETTE[0], linewidth=2)
axes[0].axvline(4, color=PALETTE[2], linestyle="--", alpha=0.7, label="K = 4")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia (Within-cluster SSE)")
axes[0].set_title("Elbow Method")
axes[0].legend()

axes[1].plot(K_range, sil_scores, "s-", color=PALETTE[1], linewidth=2)
axes[1].axvline(4, color=PALETTE[2], linestyle="--", alpha=0.7, label="K = 4")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Analysis")
axes[1].legend()

plt.tight_layout()
plt.savefig("elbow_silhouette.png", bbox_inches="tight")
plt.close()
print("   Saved → elbow_silhouette.png")


#------------------------------------------
# 4. Final K-Means (K = 4)
#------------------------------------------
print("  Fitting K-Means with K = 4 …")
km_final = KMeans(n_clusters=4, random_state=42, n_init=20)
cluster_labels = km_final.fit_predict(X_scaled)

ari = adjusted_rand_score(y_true, cluster_labels)
sil = silhouette_score(X_scaled, cluster_labels)
print(f"   Silhouette Score : {sil:.3f}")
print(f"   Adjusted Rand Index (vs latent types): {ari:.3f}")

df_raw["cluster"] = cluster_labels


#------------------------------------------
# 5a. PCA scatter
#------------------------------------------
print("  Generating PCA scatter plot …")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(9, 6))
for c in range(4):
    mask = cluster_labels == c
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=PALETTE[c], label=PERSONA_NAMES[c].replace("\n", " "),
               alpha=0.75, edgecolors="white", linewidths=0.4, s=60)

ax.set_xlabel(f"PC 1 ({var[0]:.1f}% variance)")
ax.set_ylabel(f"PC 2 ({var[1]:.1f}% variance)")
ax.set_title("Learner Personas — PCA Projection", fontsize=13, fontweight="bold")
ax.legend(title="Persona", framealpha=0.9)
plt.tight_layout()
plt.savefig("pca_clusters.png", bbox_inches="tight")
plt.close()
print("   Saved → pca_clusters.png")


#------------------------------------------
# 5b. Radar chart
#------------------------------------------
print("  Generating radar chart …")

RADAR_FEATURES = [
    "total_clicks", "active_days", "avg_session_min",
    "video_clicks", "quiz_attempts", "avg_quiz_score",
    "help_requests", "forum_posts",
]
RADAR_LABELS = [
    "Total Clicks", "Active Days", "Session Length",
    "Video Views", "Quiz Attempts", "Quiz Score",
    "Help Requests", "Forum Posts",
]

cluster_means = df_raw.groupby("cluster")[RADAR_FEATURES].mean()

# Normalise 0-1 per feature
norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-9)

angles = np.linspace(0, 2 * np.pi, len(RADAR_FEATURES), endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(2, 2, figsize=(12, 10), subplot_kw=dict(polar=True))
fig.suptitle("Behavioral Radar — Learner Personas", fontsize=14, fontweight="bold")

for idx, ax in enumerate(axes.flat):
    values = norm.loc[idx].tolist() + norm.loc[idx].tolist()[:1]
    ax.plot(angles, values, color=PALETTE[idx], linewidth=2)
    ax.fill(angles, values, color=PALETTE[idx], alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RADAR_LABELS, size=8)
    ax.set_yticklabels([])
    ax.set_title(PERSONA_NAMES[idx].replace("\n", " "),
                 color=PALETTE[idx], fontweight="bold", pad=14)

plt.tight_layout()
plt.savefig("radar_personas.png", bbox_inches="tight")
plt.close()
print("   Saved → radar_personas.png")


#------------------------------------------
# 5c. Feature heatmap
#------------------------------------------
print("  Generating feature heatmap …")

heatmap_data = df_raw.groupby("cluster")[FEATURES].mean()
heatmap_norm = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()
heatmap_norm.index = [PERSONA_NAMES[i].replace("\n", " ") for i in range(4)]

fig, ax = plt.subplots(figsize=(13, 4))
sns.heatmap(heatmap_norm, annot=True, fmt=".2f", cmap="RdYlGn",
            linewidths=0.5, ax=ax, cbar_kws={"label": "Z-score"})
ax.set_title("Feature Z-scores per Learner Persona", fontsize=13, fontweight="bold")
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("heatmap_features.png", bbox_inches="tight")
plt.close()
print("   Saved → heatmap_features.png")


#------------------------------------------
# 6. Persona summary table
#------------------------------------------
print("\n" + "═" * 60)
print("  LEARNER PERSONA SUMMARY")
print("═" * 60)

PERSONA_DESC = {
    0: ("High engagement, deep content exploration, "
        "strong quiz scores, minimal help requests."),
    1: ("Moderate engagement, high video consumption, "
        "frequent help requests, relies on guided instruction."),
    2: ("Low general activity, spikes near assessments, "
        "high quiz attempt rate, strategic content access."),
    3: ("Very low clicks and active days, poor quiz scores, "
        "minimal interaction — high dropout risk."),
}

CONTENT_REC = {
    0: "Research papers, open-ended projects, peer discussion forums.",
    1: "Step-by-step video tutorials, worked examples, immediate feedback quizzes.",
    2: "Concise summaries, flashcards, practice exams, spaced-repetition tools.",
    3: "Short micro-content, gamification, push notifications, mentor check-ins.",
}

for c in range(4):
    n_students = (cluster_labels == c).sum()
    print(f"\n  Cluster {c} — {PERSONA_NAMES[c].replace(chr(10), ' ')} "
          f"(n = {n_students}, {100*n_students/len(cluster_labels):.0f}%)")
    print(f"  Profile    : {PERSONA_DESC[c]}")
    print(f"  Content Rx : {CONTENT_REC[c]}")

print("\n" + "═" * 60)

# Export labelled dataset
out_path = "learner_profiles.csv"
df_export = df_raw.copy()
df_export["persona"] = df_export["cluster"].map(
    lambda c: PERSONA_NAMES[c].replace("\n", " "))
df_export.to_csv(out_path)
print(f"\n  Full results saved → {out_path}")
print("✓ Pipeline complete.")
