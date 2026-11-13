# Learner Persona Detection via Behavioral Clustering

Identifying distinct student behavioral profiles from online learning activity data,
with the goal of informing **adaptive content design** for educational teams.

---

## Motivation

In online learning environments, students differ not just in academic performance
but in *how* they engage, when they study, what content types they gravitate toward,
how frequently they seek help, and whether they study consistently or only before exams.

These behavioral differences suggest that a one-size-fits-all content strategy is
suboptimal. This project explores whether unsupervised clustering of student
interaction data can reveal meaningful, actionable learner personas, and whether
those personas can guide content teams toward more targeted instructional design.

This work is directly informed by my experience designing a student behavioral data
infrastructure at a virtual university, where the core challenge was translating raw
interaction logs into pedagogically meaningful insights.

---

## Dataset

We use a **synthetic dataset** modeled on the statistical properties of the
[OULAD (Open University Learning Analytics Dataset)](https://analyse.kmi.open.ac.uk/open_dataset) —
a publicly available dataset from the UK Open University containing real behavioral
logs from thousands of online students.

The synthetic version (`data/simulate_oulad.py`) replicates OULAD's key behavioral
distributions across 400 student profiles, enabling reproducible experimentation
without requiring manual data download.

**Features used:**

| Feature | Description |
|---|---|
| `total_clicks` | Total interactions with learning platform |
| `active_days` | Number of days with at least one login |
| `avg_session_min` | Average session duration (minutes) |
| `forum_posts` | Number of posts in discussion forums |
| `resource_clicks` | Clicks on reading materials and PDFs |
| `video_clicks` | Clicks on video content |
| `quiz_attempts` | Total number of quiz/assessment attempts |
| `avg_quiz_score` | Average score across all assessments |
| `assignment_submit_lag` | Days before/after deadline of submission |
| `help_requests` | Number of help or support requests |
| `login_freq_per_week` | Average logins per week |

---

## Method

**Unsupervised K-Means Clustering** with the following pipeline:

```
Raw behavioral logs
      │
      ▼
StandardScaler (z-score normalization)
      │
      ▼
Elbow Method + Silhouette Analysis  →  K = 4
      │
      ▼
K-Means Clustering
      │
      ▼
PCA (2D projection for visualization)
      │
      ▼
Persona labeling + content recommendations
```

---

## Results

Four distinct learner personas emerged from the clustering:

### Cluster - Independent Deep Learner (26%)
**Behavioral signature:** High total engagement, long sessions, strong quiz scores,
minimal help-seeking, active forum participation.

**Content recommendation:** Research papers, open-ended projects, peer discussion
forums, and challenge-level assessments.

---

### Cluster 1 - Guided Learner (27%)
**Behavioral signature:** Moderate engagement, high video consumption, frequent
help requests, relies on structured instruction rather than self-directed exploration.

**Content recommendation:** Step-by-step video tutorials, worked examples,
immediate-feedback quizzes, and scaffolded learning paths.

---

### Cluster 2 - Exam-Driven Learner (19%)
**Behavioral signature:** Low baseline activity with spikes near assessment deadlines,
high quiz attempt rate, strategic and selective content access.

**Content recommendation:** Concise summaries, flashcards, practice exams,
spaced-repetition tools, and deadline-anchored reminders.

---

### Cluster 3 - At-Risk / Disengaged (28%)
**Behavioral signature:** Very low click counts and active days, poor quiz scores,
minimal forum or help interaction, high probability of dropout.

**Content recommendation:** Short micro-content units, gamification elements,
proactive push notifications, and early mentor intervention.

---

## Visualizations

| Plot | Description |
|---|---|
| `elbow_silhouette.png` | Elbow curve and silhouette scores for K selection |
| `pca_clusters.png` | 2D PCA projection of the four clusters |
| `radar_personas.png` | Behavioral radar charts per persona |
| `heatmap_features.png` | Z-score heatmap of features across clusters |

---

## Performance

| Metric | Value |
|---|---|
| Silhouette Score | 0.488 |
| Adjusted Rand Index (vs. latent types) | 1.000 |

---

## Project Structure

```
learner-profiling/
├── data/
│   └── simulate_oulad.py        # Synthetic OULAD-like data generator
├── analysis/
│   └── learner_profiling.py     # Full clustering pipeline
├── notebooks/
│   └── exploration.ipynb        # Interactive walkthrough
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/najigit/learner-profiling.git
cd learner-profiling
pip install -r requirements.txt
python analysis/learner_profiling.py
```

---

## Future Work

- Replace synthetic data with real OULAD download for validation
- Explore **Gaussian Mixture Models** for soft cluster membership
- Incorporate temporal dynamics, how personas shift over a semester
- Frame as a **reinforcement learning** problem: an agent that selects content
  type based on evolving student state, optimizing for long-term engagement
- Connect to neuroscientific models of learning and memory consolidation

---

