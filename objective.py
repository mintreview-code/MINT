"""Selectable cumulative-rank objective; cached predictor ranks are unchanged."""

THRESHOLDS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
MODE = "broad2"


def configure(mode="broad2"):
    global MODE
    if mode not in ("broad2", "weighted8"):
        raise ValueError("objective_mode must be broad2 or weighted8")
    MODE = mode


def cumulative(ranks):
    ranks = list(ranks)
    return [sum(rank < threshold for rank in ranks) for threshold in THRESHOLDS]


def vector(profile):
    if "cutoff_counts" in profile:
        return [profile["cutoff_counts"][str(t)] for t in THRESHOLDS]
    if "rows" in profile:
        return cumulative(row["rank"] for row in profile["rows"])
    raise ValueError("All eight thresholds required; cannot infer from three counts")


def vector_score(counts):
    return sum(counts) if MODE == "weighted8" else counts[-1]


def score(profile):
    return sum(vector(profile)) if MODE == "weighted8" else profile["below_2"]


def annotate(profile):
    counts = vector(profile)
    profile["cutoff_counts"] = dict(zip(map(str, THRESHOLDS), counts))
    profile["weighted_sum"] = sum(counts)
    profile["objective_mode"] = MODE
    profile["objective_score"] = vector_score(counts)
    return profile
