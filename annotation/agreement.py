"""Inter-annotator agreement computation."""

from typing import Any, Dict, List, Optional, Set, Tuple

from . import db


def _cohens_kappa(labels_a: List, labels_b: List) -> float:
    """Compute Cohen's kappa for two label sequences."""
    assert len(labels_a) == len(labels_b)
    n = len(labels_a)
    if n == 0:
        return 1.0

    all_labels = sorted(set(labels_a) | set(labels_b))
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    k = len(all_labels)

    # Confusion matrix
    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(labels_a, labels_b):
        matrix[label_to_idx[a]][label_to_idx[b]] += 1

    # Observed agreement
    po = sum(matrix[i][i] for i in range(k)) / n

    # Expected agreement
    pe = 0.0
    for i in range(k):
        row_sum = sum(matrix[i])
        col_sum = sum(matrix[j][i] for j in range(k))
        pe += (row_sum * col_sum) / (n * n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def _jaccard(set_a: Set, set_b: Set) -> float:
    """Jaccard index between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def compute_agreement(db_path=None) -> Dict[str, Any]:
    """Compute inter-annotator agreement metrics.

    Returns binary agreement, position agreement, label agreement,
    and per-stimulus breakdown.
    """
    with db.get_conn(db_path) as conn:
        # Get stimuli with exactly 2 annotations
        stim_rows = conn.execute(
            """
            SELECT stimulus_id, COUNT(DISTINCT user_id) AS cnt
            FROM annotations
            GROUP BY stimulus_id
            HAVING cnt >= 2
            """
        ).fetchall()

        if not stim_rows:
            return {
                "n_doubly_annotated": 0,
                "binary": {"percent_agreement": None, "kappa": None},
                "position": {"exact_match_rate": None, "mean_jaccard": None},
                "label": {"exact_match_rate": None, "kappa": None},
                "per_stimulus": [],
            }

        binary_a = []
        binary_b = []
        position_exact_matches = 0
        position_jaccards = []
        label_matches = 0
        label_comparisons = 0
        label_a_list = []
        label_b_list = []
        per_stimulus = []
        both_yes_count = 0

        for row in stim_rows:
            sid = row["stimulus_id"]
            anns = conn.execute(
                """
                SELECT a.*, u.username
                FROM annotations a
                JOIN users u ON u.id = a.user_id
                WHERE a.stimulus_id = ?
                ORDER BY a.user_id
                LIMIT 2
                """,
                (sid,),
            ).fetchall()

            if len(anns) < 2:
                continue

            a, b = anns[0], anns[1]
            a_yes = bool(a["has_null_subject"])
            b_yes = bool(b["has_null_subject"])
            binary_a.append(int(a_yes))
            binary_b.append(int(b_yes))

            # Classify disagreement
            category = "full_agreement"

            if a_yes != b_yes:
                category = "binary_disagreement"
            elif a_yes and b_yes:
                both_yes_count += 1
                # Compare positions
                a_pronouns = conn.execute(
                    "SELECT * FROM placed_pronouns WHERE annotation_id = ? ORDER BY ordering",
                    (a["id"],),
                ).fetchall()
                b_pronouns = conn.execute(
                    "SELECT * FROM placed_pronouns WHERE annotation_id = ? ORDER BY ordering",
                    (b["id"],),
                ).fetchall()

                a_positions = {p["position"] for p in a_pronouns}
                b_positions = {p["position"] for p in b_pronouns}

                if a_positions == b_positions:
                    position_exact_matches += 1
                    # Compare labels at matching positions
                    a_labels = {p["position"]: p["label"] for p in a_pronouns}
                    b_labels = {p["position"]: p["label"] for p in b_pronouns}
                    all_match = True
                    for pos in a_positions:
                        la = a_labels.get(pos)
                        lb = b_labels.get(pos)
                        if la and lb:
                            label_a_list.append(la)
                            label_b_list.append(lb)
                            label_comparisons += 1
                            if la == lb:
                                label_matches += 1
                            else:
                                all_match = False
                    if not all_match:
                        category = "label_disagreement"
                else:
                    category = "position_disagreement"

                jacc = _jaccard(a_positions, b_positions)
                position_jaccards.append(jacc)

            per_stimulus.append({
                "stimulus_id": sid,
                "category": category,
                "annotator_a": a["username"],
                "annotator_b": b["username"],
                "a_decision": "yes" if a_yes else "no",
                "b_decision": "yes" if b_yes else "no",
            })

        n = len(binary_a)
        binary_agree = sum(1 for x, y in zip(binary_a, binary_b) if x == y)
        binary_pct = binary_agree / n if n else None
        binary_kappa = _cohens_kappa(binary_a, binary_b) if n else None

        pos_exact = (
            position_exact_matches / both_yes_count if both_yes_count else None
        )
        mean_jacc = (
            sum(position_jaccards) / len(position_jaccards)
            if position_jaccards
            else None
        )

        label_exact = (
            label_matches / label_comparisons if label_comparisons else None
        )
        label_kappa = (
            _cohens_kappa(label_a_list, label_b_list)
            if label_comparisons
            else None
        )

        return {
            "n_doubly_annotated": n,
            "binary": {
                "percent_agreement": binary_pct,
                "kappa": binary_kappa,
            },
            "position": {
                "exact_match_rate": pos_exact,
                "mean_jaccard": mean_jacc,
            },
            "label": {
                "exact_match_rate": label_exact,
                "kappa": label_kappa,
            },
            "per_stimulus": per_stimulus,
        }
