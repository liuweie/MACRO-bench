import json
import re
import difflib
import pandas as pd
from sklearn.metrics import cohen_kappa_score


# =========================
# 1. Score sequence parsing (robust)
# =========================

def parse_score_sequence(score_str):
    """
    Parse '1;0;1' / '1；0；1' / '1,0,1' / '1 0 1' into [1,0,1].
    """
    if score_str is None or pd.isna(score_str):
        return None

    tokens = re.split(r"[;；,，|\s]+", str(score_str).strip())
    scores = []

    for t in tokens:
        if t == "":
            continue
        if t not in {"0", "1"}:
            raise ValueError(f"Invalid score token: {t}")
        scores.append(int(t))

    return scores


# =========================
# 2. Load human Excel
# =========================

def load_human_annotations(xlsx_path, sheet_name="human-lhj"):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    records = []

    for _, row in df.iterrows():
        query = str(row["query"]).strip()

        try:
            sys_scores = parse_score_sequence(row["sys_scores"])
            user_scores = parse_score_sequence(row["user_scores"])
        except Exception as e:
            print(f"[Human Parse Error] Query='{query[:60]}...' | {e}")
            continue

        records.append({
            "query": query,
            "system_scores": sys_scores,
            "user_scores": user_scores,
        })

    print(f"[Human] Loaded {len(records)} records")
    return records


# =========================
# 3. Load LLM JSON
# =========================

def load_llm_annotations(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []

    for item in data.get("detailed_results", []):
        llm_scores = item.get("llm_milestone_scores", {})
        if not llm_scores:
            continue

        # Use passed_flags
        user_side = llm_scores.get("user_side", {})
        system_side = llm_scores.get("system_side", {})

        user_flags = user_side.get("passed_flags", [])
        system_flags = system_side.get("passed_flags", [])

        # Skip records without any milestones
        if not user_flags and not system_flags:
            continue

        records.append({
            "query": item.get("initial_query", "").strip(),
            "transaction_id": item.get("transaction_id"),
            "task_id": item.get("matched_task_id"),
            "user_scores": [int(x) for x in user_flags],
            "system_scores": [int(x) for x in system_flags],
        })

    print(f"[LLM] Loaded {len(records)} records (llm_milestone_scores)")
    return records


# =========================
# 4. Query normalization & matching
# =========================

def normalize_query(q: str):
    q = q.lower().strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^\w\s]", "", q)
    return q


def match_human_to_llm(human_records, llm_records, threshold=0.92):
    llm_index = {}

    for r in llm_records:
        nq = normalize_query(r["query"])
        llm_index[nq] = r

    matched = []
    dropped = []

    for h in human_records:
        hq_norm = normalize_query(h["query"])

        # 1️⃣ exact normalized match
        if hq_norm in llm_index:
            matched.append((h, llm_index[hq_norm]))
            continue

        # 2️⃣ fuzzy match
        best_score = 0
        best_match = None

        for nq, lr in llm_index.items():
            score = difflib.SequenceMatcher(None, hq_norm, nq).ratio()
            if score > best_score:
                best_score = score
                best_match = lr

        if best_score >= threshold:
            matched.append((h, best_match))
        else:
            dropped.append(h["query"])

    print(f"[Match] matched={len(matched)}, dropped={len(dropped)}")
    return matched


# =========================
# 5. Milestone alignment
# =========================

def align_milestones(matched_pairs):
    rows = []

    for human, llm in matched_pairs:
        for side in ["system", "user"]:
            h_scores = human.get(f"{side}_scores")
            l_scores = llm.get(f"{side}_scores")

            if h_scores is None or l_scores is None:
                continue

            if len(h_scores) != len(l_scores):
                print(
                    f"[Length Mismatch] Query='{human['query'][:60]}...' "
                    f"{side}: human={len(h_scores)}, llm={len(l_scores)}"
                )

            # Align using the shorter length
            min_len = min(len(h_scores), len(l_scores))
            for idx in range(min_len):
                rows.append({
                    "query": human["query"],
                    "task_id": llm["task_id"],
                    "side": side,
                    "milestone_idx": idx,
                    "human_label": h_scores[idx],
                    "llm_label": l_scores[idx],
                })

    df = pd.DataFrame(rows)
    print(f"[Aligned] total milestone pairs = {len(df)}")
    return df


# =========================
# 6. Main entry
# =========================

def build_alignment_table(
    human_xlsx,
    llm_json,
    sheet_name="human-lhj",
    output_csv="./trace_output/exp_consistency/aligned_milestones.csv"
):
    human = load_human_annotations(human_xlsx, sheet_name)
    llm = load_llm_annotations(llm_json)

    matched = match_human_to_llm(human, llm)
    df = align_milestones(matched)

    df.to_csv(output_csv, index=False)
    print(f"[Saved] {output_csv}")

    return df


# =========================
# 7. Compute metrics
# =========================

def raw_agreement(df):
    return (df["human_label"] == df["llm_label"]).mean()


def cohen_kappa(df):
    return cohen_kappa_score(df["human_label"], df["llm_label"])


def compute_overall_metrics(df):
    ra = raw_agreement(df)
    kappa = cohen_kappa(df)

    print("===== Overall Milestone-level Agreement =====")
    print(f"Raw Agreement : {ra:.4f}")
    print(f"Cohen's Kappa : {kappa:.4f}")

    return {
        "raw_agreement": ra,
        "cohen_kappa": kappa,
    }


def compute_metrics_by_side(df):
    results = {}

    print("\n===== Agreement by Side =====")

    for side in ["user", "system"]:
        sub_df = df[df["side"] == side]

        if len(sub_df) == 0:
            continue

        ra = raw_agreement(sub_df)
        kappa = cohen_kappa(sub_df)

        print(f"[{side.upper()}]")
        print(f"  Raw Agreement : {ra:.4f}")
        print(f"  Cohen's Kappa : {kappa:.4f}")

        results[side] = {
            "raw_agreement": ra,
            "cohen_kappa": kappa,
            "num_milestones": len(sub_df),
        }

    return results


def print_basic_stats(df):
    print("\n===== Basic Stats =====")
    print(f"Total milestones : {len(df)}")
    if "side" in df.columns:
        print(df["side"].value_counts())
    print("\nLabel distribution (Human):")
    print(df["human_label"].value_counts(normalize=True))
    print("\nLabel distribution (LLM):")
    print(df["llm_label"].value_counts(normalize=True))

def extract_disagreement_cases(
    df,
    output_csv="./trace_output/exp_consistency/disagreement_cases.csv",
    output_query_csv="./trace_output/exp_consistency/disagreement_queries.csv",
):
    """
    Export disagreement cases between human and LLM:
    - milestone-level details
    - query-level summary
    """

    # =========================
    # 1) milestone-level disagreements
    # =========================
    disagree_df = df[df["human_label"] != df["llm_label"]].copy()

    print("\n===== Disagreement Stats =====")
    print(f"Total disagreement milestones : {len(disagree_df)}")

    if len(disagree_df) == 0:
        print("No disagreement cases found 🎉")
        return None, None

    disagree_df.to_csv(output_csv, index=False)
    print(f"[Saved] Milestone-level disagreements -> {output_csv}")

    # =========================
    # 2) query-level summary
    # =========================
    query_summary = (
        disagree_df
        .groupby(["query", "task_id"])
        .agg(
            num_disagreements=("milestone_idx", "count"),
            sides_involved=("side", lambda x: sorted(set(x))),
        )
        .reset_index()
        .sort_values("num_disagreements", ascending=False)
    )

    query_summary.to_csv(output_query_csv, index=False)
    print(f"[Saved] Query-level disagreements -> {output_query_csv}")

    return disagree_df, query_summary

if __name__ == "__main__":
    HUMAN_XLSX = "./trace_output/exp_consistency/oneflow_20260105_human_eval.xlsx"
    LLM_JSON = "./trace_output/exp_consistency/eval_results_detailed_20260105.json"

    df = build_alignment_table(
        human_xlsx=HUMAN_XLSX,
        llm_json=LLM_JSON,
        sheet_name="human-lhj",
        output_csv="./trace_output/exp_consistency/aligned_milestones.csv",
    )

    # ===== Metrics =====
    print_basic_stats(df)
    compute_overall_metrics(df)
    compute_metrics_by_side(df)

    extract_disagreement_cases(df)
