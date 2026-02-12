"""
Null subject detection audit - PART 2: Deep dive.

Focuses on understanding why child null subject rate < adult,
and whether diary drop is being captured.
"""

import sys
sys.path.insert(0, ".")

import polars as pl
from analysis.corpus_descriptives.query import load_layered_corpus

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(120)

print("Loading corpus...")
lf = load_layered_corpus("data/output/test_10M/annotated_corpus/")
df = lf.collect()

print(f"Total sentences: {df.height}")
print()

# ============================================================
# KEY QUESTION: Where do the remaining null subjects come from
# after fragment/imperative exclusion?
# ============================================================
print("=" * 80)
print("WHERE DO CORRECTED has_null_subject=True SENTENCES COME FROM?")
print("=" * 80)

corrected_null = df.filter(pl.col("has_null_subject") == True)
print(f"Total corrected null subject sentences: {corrected_null.height}")
print()

print("--- By genre ---")
print(corrected_null.group_by("genre").agg(pl.len().alias("n")).sort("n", descending=True))
print()

print("--- By role ---")
print(corrected_null.group_by("role").agg(pl.len().alias("n")).sort("n", descending=True))
print()

# What do the CHILDES corrected null subjects look like?
for role_val in ["child", "adult"]:
    childes_null = corrected_null.filter(
        pl.col("role") == role_val
    )
    print(f"\n--- CHILDES {role_val}: {childes_null.height} corrected null subject sentences ---")

    # Show all of them (there aren't many)
    for row in childes_null.select(["text", "clauses", "is_fragment", "is_imperative", "genre", "pos_tags", "tokens"]).head(30).iter_rows(named=True):
        text = row["text"]
        clauses = row["clauses"]
        pos = row["pos_tags"][:8] if row["pos_tags"] else []
        tokens = row["tokens"][:8] if row["tokens"] else []
        clause_summary = []
        if clauses:
            for c in clauses:
                marker = " <<<" if c["subject_status"] == "none" and c["is_finite"] else ""
                clause_summary.append(f"    {c['clause_type']}|{c['verb_lemma']}|subj={c['subject_status']}|fin={c['is_finite']}{marker}")
        print(f"  TEXT: {text}")
        print(f"    POS: {list(zip(tokens, pos))}")
        for cs in clause_summary:
            print(cs)
        print()

# ============================================================
# CRITICAL: Check whether the imperative detector is too aggressive
# for children - are "want cookie", "go outside", "need milk" etc.
# being tagged as imperatives when they are really null subjects?
# ============================================================
print("=" * 80)
print("IMPERATIVE vs NULL SUBJECT AMBIGUITY")
print("Top verb lemmas in child 'imperatives' vs adult 'imperatives'")
print("=" * 80)

for role_val in ["child", "adult"]:
    imp_sents = df.filter(
        (pl.col("role") == role_val)
        & (pl.col("is_imperative") == True)
    )
    print(f"\n--- {role_val}: {imp_sents.height} imperative sentences ---")

    # Get the ROOT verb lemma distribution
    imp_verbs = (
        imp_sents.select(["clauses"])
        .explode("clauses")
        .filter(pl.col("clauses").is_not_null())
        .filter(pl.col("clauses").struct.field("clause_type") == "ROOT")
        .select(pl.col("clauses").struct.field("verb_lemma").alias("verb_lemma"))
        .group_by("verb_lemma")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .head(30)
    )
    print(imp_verbs)
    print()

# ============================================================
# CRITICAL: Compare verb-initial sentences by role to check
# for systematic diary drop misses
# ============================================================
print("=" * 80)
print("VERB-INITIAL ANALYSIS BY ROLE")
print("Sentences where pos_tags[0] == 'VERB', broken down by detection status")
print("=" * 80)

for role_val in ["child", "adult"]:
    verb_initial = df.filter(
        (pl.col("role") == role_val)
        & (pl.col("pos_tags").list.get(0) == "VERB")
    )
    total = verb_initial.height
    flagged_null = verb_initial.filter(pl.col("has_null_subject") == True).height
    flagged_imp = verb_initial.filter(pl.col("is_imperative") == True).height
    flagged_frag = verb_initial.filter(pl.col("is_fragment") == True).height
    has_subj_pron = verb_initial.filter(pl.col("has_overt_subject_pronoun") == True).height
    none_of_above = verb_initial.filter(
        (pl.col("has_null_subject") == False)
        & (pl.col("is_imperative") == False)
        & (pl.col("is_fragment") == False)
    ).height

    print(f"\n--- {role_val}: {total} verb-initial sentences ---")
    print(f"  has_null_subject=True: {flagged_null} ({flagged_null/total*100:.1f}%)")
    print(f"  is_imperative=True:   {flagged_imp} ({flagged_imp/total*100:.1f}%)")
    print(f"  is_fragment=True:     {flagged_frag} ({flagged_frag/total*100:.1f}%)")
    print(f"  has_overt_subj_pron:  {has_subj_pron} ({has_subj_pron/total*100:.1f}%)")
    print(f"  none of above:        {none_of_above} ({none_of_above/total*100:.1f}%)")

    # Sample the "none of above" - these might be diary drop
    uncaptured = verb_initial.filter(
        (pl.col("has_null_subject") == False)
        & (pl.col("is_imperative") == False)
        & (pl.col("is_fragment") == False)
    )
    if uncaptured.height > 0:
        print(f"\n  Sample of UNCAPTURED verb-initial sentences ({role_val}):")
        for row in uncaptured.select(["text", "clauses", "pos_tags", "tokens"]).head(15).iter_rows(named=True):
            text = row["text"]
            tokens = row["tokens"][:6] if row["tokens"] else []
            pos = row["pos_tags"][:6] if row["pos_tags"] else []
            clauses = row["clauses"]
            clause_summary = []
            if clauses:
                for c in clauses:
                    clause_summary.append(f"      {c['clause_type']}|{c['verb_lemma']}|subj={c['subject_status']}|fin={c['is_finite']}")
            print(f"    TEXT: {text}")
            print(f"      POS: {list(zip(tokens, pos))}")
            for cs in clause_summary:
                print(cs)
            print()

# ============================================================
# KEY ANALYSIS: Look at specific "diary drop" verbs
# "got", "went", "had", "saw", "said", "thought", "know"
# These are common first-person diary-drop verbs in CDS
# ============================================================
print("=" * 80)
print("DIARY DROP VERB ANALYSIS")
print("Looking at specific past tense verbs that commonly appear in diary drop")
print("=" * 80)

diary_verbs = ["get", "go", "have", "see", "say", "think", "know", "want", "need", "like"]

for role_val in ["child", "adult"]:
    print(f"\n--- {role_val} ---")
    for verb in diary_verbs:
        # Sentences with this verb as ROOT, finite, subject_status=="none"
        null_count = (
            df.filter(pl.col("role") == role_val)
            .select(["clauses", "is_imperative"])
            .explode("clauses")
            .filter(pl.col("clauses").is_not_null())
            .filter(
                (pl.col("clauses").struct.field("verb_lemma") == verb)
                & (pl.col("clauses").struct.field("is_finite") == True)
                & (pl.col("clauses").struct.field("clause_type") == "ROOT")
            )
            .select([
                pl.col("clauses").struct.field("subject_status").alias("subject_status"),
                "is_imperative",
            ])
        )
        total = null_count.height
        null_ct = null_count.filter(pl.col("subject_status") == "none").height
        null_imp = null_count.filter((pl.col("subject_status") == "none") & (pl.col("is_imperative") == True)).height
        null_nonimp = null_count.filter((pl.col("subject_status") == "none") & (pl.col("is_imperative") == False)).height
        overt_ct = null_count.filter(pl.col("subject_status") == "overt").height

        if total > 0:
            print(f"  {verb:10s}: total={total:5d}, null={null_ct:4d} (imp={null_imp:4d}, non-imp={null_nonimp:3d}), overt={overt_ct:5d}, null_rate={null_ct/total:.3f}")

# ============================================================
# SPECIFIC CHECK: "want" and "need" in child speech
# These are key null-subject indicators in child language acquisition
# "want cookie" = I want cookie (null subject, NOT imperative)
# ============================================================
print()
print("=" * 80)
print("WANT/NEED IN CHILD SPEECH - potential misclassification as imperative")
print("=" * 80)

for verb in ["want", "need", "like", "got", "know"]:
    child_verb_sents = (
        df.filter(
            (pl.col("role") == "child")
        )
        .select(["text", "clauses", "is_imperative", "is_fragment", "tokens", "pos_tags"])
        .explode("clauses")
        .filter(pl.col("clauses").is_not_null())
        .filter(
            (pl.col("clauses").struct.field("verb_lemma") == verb)
            & (pl.col("clauses").struct.field("clause_type") == "ROOT")
            & (pl.col("clauses").struct.field("subject_status") == "none")
        )
        .select([
            "text", "is_imperative", "is_fragment",
            pl.col("clauses").struct.field("is_finite").alias("is_finite"),
        ])
    )
    print(f"\n--- '{verb}' with null subject in child ROOT clauses: {child_verb_sents.height} ---")
    imp_count = child_verb_sents.filter(pl.col("is_imperative") == True).height
    frag_count = child_verb_sents.filter(pl.col("is_fragment") == True).height
    neither = child_verb_sents.filter((pl.col("is_imperative") == False) & (pl.col("is_fragment") == False)).height
    print(f"  imperative: {imp_count}, fragment: {frag_count}, neither: {neither}")

    # Show samples
    sample = child_verb_sents.head(10)
    for row in sample.iter_rows(named=True):
        print(f"  TEXT: {row['text']}")
        print(f"    is_imperative={row['is_imperative']}, is_fragment={row['is_fragment']}, is_finite={row['is_finite']}")

# ============================================================
# QUANTIFY THE PROBLEM: If "want/need/like/know/got" null subjects
# in child speech are being classified as imperatives, this would
# systematically suppress the child null-subject rate
# ============================================================
print()
print("=" * 80)
print("QUANTIFYING THE IMPERATIVE MISCLASSIFICATION IMPACT")
print("=" * 80)

non_imperative_verbs = {"want", "need", "like", "know", "got", "have", "think", "see", "hear", "say",
                         "mean", "remember", "feel", "wish", "hope", "guess", "bet", "suppose",
                         "believe", "wonder", "mind", "care"}

for role_val in ["child", "adult"]:
    all_imp = df.filter(
        (pl.col("role") == role_val)
        & (pl.col("is_imperative") == True)
    )

    # Get ROOT verb lemmas of imperatives
    imp_root_verbs = (
        all_imp.select(["text", "clauses"])
        .explode("clauses")
        .filter(pl.col("clauses").is_not_null())
        .filter(pl.col("clauses").struct.field("clause_type") == "ROOT")
        .select([
            "text",
            pl.col("clauses").struct.field("verb_lemma").alias("verb_lemma"),
        ])
    )

    suspicious = imp_root_verbs.filter(pl.col("verb_lemma").is_in(list(non_imperative_verbs)))

    print(f"\n--- {role_val} ---")
    print(f"  Total imperatives: {all_imp.height}")
    print(f"  Imperatives with suspicious verb lemma (want/need/know/etc.): {suspicious.height}")
    if all_imp.height > 0:
        print(f"  Percentage suspicious: {suspicious.height/all_imp.height*100:.1f}%")

    if suspicious.height > 0:
        # Top suspicious verbs
        print("  Top suspicious verb lemmas:")
        top = suspicious.group_by("verb_lemma").agg(pl.len().alias("n")).sort("n", descending=True).head(15)
        print(top)

        # Sample
        print(f"\n  Sample sentences (tagged imperative but verb suggests null subject):")
        for row in suspicious.select("text").head(10).iter_rows(named=True):
            print(f"    {row['text']}")

print()
print("=" * 80)
print("DONE")
print("=" * 80)
