# TTQ Corpus Report: Italian Pronoun Recovery Training Data

## 1. What is TTQ?

**TTQ** stands for **TED2020 + Tatoeba + QED** — three parallel EN-IT corpora combined into a single alignment dataset for generating Italian pronoun recovery training data.

| Source Corpus | Pairs | Register |
|---------------|-------|----------|
| TED2020 | 373K | Personal narrative, public talks |
| Tatoeba | 502K | Everyday conversational sentences |
| QED | 580K | Educational content, diverse topics |
| **Total** | **1,455,049** | **Mixed naturalistic** |

### Why TTQ?

The existing training data came from **Europarl** (European Parliament proceedings, 1.9M sentence pairs). Europarl has a systematic weakness: the formal parliamentary register avoids 1st and 2nd person pronouns through passivisation and impersonal constructions. When an Italian speaker says _"Dichiaro ripresa la sessione"_ (dropping _io_), the English translation becomes _"The session is declared reopened"_ — no "I" appears in either language, so the EN-IT alignment methodology cannot detect the null subject.

This creates a gold data distribution heavily skewed toward 1st person plural (the parliamentary "we") and starved of 2nd and 3rd person examples:

| Label | Europarl (50K pilot) | % | TTQ (1.45M) | % |
|-------|---------------------|---|-------------|---|
| PRO.1pl | 5,380 | 68.1 | 99,000 | 39.4 |
| PRO.1sg | 2,176 | 27.6 | 76,000 | 30.3 |
| PRO.3pl | 190 | 2.4 | 23,000 | 9.2 |
| PRO.3sg | 106 | 1.3 | 21,000 | 8.4 |
| PRO.2sg | 4 | 0.1 | 16,000 | 6.4 |
| PRO.2pl | 42 | 0.5 | 16,000 | 6.4 |
| **Total** | **7,898** | | **251,000** | |

The shift is dramatic. In Europarl, 2nd and 3rd person labels account for just **4.3%** of markers. In TTQ, they account for **30.3%** — a 7x improvement in non-1st-person coverage. TED talks include personal address ("you"), QED educational content uses 3rd person description, and Tatoeba's conversational sentences span all persons naturally.

## 2. Construction Pipeline

### 2.1 Source Preparation

The three corpora were obtained as line-aligned EN-IT parallel text files and concatenated:

```
data/italian/ttq/
├── TED2020.en-it.en / .it   (373K pairs)
├── Tatoeba.en-it.en / .it   (502K pairs)
├── QED.en-it.en / .it       (580K pairs)
├── ttq.en                    (1,455,049 lines, concatenated)
└── ttq.it                    (1,455,049 lines, concatenated)
```

The concatenated files were uploaded to a Kubernetes persistent volume claim (`europarl-sweep-data`) via `k8s/upload-ttq-data.sh` (184 MB total).

### 2.2 Alignment Pipeline

The full annotation was run as a single K8s job (`k8s/job-ttq-data-gen.yaml`) on an NVIDIA A10 GPU with 8 GiB memory. The job uses the same `EuroparlAlignmentGenerator` that was developed and validated on Europarl, configured with larger batch sizes to exploit the GPU:

| Parameter | Europarl | TTQ |
|-----------|----------|-----|
| Input pairs | 50K (pilot) / 500K (train) | 1,455,049 |
| spaCy batch size | 50 | 128 |
| Alignment batch size | 32 | 128 |
| GPU | A10 | A10 |
| Memory | 8 GiB | 8 GiB |

The pipeline processes pairs in chunks of 2,000 with checkpoint writes every 10,000 records.

### 2.3 Processing Steps Per Pair

Each EN-IT sentence pair passes through six stages:

1. **Dual-language parse**: English with `en_core_web_trf` (transformer-based, highest accuracy for subject detection), Italian with `it_core_news_lg`.

2. **Quality filter**: Rejects empty, too-short (<3 tokens), too-long (>128 tokens), or extreme-ratio (>3x) pairs. Europarl pilot: 536/50,000 pairs rejected (1.1%).

3. **English pronoun extraction**: Finds `nsubj` + `PRON` tokens from {I, we, he, she, they, you}. Filters out relative pronouns; skips all instances of "it" (too noisy — could be expletive or referential).

4. **Italian verb detection**: Classifies every `VerbForm=Fin` token's subject status (overt, clausal, expletive, inherited, null) and extracts Person/Number morphology.

5. **Word alignment**: `aneuraz/awesome-align-with-co` (fine-tuned multilingual BERT) produces token-level EN-IT alignments via softmax extraction at layer 8.

6. **Label resolution**: For each English pronoun, follows the alignment to the corresponding Italian token. If the aligned token is not a verb, walks up the dependency tree (max 3 hops). Skips verbs with overt subjects. Derives the label from the English pronoun text; for "you", uses Italian verb morphology to disambiguate 2sg/2pl. Cross-checks the label against Italian morphology and discards disagreements. Assigns confidence: "high" if morphology agrees, "medium" if no morphology available.

### 2.4 Passage Packing

Per-sentence records are grouped into multi-sentence passages (max 180 words, ~234 subword tokens at 1.3x expansion, fitting under the 256-token training window). Non-marker sentences are included as discourse context. Empty lines (speaker/debate boundaries) force a passage break. Only passages containing at least one marker are emitted.

## 3. Annotation Results

### 3.1 Yield

| Metric | Value |
|--------|-------|
| Input pairs | 1,455,049 |
| Aligned records (with markers) | 227,790 |
| Yield rate | 15.7% |
| Packed passages | 77,553 |
| Total markers | ~251,000 |
| Avg markers per record | ~1.1 |

The 15.7% yield rate substantially exceeds the initial estimate of ~6% (which was based on the Europarl pilot). The naturalistic registers of TED/Tatoeba/QED produce more sentences with overt English pronouns aligned to Italian null subjects than formal parliamentary text.

### 3.2 Output Format

```
/mnt/data/pronoun_recovery/europarl_aligned/it_ttq/
├── aligned_checkpoint.jsonl   # 227,790 per-sentence records
├── packed_checkpoint.jsonl    # 77,553 packed passages (~180 words)
└── pilot_statistics.json      # yield stats, label distribution
```

Each packed record:
```json
{
  "clean_text": "Dichiaro ripresa la sessione del Parlamento europeo.",
  "markers": [{"label": "PRO.1sg", "lexical_form": "io", "position": 0}],
  "id": "europarl_passage:42"
}
```

### 3.3 Label Distribution

| Label | Count | % | Europarl % | Improvement |
|-------|-------|---|------------|-------------|
| PRO.1pl | 99,000 | 39.4 | 68.1 | More balanced |
| PRO.1sg | 76,000 | 30.3 | 27.6 | Similar |
| PRO.3pl | 23,000 | 9.2 | 2.4 | **3.8x** |
| PRO.3sg | 21,000 | 8.4 | 1.3 | **6.3x** |
| PRO.2sg | 16,000 | 6.4 | 0.1 | **128x** |
| PRO.2pl | 16,000 | 6.4 | 0.5 | **12x** |

The most striking improvements are in 2nd person singular (PRO.2sg: 4 examples in Europarl pilot vs 16,000 in TTQ) and 3rd person singular (PRO.3sg: 106 vs 21,000). These are exactly the categories where Europarl's parliamentary register fails.

### 3.4 Comparison with Europarl Data

| | Europarl (full train) | TTQ |
|---|---|---|
| Source pairs | 500,000 | 1,455,049 |
| Aligned records | ~72,000 (est.) | 227,790 |
| Packed passages | ~24,000 (est.) | 77,553 |
| Register | Formal parliamentary | Mixed naturalistic |
| 1st person % | ~96% | ~70% |
| 2nd/3rd person % | ~4% | ~30% |
| Confidence | 100% high | 100% high |

## 4. Why This Matters: The Europarl Gold Gap Problem

The full analysis of Europarl's limitations is documented in `data/pronoun_recovery/tree_detector/it/fp_analysis/gold_gap_report.md`. In summary:

When the tree detector was evaluated against Europarl gold labels, 596 apparent false positives were produced. Manual categorization revealed that **55.9% (333/596) were not model errors** — they were genuine Italian null subjects that the gold data failed to capture because the English source text also avoided the pronoun.

The root mechanism is register-driven: English parliamentary speech systematically restructures 1st-person constructions into passives and impersonals ("it is declared" instead of "I declare"), making the pronoun invisible to EN-IT alignment. Two remediation passes were applied to `label_aligner.py`:

1. **Structural propagation** — propagating gold labels along aux/cop/conj chains recovered 507 labels.
2. **Morphological heuristic** — relabelling 1st/2nd person finite verbs with no overt subject (and not imperative/xcomp) recovered 1,830 labels.

These corrections improved the tree detector from F1 0.802/0.822 (DT/HGB) to **F1 0.897/0.905**.

TTQ addresses this problem from the data side rather than the label-correction side. By drawing from registers where speakers naturally use all persons (TED talks with "you" and "they", Tatoeba conversations with "tu" and "io"), the alignment pipeline can capture null subjects across the full person/number spectrum without needing heuristic correction.

## 5. Implications for Tree Detector Training

The existing tree detector was trained on Europarl data (7,294 records, 20,476 verb feature rows after label corrections). Its performance on Europarl test data:

| Model | F1 | Precision | Recall |
|-------|----|-----------||--------|
| HGB | 0.905 | 0.886 | 0.926 |
| DT | 0.897 | 0.871 | 0.926 |

The per-label recall reveals the limitation: 1st-person labels achieve 93-94% recall, but 3rd-person labels only reach 69-81% — partly because there are so few 3rd-person examples in training (59 PRO.3pl, 27 PRO.3sg in the test set).

TTQ training data should improve this in two ways:

1. **Volume**: 227,790 aligned records vs ~7,300 from Europarl — roughly 30x more data for feature extraction and label alignment.

2. **Balance**: 2nd/3rd person labels go from ~4% to ~30% of the marker distribution. The tree detector should see enough 3rd-person examples to learn the structural patterns (relative clauses, impersonal constructions, copular predicates) that distinguish genuine 3rd-person null subjects from non-referential uses.

### Next Steps

A new tree detector config pointing to the TTQ data path would enable retraining:

```yaml
# configs/analysis/pronoun_recovery/pronoun_recovery_it_tree_detector_ttq.yaml
aligned_data_path: data/pronoun_recovery/europarl_aligned/it_ttq/aligned_checkpoint.jsonl
output_path: data/pronoun_recovery/tree_detector/it_ttq
```

Expected improvements:
- PRO.3sg/3pl recall: 70-81% (Europarl) -> potentially 85-90%+ (TTQ)
- PRO.2sg: 10 test examples (Europarl) -> hundreds (TTQ), enabling reliable evaluation
- Overall F1: likely similar or improved, with the gains concentrated in underrepresented labels
- The morphological heuristic in `label_aligner.py` may need less aggressive correction for TTQ data, since the naturalistic registers produce more directly-aligned 1st/2nd person markers

### Combined Training

The most promising approach would be training on Europarl + TTQ combined (~300K aligned records, ~100K packed passages), giving the detector exposure to both formal parliamentary and naturalistic registers. This would produce the most robust classifier for application to the full 90M Italian corpus, which spans 8 genres including Europarl, web text, spoken corpora, and literary sources.

## 6. Technical Details

### Infrastructure

- **K8s cluster**: `lemn-lab` namespace
- **GPU**: NVIDIA A10 (single)
- **PVC**: `europarl-sweep-data` mounted at `/mnt/data`
- **Container**: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime`
- **spaCy models**: `en_core_web_trf-3.7.3` (EN), `it_core_news_lg-3.7.0` (IT)
- **Alignment model**: `aneuraz/awesome-align-with-co`
- **CuPy**: `cupy-cuda12x<14` (for spaCy GPU with numpy compat)

### Config

```yaml
# configs/analysis/pronoun_recovery/pronoun_recovery_it_ttq_align.yaml
europarl_en_path: data/italian/ttq/ttq.en
europarl_it_path: data/italian/ttq/ttq.it
output_path: data/pronoun_recovery/europarl_aligned/it_ttq
language: it
en_spacy_model: en_core_web_trf
it_spacy_model: it_core_news_lg
spacy_batch_size: 128
align_model: aneuraz/awesome-align-with-co
align_batch_size: 128
skip_all_it: true
pack_passages: true
max_passage_words: 180
```

### Key Files

| File | Description |
|------|-------------|
| `configs/analysis/pronoun_recovery/pronoun_recovery_it_ttq_align.yaml` | TTQ alignment config |
| `k8s/job-ttq-data-gen.yaml` | K8s job definition |
| `k8s/upload-ttq-data.sh` | PVC upload script |
| `analysis/pronoun_recovery/parallel_data/generator.py` | Alignment pipeline |
| `analysis/pronoun_recovery/parallel_data/passage_packer.py` | Passage packing |
| `analysis/pronoun_recovery/parallel_data/label_resolver.py` | Label resolution |
| `analysis/pronoun_recovery/tree_detector/label_aligner.py` | Gold label correction |
| `analysis/pronoun_recovery/tree_detector/feature_extractor.py` | 52-feature extraction |
| `analysis/pronoun_recovery/tree_detector/trainer.py` | DT/HGB training |
| `analysis/pronoun_recovery/tree_detector/inference.py` | Production inference |
