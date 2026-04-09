# Spanish BebeLM Corpus & Parallel Data

## Overview

Spanish-language corpus assembly for BebeLM language model training and
Spanish-English parallel corpora for pronoun recovery model training.
Built 2026-02-20/22.

## Directory Structure

```
data/spanish/
├── raw/                  # Downloaded monolingual source files
├── train_100M/           # Assembled 100M-word corpus (after assembly)
├── train_90M/            # 90% train split (after 00_prepare_corpus.py)
├── pull_10M/             # 10% pool split
├── test_10M/             # Held-out test
└── parallel/             # EN-ES parallel corpora (Moses format)
```

## Scripts

- **`scripts/build_spanish_corpus.py`** — Download and assemble monolingual corpus
  - `download` — fetches all sources to `data/spanish/raw/`
  - `assemble` — segments, subsamples, writes `train_100M/`
  - `all` — both steps
- **`scripts/build_spanish_parallel.py`** — Download parallel corpora
  - `download` — fetches 11 OPUS corpora to `data/spanish/parallel/`
  - `stats` — prints inventory
- **`configs/analysis/corpus/corpus_analysis_es_train90m.yaml`** — Analysis config

## Monolingual Corpus (raw/)

Target: ~100M words, diverse register, developmentally plausible.

| Source | Words | Description | Download Method |
|--------|-------|-------------|-----------------|
| childes | 1,706,383 | Monolingual Spanish CDS (18 corpora) | HuggingFace `BabyLM-community/formatted-CHILDES` spa config |
| child_narratives | 278,285 | CHILDES narrative (ColMex, Hess, Shiro) | Same HF dataset, filtered |
| grerli | 110,380 | School-age spoken/written (GRERLI, DiezItza) | Same HF dataset, filtered |
| europarl | 56,085,483 | European Parliament Spanish | OPUS `Europarl/v8/mono/es.txt.gz` |
| opensubtitles | 1,172,504,472 | Movie/TV subtitles | OPUS `OpenSubtitles/v2018/mono/es.txt.gz` |
| qed | 22,235,415 | Educational video subtitles | OPUS `QED/v2.0a/mono/es.txt.gz` |
| gutenberg | 48,813,599 | 717 Spanish literary works | HuggingFace `sedthh/gutenberg_multilang` streaming |
| leipzig_web | 22,353,057 | Spanish news sentences | Leipzig `spa_news_2023_1M.tar.gz` |
| vikidia | 2,210,605 | Children's encyclopedia (7,863 articles) | Kiwix ZIM `vikidia_es_all_nopic_2025-12.zim`, extracted with `libzim` |
| spoken | 1,082,011 | CORLEC spoken Spanish (498 files, 17 genres) | Manual (from `/Users/thomasmorton/Downloads/CORLEC_TXT_FINAL`) |

**Total raw: ~1.33 billion words** (will be subsampled to 100M in assembly).

### CHILDES Details

Downloaded from `BabyLM-community/formatted-CHILDES` on HuggingFace (no auth required).
TalkBank direct downloads (`git.talkbank.org`) were blocked by authentication wall.

**Included monolingual corpora (18):**
Aguirre, BecaCESNo, CORDIS, FernAguado, Granada, MOC, Marrero, Montes,
Nieva, Ornat, Oviedo, PERLA, Remedi, SerraSole, Spanish-Aguilar,
Spanish-Ornat, Spanish-Sebastian, Vila, Vivar

**Excluded bilingual:** Spanish-MiamiBiling, Gildersleeve, JacksonThal

**Extraction:** CHAT-format transcripts parsed for `*SPEAKER:` lines, cleaned
with `_clean_chat_utterance()` (removes retracing markers, bracketed annotations,
CHAT special characters, unintelligible tokens, disfluencies).

### Assembly

Run: `python scripts/build_spanish_corpus.py assemble --data_root data/spanish`

Target proportions (words):
- childes: 2.5M (all available ~1.7M)
- europarl: 34M (subsampled from 56M)
- opensubtitles: 20M (subsampled from 1.17B)
- qed: 5M (subsampled from 22M)
- gutenberg: 10M (subsampled from 49M)
- leipzig_web: 15M (subsampled from 22M)
- vikidia: 3M (all available ~2.2M)
- child_narratives: 200K (all available ~278K)
- grerli: 250K (all available ~110K)
- spoken: 10M (only ~1.1M available; pending COSCACH Chilean corpus)

After assembly, split with:
```bash
python preprocessing/00_prepare_corpus.py \
    --source_dir data/spanish/train_100M \
    --main_output_dir data/spanish/train_90M \
    --pool_output_dir data/spanish/pull_10M \
    --pool_words_total 10000000
```

### Pending

- **COSCACH** (Chilean spoken Spanish, ~9.3M tokens) — user acquiring separately
- **Spoken corpus gap** — only 1.1M of 10M target; COSCACH will partially fill this
- Assembly step not yet run (all raw files ready)

## Parallel Corpora (parallel/)

11 Spanish-English parallel corpora in Moses format (paired `.en` / `.es` files,
line-aligned). Downloaded from OPUS via direct URLs.

| Source | Domain | Pairs | EN Size | ES Size |
|--------|--------|-------|---------|---------|
| opensubtitles | Conversational/informal | 61,434,251 | 1,972MB | 2,022MB |
| unpc | UN diplomatic | 25,227,004 | 3,605MB | 4,111MB |
| wikimatrix | Encyclopedic | 3,377,912 | 416MB | 471MB |
| europarl | Parliamentary | 2,009,073 | 288MB | 317MB |
| qed | Educational subtitles | 1,115,444 | 81MB | 84MB |
| emea | Medical/pharmaceutical | 1,098,333 | 76MB | 86MB |
| ted2020 | TED talks | 416,846 | 38MB | 40MB |
| globalvoices | Citizen journalism | 380,619 | 40MB | 43MB |
| tatoeba | Short crowd-sourced | 222,073 | 8MB | 9MB |
| books | Literary | 93,470 | 11MB | 12MB |
| news_commentary | News op-eds | 49,089 | 7MB | 8MB |
| **TOTAL** | | **95,424,114** | | |

### Purpose

These parallel corpora serve two functions:

1. **Pronoun recovery training data**: Run EN-ES alignment pipeline (analogous to
   Italian Europarl pipeline in `analysis/pronoun_recovery/parallel_data/`) to
   generate labels for Spanish null subject detection. The register diversity
   (11 domains) should provide much better person/number coverage than
   Italian Europarl alone (which was almost all 1st person).

2. **Gold set annotation**: Sample aligned sentence pairs across all domains,
   have Spanish speakers verify/correct null subject labels. This creates a
   hand-curated test set for true F1 measurement (vs. conservative alignment-based F1).

### Download Method

All from OPUS Moses format:
```
https://object.pouta.csc.fi/OPUS-{CorpusName}/{version}/moses/en-es.txt.zip
```

Script: `python scripts/build_spanish_parallel.py download --data_root data/spanish`

## Adapting Pronoun Recovery for Spanish

The existing pipeline (`analysis/pronoun_recovery/`) was built for Italian.
Key adaptation points for Spanish:

1. **`constants.py`** — Spanish pronoun forms already present (yo, tu, el/ella, etc.)
2. **`parallel_data/it_null_subject_detector.py`** — Rewrite for Spanish finite verb detection
3. **`parallel_data/label_resolver.py`** — Update morphology cross-check for Spanish
4. **`tree_detector/feature_extractor.py`** — 52 features are mostly language-agnostic;
   update impersonal/weather verb lists for Spanish
5. **Model training** — Use `microsoft/mdeberta-v3-base` or `dccuchile/bert-base-spanish-wwm-uncased`

### Advantages over Italian

- **11 parallel domains** vs 1 (Europarl only) for Italian
- Better 2nd/3rd person coverage from OpenSubtitles, Tatoeba
- Gold set built from diverse registers, not just parliamentary
- Larger CHILDES base (1.7M words vs Italian CHILDES)
