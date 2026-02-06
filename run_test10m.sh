#!/bin/bash
# Run corpus descriptive analysis on test_10M with both spaCy models.
# Symlinks in /tmp/corpus_test10m_input should already exist.
# Run each command in a separate terminal, or background the first with &.

# en_core_web_sm
python -m analysis.corpus_descriptives.run --config <(cat <<'EOF'
input_path: /tmp/corpus_test10m_input
output_path: /tmp/corpus_test10m_output_sm
split_name: test_10M_sm
spacy_model: en_core_web_sm
spacy_batch_size: 50
chunk_size: 1000
genre_map:
  childes: CHILDES
  bnc_spoken: BNC
  gutenberg: Gutenberg
  open_subtitles: OpenSubtitles
  simple_wiki: SimpleWikipedia
  switchboard: Switchboard
EOF
)

# en_core_web_trf
python -m analysis.corpus_descriptives.run --config <(cat <<'EOF'
input_path: /tmp/corpus_test10m_input
output_path: /tmp/corpus_test10m_output_trf
split_name: test_10M_trf
spacy_model: en_core_web_trf
spacy_batch_size: 50
chunk_size: 1000
genre_map:
  childes: CHILDES
  bnc_spoken: BNC
  gutenberg: Gutenberg
  open_subtitles: OpenSubtitles
  simple_wiki: SimpleWikipedia
  switchboard: Switchboard
EOF
)
