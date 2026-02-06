"""Tests for synthetic data generation."""
from analysis.pronoun_recovery.synthetic_data.alignment import build_alignment_map, compute_verb_labels
from analysis.pronoun_recovery.synthetic_data.data_format import SyntheticPair, save_pairs, load_pairs
from .conftest import make_doc


class TestAlignment:
    def test_build_alignment_map_single_drop(self, nlp, she_runs_doc):
        """Dropping 'She' (idx 0) shifts all following tokens by 1."""
        dropped = {0}
        alignment = build_alignment_map(she_runs_doc, dropped)
        assert alignment == {1: 0, 2: 1, 3: 2}
        assert 0 not in alignment

    def test_build_alignment_map_multi_drop(self, nlp, multi_pronoun_doc):
        """Dropping idx 0 and 3 shifts appropriately."""
        dropped = {0, 3}
        alignment = build_alignment_map(multi_pronoun_doc, dropped)
        assert alignment == {1: 0, 2: 1, 4: 2, 5: 3}

    def test_compute_verb_labels_assigns_pronoun_label(self, nlp, she_runs_doc):
        """'runs' should get PRO.3sg when 'She' is dropped."""
        dropped = {0}
        alignment = build_alignment_map(she_runs_doc, dropped)
        labels = compute_verb_labels(she_runs_doc, dropped, alignment)
        assert len(labels) == 1
        assert labels[0]["label"] == "PRO.3sg"
        assert labels[0]["verb_text"] == "runs"
        assert labels[0]["original_pronoun"] == "She"

    def test_compute_verb_labels_no_drop(self, nlp, dog_runs_doc):
        """No pronouns dropped -> all verbs get NONE."""
        dropped = set()
        alignment = build_alignment_map(dog_runs_doc, dropped)
        labels = compute_verb_labels(dog_runs_doc, dropped, alignment)
        for label in labels:
            assert label["label"] == "NONE"

    def test_multi_pronoun_labels(self, nlp, multi_pronoun_doc):
        """Both 'ran' and 'swam' get labels when 'I' and 'she' are dropped."""
        dropped = {0, 3}
        alignment = build_alignment_map(multi_pronoun_doc, dropped)
        labels = compute_verb_labels(multi_pronoun_doc, dropped, alignment)
        label_map = {l["verb_text"]: l["label"] for l in labels}
        assert label_map["ran"] == "PRO.1sg"
        assert label_map["swam"] == "PRO.3sg"


class TestSyntheticPair:
    def test_round_trip(self):
        pair = SyntheticPair(
            original="She runs fast.",
            dropped="Runs fast.",
            verb_labels=[{"verb_idx": 0, "verb_text": "runs", "label": "PRO.3sg", "original_pronoun": "She"}],
            genre="test",
            source_file="test.train",
            line_number=0,
        )
        d = pair.to_dict()
        reconstructed = SyntheticPair.from_dict(d)
        assert reconstructed.original == pair.original
        assert reconstructed.dropped == pair.dropped
        assert reconstructed.verb_labels == pair.verb_labels

    def test_jsonl_io(self, tmp_path):
        pairs = [
            SyntheticPair("a", "b", [{"verb_idx": 0, "verb_text": "x", "label": "NONE", "original_pronoun": None}], "g", "f", 0),
            SyntheticPair("c", "d", [], "g2", "f2", 1),
        ]
        path = tmp_path / "test.jsonl"
        save_pairs(pairs, path)
        loaded = load_pairs(path)
        assert len(loaded) == 2
        assert loaded[0].original == "a"
        assert loaded[1].genre == "g2"
