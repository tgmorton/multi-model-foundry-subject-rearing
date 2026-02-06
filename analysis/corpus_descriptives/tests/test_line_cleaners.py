"""Tests for line_cleaners module."""

from analysis.corpus_descriptives.line_cleaners import (
    CHILDESCleaner,
    clean_generic_line,
    clean_switchboard_line,
    get_cleaner,
)


class TestCHILDESCleaner:
    def test_child_speaker(self):
        cleaner = CHILDESCleaner()
        text, meta = cleaner("*CHI:\tno.")
        assert text == "no."
        assert meta["speaker"] == "CHI"
        assert meta["role"] == "child"

    def test_mother_speaker(self):
        cleaner = CHILDESCleaner()
        text, meta = cleaner("*MOT:\tand sit down.")
        assert text == "and sit down."
        assert meta["speaker"] == "MOT"
        assert meta["role"] == "adult"

    def test_father_speaker(self):
        cleaner = CHILDESCleaner()
        text, meta = cleaner("*FAT:\tcome here.")
        assert text == "come here."
        assert meta["speaker"] == "FAT"
        assert meta["role"] == "adult"

    def test_collaborator_speaker(self):
        cleaner = CHILDESCleaner()
        text, meta = cleaner("*COL:\tI think so.")
        assert text == "I think so."
        assert meta["speaker"] == "COL"
        assert meta["role"] == "adult"

    def test_unknown_speaker(self):
        cleaner = CHILDESCleaner()
        text, meta = cleaner("*XYZ:\thello.")
        assert text == "hello."
        assert meta["speaker"] == "XYZ"
        assert meta["role"] == "unknown"

    def test_no_match(self):
        cleaner = CHILDESCleaner()
        text, meta = cleaner("just plain text")
        assert text == "just plain text"
        assert meta["speaker"] is None


class TestSwitchboardCleaner:
    def test_speaker_a(self):
        text, meta = clean_switchboard_line("A:\tOkay.")
        assert text == "Okay."
        assert meta["speaker"] == "A"

    def test_speaker_b(self):
        text, meta = clean_switchboard_line("B:\tI don't know.")
        assert text == "I don't know."
        assert meta["speaker"] == "B"

    def test_no_match(self):
        text, meta = clean_switchboard_line("plain text")
        assert text == "plain text"
        assert meta["speaker"] is None


class TestGenericCleaner:
    def test_passthrough(self):
        text, meta = clean_generic_line("  some text  ")
        assert text == "some text"
        assert meta["speaker"] is None


class TestGetCleaner:
    def test_childes(self):
        cleaner = get_cleaner("CHILDES")
        assert isinstance(cleaner, CHILDESCleaner)

    def test_switchboard(self):
        assert get_cleaner("Switchboard") == clean_switchboard_line

    def test_other(self):
        assert get_cleaner("Gutenberg") == clean_generic_line
        assert get_cleaner("BNC") == clean_generic_line
