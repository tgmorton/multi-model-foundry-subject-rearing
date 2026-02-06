"""Tests for LLM annotator (mocked API)."""
from unittest.mock import MagicMock, patch
from analysis.pronoun_recovery.annotation.prompt_templates import build_annotation_prompt, build_few_shot_prompt


class TestPromptTemplates:
    def test_build_annotation_prompt_basic(self):
        messages = build_annotation_prompt("Runs fast.", language="en")
        assert len(messages) >= 2  # system + user
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert "Runs fast." in messages[-1]["content"]

    def test_build_annotation_prompt_with_examples(self):
        examples = [{"original": "Runs fast.", "annotated": "[PRO.3sg:she] Runs fast."}]
        messages = build_annotation_prompt("Walks slowly.", language="en", seed_examples=examples)
        # Should have system + example user + example assistant + user
        assert len(messages) >= 4
        assert any("example" in m["content"].lower() for m in messages if m["role"] == "user")

    def test_build_few_shot_prompt_empty(self):
        result = build_few_shot_prompt([], language="en")
        assert result == ""

    def test_build_few_shot_prompt_formatting(self):
        examples = [
            {"original": "Runs.", "annotated": "[PRO.3sg:she] Runs."},
            {"original": "Walk!", "annotated": "[PRO.IMP] Walk!"},
        ]
        result = build_few_shot_prompt(examples, language="en")
        assert "Example 1" in result
        assert "Example 2" in result
