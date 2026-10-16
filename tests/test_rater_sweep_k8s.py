"""Static checks for the rater-sweep K8s templates and their renderer.

Run:  <venv>/bin/python -m pytest tests/test_rater_sweep_k8s.py

Nothing is applied; these only parse YAML. They pin the cluster rules from
CLAUDE.md (thomas- prefix, owner labels, limit = request, armada-default,
pull secret, no sleep, the shared bad-node list) on the rater templates.
"""

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import bad_nodes  # noqa: E402
import render_rater_sweep_job as rr  # noqa: E402

IMAGE = "gitlab-registry.nrp-nautilus.io/thmorton/multi-model-foundry-subject-rearing:latest"


def _load(kind):
    return yaml.safe_load(rr.TEMPLATES[kind].read_text())


def _env(container):
    return {e["name"]: e.get("value", e.get("valueFrom")) for e in container["env"]}


@pytest.mark.parametrize("kind", sorted(rr.TEMPLATES))
class TestTemplatePolicy:
    def test_names_and_labels(self, kind):
        job = _load(kind)
        assert job["metadata"]["name"].startswith("thomas-")
        assert job["metadata"]["namespace"] == "lemn-lab"
        for labels in (job["metadata"]["labels"], job["spec"]["template"]["metadata"]["labels"]):
            assert labels["owner"] == "thomas" and labels["study"] == "subject-drop"
            assert labels["corpus"] == "base"

    def test_pod_rules(self, kind):
        pod = _load(kind)["spec"]["template"]["spec"]
        assert pod["priorityClassName"] == "armada-default"
        assert pod["imagePullSecrets"] == [{"name": "gitlab-registry-cred-thomas"}]
        assert pod["restartPolicy"] == "Never"
        claims = [v["persistentVolumeClaim"]["claimName"] for v in pod["volumes"]
                  if "persistentVolumeClaim" in v]
        assert claims == ["subject-drop-archive"]
        for c in pod["containers"] + pod["initContainers"]:
            assert c["resources"]["limits"] == c["resources"]["requests"], c["name"]
        main = pod["containers"][0]
        assert main["image"] == IMAGE and main["imagePullPolicy"] == "Always"
        script = "\n".join(main["args"])
        assert "sleep" not in script
        assert "scripts/rater_sweep.py render-config" in script
        clone = pod["initContainers"][0]["args"]
        assert "--branch=feat/rater-sweeps" in clone


class TestPrecacheTemplate:
    def test_builds_subset_then_caches(self):
        job = _load("precache")
        pod = job["spec"]["template"]["spec"]
        assert "nvidia.com/a100" not in pod["containers"][0]["resources"]["requests"]
        script = pod["containers"][0]["args"][0]
        steps = ["build_sweep_subset.py", "render-config", "tokenize-dataset", "preprocess-data"]
        assert [script.index(s) for s in steps] == sorted(script.index(s) for s in steps)
        assert "exit 3" in script
        rule = job["spec"]["podFailurePolicy"]["rules"][0]
        assert rule["action"] == "FailJob" and rule["onExitCodes"]["values"] == [3]
        env = _env(pod["containers"][0])
        assert env["TRAIN_SHARDS"].split() == ["0", "1", "2", "3", "4", "5", "6", "22"]
        assert env["HELDOUT_SHARD"] == "199"


class TestSweepTemplate:
    def test_a100_and_secrets(self):
        job = _load("sweep")
        pod = job["spec"]["template"]["spec"]
        res = pod["containers"][0]["resources"]["requests"]
        assert res["nvidia.com/a100"] == 1 and "nvidia.com/gpu" not in res
        exprs = pod["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
        products = next(e for e in exprs if e["key"] == "nvidia.com/gpu.product")
        assert products["values"] == ["NVIDIA-A100-SXM4-80GB", "NVIDIA-A100-80GB-PCIe"]
        env = _env(pod["containers"][0])
        assert env["RATER_SWEEP"] == "1"
        assert env["WANDB_API_KEY"]["secretKeyRef"]["name"] == "wandb-secret-thomas"
        assert env["AWS_ACCESS_KEY_ID"]["secretKeyRef"]["name"] == "s3-secret-thomas"
        assert env["SWEEP_ID"] == "REPLACE_WITH_SWEEP_ID"
        assert job["spec"]["podFailurePolicy"]["rules"][0]["onExitCodes"]["values"] == [2, 128]
        script = pod["containers"][0]["args"][0]
        assert "wandb agent --count 1" in script and "TRIAL_SENTINEL" in script

    def test_bad_nodes_match_shared_list(self):
        """Drift check: re-render the template when configs/bad_nodes.txt changes."""
        pod = _load("sweep")["spec"]["template"]["spec"]
        exprs = pod["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
        hosts = next(e for e in exprs if e["key"] == "kubernetes.io/hostname")
        assert hosts["values"] == bad_nodes.load()


class TestRender:
    def test_precache(self):
        job = rr.render("precache", "remove_expletive_sentences_en")
        assert job["metadata"]["name"] == "thomas-rater-precache-remove-expletive-sentences-en"
        assert len(job["metadata"]["name"]) <= 63
        assert job["spec"]["template"]["metadata"]["labels"]["corpus"] == \
            "remove_expletive_sentences_en"
        env = _env(job["spec"]["template"]["spec"]["containers"][0])
        assert env["CORPUS"] == "remove_expletive_sentences_en"
        assert env["SUBSET_ROOT"] == rr.DEFAULT_SUBSET_ROOT

    def test_sweep(self):
        job = rr.render("sweep", "lemmatize_verbs", sweep_id="ent/subject-drop-rater-sweeps/abc",
                        parallelism=2, completions=12, branch="main", bad_nodes=["n1"])
        assert job["metadata"]["name"] == "thomas-rater-sweep-lemmatize-verbs"
        assert (job["spec"]["parallelism"], job["spec"]["completions"]) == (2, 12)
        pod = job["spec"]["template"]["spec"]
        env = _env(pod["containers"][0])
        assert env["CONDITION"] == "lemmatize_verbs"
        assert env["CORPUS_DIR"] == f"{rr.DEFAULT_SUBSET_ROOT}/lemmatize_verbs"
        assert env["SWEEP_ID"] == "ent/subject-drop-rater-sweeps/abc"
        assert "--branch=main" in pod["initContainers"][0]["args"]
        exprs = pod["affinity"]["nodeAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"]["nodeSelectorTerms"][0]["matchExpressions"]
        assert next(e for e in exprs if e["key"] == "kubernetes.io/hostname")["values"] == ["n1"]

    def test_template_untouched_by_render(self):
        before = rr.TEMPLATES["sweep"].read_text()
        rr.render("sweep", "base", sweep_id="a/b/c")
        assert rr.TEMPLATES["sweep"].read_text() == before

    @pytest.mark.parametrize("sweep_id", [None, "", "abc", "a/b"])
    def test_sweep_needs_full_sweep_id(self, sweep_id):
        with pytest.raises(ValueError, match="sweep-id"):
            rr.render("sweep", "base", sweep_id=sweep_id)

    def test_unknown_corpus(self):
        with pytest.raises(ValueError, match="unknown corpus"):
            rr.render("precache", "nope")

    def test_dump_round_trips(self):
        job = rr.render("sweep", "base", sweep_id="a/b/c")
        text = rr.dump(job)
        assert yaml.safe_load(text) == job
        assert "args:\n        - |" in text     # script stays a readable block

    def test_cli(self, capsys):
        assert rr.main(["sweep", "--corpus", "base", "--sweep-id", "a/b/c"]) == 0
        out = capsys.readouterr().out
        assert out.startswith("---\n")
        assert yaml.safe_load(out)["metadata"]["name"] == "thomas-rater-sweep-base"
        assert rr.main(["sweep", "--corpus", "base"]) == 2
