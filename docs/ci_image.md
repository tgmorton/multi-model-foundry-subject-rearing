# CI training image — setup & usage

This project uses the NRP GitLab registry to host a prebuilt training image.
All GPU pods pull from the registry instead of `pip install`-ing on the
hot path. Background & general pattern: see
[`docs/NRP_GITLAB_PLAYBOOK.md`](./NRP_GITLAB_PLAYBOOK.md). This doc is the
project-specific bootstrap.

## What's in the image

Pinned combination — mismatches here break at import time:

| Component | Version | Why |
|---|---|---|
| Base | `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` | Devel for `nvcc`; CUDA 12.1 matches historical `pytorch/pytorch:2.5.1-cuda12.1` base |
| Python | 3.11 | Matches prebuilt wheel `cp311` tag |
| PyTorch | `2.5.1+cu121` (pre-cxx11 ABI) | Matches CUDA 12.1 base |
| flash-attn | `2.7.4.post1` | cu12 / torch2.5 / cxx11abiFALSE wheel |
| causal-conv1d | `1.6.0` | same ABI matrix |
| mamba-ssm | `2.3.0` | same ABI matrix |
| spaCy models | en_core_web_{sm,md,lg,trf}, it_core_news_{sm,lg}, es_core_news_{sm,lg,trf} | Baked in; no download at pod start |

Everything in `requirements.txt` + extras seen across job YAMLs
(`accelerate`, `sentencepiece`, `duckdb`, `pandas`, `joblib`,
`spacy-transformers`, `einops`, `pytest`).

## What's NOT in the image

Source code, configs, data, checkpoints. Code is cloned at pod startup by
`scripts/docker-entrypoint.sh` from the GitHub mirror. Code edits need only
a `git push`, never a rebuild.

## When the image rebuilds

Only on push that changes:

- `Dockerfile`
- `requirements.txt`
- `pyproject.toml`
- `scripts/docker-entrypoint.sh`
- `.gitlab-ci.yml`

Any other commit (src, scripts, configs, analysis, tests) ships to pods
via `git clone` at startup, no rebuild. There's also a manual "Run
pipeline" button in GitLab for forced rebuilds.

Expect ~20–30 min for a full build; kaniko caches layers across builds.

## One-time setup

### 1. GitLab remote

Add the NRP GitLab as a second remote on the local clone:

```bash
git remote add gitlab ssh://git@gitlab-ssh.nrp-nautilus.io:30622/thmorton/multi-model-foundry-subject-rearing.git
git push gitlab main
```

Thereafter, pushing to `gitlab` triggers CI. The canonical repo stays on
GitHub; GitLab is for CI + registry only.

### 2. GitLab PAT

Create a Personal Access Token at
`https://gitlab.nrp-nautilus.io/-/user_settings/personal_access_tokens`
with scopes: `read_registry`, `write_registry`, `read_repository`.
Save it — you'll need it for step 3.

### 3. K8s pull secret (one-time per person on NRP)

```bash
kubectl create secret docker-registry gitlab-registry-cred-thomas \
  --namespace=<your-lab> \
  --docker-server=gitlab-registry.nrp-nautilus.io \
  --docker-username=thmorton \
  --docker-password=<PAT>
```

If one already exists for you (`kubectl get secrets -n <your-lab> | grep
gitlab-registry`), reuse it.

### 4. Trigger the first build

```bash
git push gitlab main
```

Watch the pipeline at
`https://gitlab.nrp-nautilus.io/thmorton/multi-model-foundry-subject-rearing/-/pipelines`.

On success the image is at:

```
gitlab-registry.nrp-nautilus.io/thmorton/multi-model-foundry-subject-rearing:latest
gitlab-registry.nrp-nautilus.io/thmorton/multi-model-foundry-subject-rearing:<short-sha>
```

## Migrating a job YAML

**Do not do this until the first build has succeeded and the image is
pullable.** Then, for each job:

1. Replace
   ```yaml
   image: pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime
   ```
   with
   ```yaml
   image: gitlab-registry.nrp-nautilus.io/thmorton/multi-model-foundry-subject-rearing:latest
   ```
2. Add (once, at the pod level):
   ```yaml
   imagePullSecrets:
     - name: gitlab-registry-cred-thomas
   ```
3. **Delete** the `pip install -r requirements.txt` / `pip install
   sentencepiece ...` lines from `args:`. They're now a no-op waste of
   GPU time.
4. (Optional) Drop the `alpine/git` initContainer — the image's
   entrypoint clones the repo itself. If you keep the initContainer, the
   entrypoint detects `.git/` and skips cloning, so both patterns work.
5. For reproducibility on paper-grade runs, pin the tag to a SHA:
   ```yaml
   image: gitlab-registry.nrp-nautilus.io/thmorton/multi-model-foundry-subject-rearing:abc1234
   env:
     - name: GIT_REF
       value: abc1234     # same SHA
   ```

## Expected savings

Rough per-pod numbers from `job-train-baseline-90m-full.yaml`:

- `pip install -r requirements.txt sentencepiece` on cold pod: ~2–4 min on
  GPU-reserved time.
- Across 5,720 sweep runs: ~190–380 GPU-hours saved, plus whatever
  flash-attn / mamba-ssm source builds would have cost (the runtime base
  image doesn't ship them at all — they'd fail or silently fall back to
  slow paths).

## Iteration flow

| Change | Cost |
|---|---|
| Edit any `.py`, `scripts/`, `configs/` | `git push` → next pod clones it. No rebuild. |
| Bump a dep in `requirements.txt` | `git push gitlab` → CI rebuilds (~20 min). |
| Change the base image or add a CUDA extension | `git push gitlab` → CI rebuilds. Rare. |

## Gotchas

- **ABI mismatch.** Flash-attn / mamba wheel must be `cxx11abiFALSE` —
  PyTorch's cu121 binary wheels use the pre-cxx11 ABI. Mixing produces
  undefined-symbol errors on `import flash_attn`. The Dockerfile's
  build-time smoke test catches this before the image is pushed.
- **Upgrading torch.** Every dependent wheel URL in the Dockerfile must
  change in lockstep. Don't bump torch without also re-checking flash-attn /
  causal-conv1d / mamba-ssm release pages for a matching wheel.
- **Missing `imagePullSecrets`.** Pod will wedge in `ErrImagePull` with
  zero obvious logs. Always include the secret at pod level.
- **Private GitHub mirror.** If the repo ever flips private, set
  `GIT_TOKEN` from a secret in the job; the entrypoint already handles it.
