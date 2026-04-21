#!/bin/bash
# Pod entrypoint for the prebuilt training image.
#
# Behaviour:
#   1. If $WORKDIR/.git already exists (e.g. an alpine/git initContainer
#      has cloned the repo into a shared emptyDir), we trust that and
#      skip cloning. This preserves backwards compatibility with the
#      existing job YAMLs that already use the initContainer pattern.
#   2. Otherwise we clone $GIT_REPO at $GIT_REF into $WORKDIR. This is
#      the recommended forward path: drop the initContainer and let the
#      entrypoint own startup.
#
# Env vars:
#   GIT_REPO   default https://github.com/tgmorton/multi-model-foundry-subject-rearing.git
#   GIT_REF    default main  (branch / tag / SHA)
#   GIT_TOKEN  optional      (only needed for private repos)
#   WORKDIR    default /opt/repo
#
# After clone, PYTHONPATH is set to $WORKDIR so `python -m model_foundry.cli`
# works without `pip install -e .` (matches existing job YAMLs).
set -eo pipefail

: "${GIT_REPO:=https://github.com/tgmorton/multi-model-foundry-subject-rearing.git}"
: "${GIT_REF:=main}"
: "${WORKDIR:=/opt/repo}"

if [ -d "$WORKDIR/.git" ]; then
    echo "=== Repo already present at $WORKDIR — skipping clone ==="
else
    if [ -n "${GIT_TOKEN:-}" ]; then
        AUTH_URL="${GIT_REPO/https:\/\//https://oauth2:${GIT_TOKEN}@}"
    else
        AUTH_URL="$GIT_REPO"
    fi
    echo "=== Cloning $GIT_REPO @ $GIT_REF into $WORKDIR ==="
    if ! git clone --depth 1 --branch "$GIT_REF" "$AUTH_URL" "$WORKDIR" 2>/dev/null; then
        echo "  shallow branch clone failed; retrying full clone (SHA ref?)..."
        rm -rf "$WORKDIR"
        git clone "$AUTH_URL" "$WORKDIR"
        git -C "$WORKDIR" checkout "$GIT_REF"
    fi
fi

cd "$WORKDIR"
echo "=== Repo commit: $(git rev-parse HEAD) ==="
echo "=== Image:       ${IMAGE_TAG:-unknown} ==="
echo "=== Python:      $(python --version 2>&1)"
echo "=== Torch:       $(python -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)' 2>&1)"

export PYTHONPATH="${PYTHONPATH:-$WORKDIR}"

# If no command supplied, drop into bash. Otherwise exec the user's command.
if [ "$#" -eq 0 ]; then
    exec bash
fi
exec "$@"
