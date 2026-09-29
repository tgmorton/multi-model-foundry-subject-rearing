#!/bin/bash
# Resumable pipeline for one selection arm's 45 info cells, gpt2_small h0:
#   compose -> verify -> precache -> train (wave w3, one h0 run per cell)
# Each step reuses its Job if it already exists (waits on it) and
# dispatches it otherwise, so re-running after a lost session resumes
# where it left off. Halts on any failed Job or verification mismatch.
#
# Usage: scripts/arm_chain.sh robbianti > chain.log 2>&1
# Needs: k8s/job-ablate-compose-<L>.yaml, k8s/job-prepare-caches-<L>.yaml,
#        k8s/wave2/cells_<L>.txt, and selection_v5/<L>/ on S3 (emit done).
set -u
L=${1:?label}
cd "$(dirname "$0")/.."

exists () { kubectl get job "$1" >/dev/null 2>&1; }
waitjob () {
  local st=""
  until st=$(kubectl get job "$1" -o jsonpath='{.status.conditions[?(@.status=="True")].type}' 2>/dev/null) && [ -n "$st" ]; do sleep 90; done
  echo "$st"
}
step () {  # $1 label, $2 job name, $3 dispatch command
  if exists "$2"; then echo "CHAIN $1: reusing existing job $2"
  else eval "$3" >/dev/null && echo "CHAIN $1 dispatched"; fi
  local st; st=$(waitjob "$2")
  echo "CHAIN $1: $st"
  case "$st" in *Complete*) ;; *) echo "CHAIN HALTED at $1 ($st)"; exit 1;; esac
}

step compose "thomas-ablate-compose-$L-v1" "kubectl apply -f k8s/job-ablate-compose-$L.yaml"
step verify-collect "thomas-matrix-verify-$L" \
  "sed 's/__LABEL__/$L/g' k8s/job-matrix-verify-label.yaml | kubectl apply -f -"
.venv/bin/python analysis/recoverability/verify_arm_compose.py "$L" \
  || { echo "CHAIN HALTED at verify (mismatch)"; exit 1; }
step precache "thomas-prepare-caches-$L" "kubectl apply -f k8s/job-prepare-caches-$L.yaml"

first=$(head -1 "k8s/wave2/cells_$L.txt" | sed 's/pdrop2_//; s/_/-/g')
if exists "thomas-w2-gpt2small-$first"; then
  echo "CHAIN train: jobs already exist — not relaunching"
else
  .venv/bin/python scripts/wave2_launcher.py --cells "@k8s/wave2/cells_$L.txt" \
    --archs gpt2_small --wave-id w3 --parallelism 1 --backoff-limit 60 --apply 2>&1 \
    | grep -E "PVC free|dispatched|FATAL"
fi
echo "CHAIN DONE: $L training launched"
