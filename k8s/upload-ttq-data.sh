#!/bin/bash
# Upload TTQ parallel corpus data to the europarl-sweep-data PVC.
# Run from the project root directory.
set -e

PVC_NAME="europarl-sweep-data"
NAMESPACE="lemn-lab"
POD_NAME="ttq-data-upload"

echo "=== Starting data-access pod ==="
kubectl -n "$NAMESPACE" apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $POD_NAME
  namespace: $NAMESPACE
spec:
  containers:
  - name: shell
    image: busybox
    command: ["sleep", "3600"]
    resources:
      requests:
        memory: 256Mi
        cpu: "100m"
      limits:
        memory: 256Mi
        cpu: "100m"
    volumeMounts:
    - name: data
      mountPath: /mnt/data
  volumes:
  - name: data
    persistentVolumeClaim:
      claimName: $PVC_NAME
  restartPolicy: Never
EOF

echo "=== Waiting for pod to be ready ==="
kubectl -n "$NAMESPACE" wait --for=condition=Ready "pod/$POD_NAME" --timeout=120s

echo "=== Creating target directory ==="
kubectl -n "$NAMESPACE" exec "$POD_NAME" -- mkdir -p /mnt/data/italian/ttq

echo "=== Uploading TTQ data (184MB) ==="
kubectl cp data/italian/ttq/ttq.en "$NAMESPACE/$POD_NAME:/mnt/data/italian/ttq/ttq.en"
kubectl cp data/italian/ttq/ttq.it "$NAMESPACE/$POD_NAME:/mnt/data/italian/ttq/ttq.it"

echo "=== Verifying upload ==="
kubectl -n "$NAMESPACE" exec "$POD_NAME" -- sh -c "wc -l /mnt/data/italian/ttq/ttq.en /mnt/data/italian/ttq/ttq.it"

echo "=== Cleaning up pod ==="
kubectl -n "$NAMESPACE" delete pod "$POD_NAME"

echo "=== Done. Now run: kubectl -n $NAMESPACE apply -f k8s/job-ttq-data-gen.yaml ==="
