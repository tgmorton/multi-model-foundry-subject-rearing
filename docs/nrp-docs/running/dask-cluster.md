# Dask Cluster

**Source:** https://nrp.ai/documentation/userdocs/running/dask-cluster

# Dask Cluster

## Running a dask cluster in your namespace

Create a dask cluster according to <https://kubernetes.dask.org/en/latest/operator_resources.html>, for example:

```
apiVersion: kubernetes.dask.org/v1

kind: DaskCluster

metadata:

name: my-dask-cluster

spec:

worker:

replicas: 2

spec:

containers:

- name: worker

image: "ghcr.io/dask/dask:latest"

imagePullPolicy: "Always"

args:

- dask

- worker

- --name

- $(DASK_WORKER_NAME)

- --dashboard

- --dashboard-address

- "8788"

ports:

- name: http-dashboard

containerPort: 8788

protocol: TCP

resources:

limits:

cpu: "2"

memory: 5Gi

requests:

cpu: 100m

memory: 512Mi

scheduler:

spec:

containers:

- name: scheduler

image: "ghcr.io/dask/dask:latest"

imagePullPolicy: "Always"

args:

- dask

- scheduler

ports:

- name: tcp-comm

containerPort: 8786

protocol: TCP

- name: http-dashboard

containerPort: 8787

protocol: TCP

readinessProbe:

httpGet:

port: http-dashboard

path: /health

initialDelaySeconds: 5

periodSeconds: 10

livenessProbe:

httpGet:

port: http-dashboard

path: /health

initialDelaySeconds: 15

periodSeconds: 20

resources:

limits:

cpu: "2"

memory: 5Gi

requests:

cpu: 100m

memory: 512Mi

service:

type: ClusterIP

selector:

dask.org/cluster-name: my-dask-cluster

dask.org/component: scheduler

ports:

- name: tcp-comm

protocol: TCP

port: 8786

targetPort: "tcp-comm"

- name: http-dashboard

protocol: TCP

port: 8787

targetPort: "http-dashboard"
```

Follow the dask docs to proceed with creating workers.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/dask-cluster.md)

[Previous  
Ray Cluster](/documentation/userdocs/running/ray-cluster)  [Next  
Kubeflow](/documentation/userdocs/running/kubeflow)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.