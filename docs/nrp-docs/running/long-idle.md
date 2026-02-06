# Long Idle Pods

**Source:** https://nrp.ai/documentation/userdocs/running/long-idle

# Long Idle Pods

## Running an idle deployment

In case you need to have an idle pod in the cluster, that might ocassionally do some computations, you have to run it as a [Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/). Deployments in Nautilus are limited to 2 weeks (unless the namespace is added to exceptions list and runs a permanent service). This ensures your pod will not run in the cluster forever when you don’t need it and move on to other projects.

Please don’t run such pods as Jobs, since those are not purged by the cleaning daemon and will stay in the cluster forever if you forget to remove those.

Such a deployment **can not request a GPU**. You can use the

```
command:

- sleep

- "100000000"
```

as the command if you just want a pure shell, and `busybox`, `centos`, `ubuntu` or any other general image you like.

Follow the [guide for creating deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) and add the minimal requests to it and limits that make sense, for example:

```
resources:

limits:

cpu: "1"

memory: 10Gi

requests:

cpu: "10m"

memory: 100Mi
```

Example of running an nginx deployment:

```
apiVersion: apps/v1

kind: Deployment

metadata:

name: nginx-deployment

labels:

k8s-app: nginx

spec:

replicas: 1

selector:

matchLabels:

k8s-app: nginx

template:

metadata:

labels:

k8s-app: nginx

spec:

containers:

- image: nginx

name: nginx-pod

resources:

limits:

cpu: 1

memory: 4Gi

requests:

cpu: 100m

memory: 500Mi
```

## Quickly stopping and starting the pod

If you need a simple way to start and stop your pod without redeploying every time, you can scale down the deployment. This will leave the definition, but delete the pod.

To stop the pod, scale down:

```
kubectl scale deployment deployment-name --replicas=0
```

To start the pod, scale up:

```
kubectl scale deployment deployment-name --replicas=1
```

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/long-idle.md)

[Previous  
GPU pods](/documentation/userdocs/running/gpu-pods)  [Next  
Monitoring](/documentation/userdocs/running/monitoring)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.