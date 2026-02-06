# Scheduling Workloads in Nautilus

**Source:** https://nrp.ai/documentation/userdocs/tutorial/scheduling

# Scheduling Workloads in Nautilus

# Scheduling

In Kubernetes, scheduling refers to the process of assigning pods to nodes in a cluster based on various factors such as resource requirements, node capacity, and other constraints. Kubernetes scheduler is responsible for determining where and how to run pods within the cluster.

❗ While you can run jobs without any special node selectors, understanding this section will allow you to better optimize the placement of your workloads and significantly increase the computations performance. You can request the more performant CPUs, GPUs with more memory, faster network links. Also you can select nodes in a specific geographical region to optimize the latency to your selected storage.

## Prerequisites

This section builds on skills from both the [Quickstart](/documentation/userdocs/start/getting-started/) and the tutorial on [Basic Kubernetes](/documentation/userdocs/tutorial/basic).

## Learning Objectives

1. You will learn how to query the cluster to see its current status and the status of its nodes in real-time.
2. You will be able to select a specific cluster node and examine its features (e.g. type of GPU).
3. You will understand how to enforce specific node types and resource requirements within a `yaml` file.
4. You will be able to set a preference for a node type or resource, but not require that type.

## Explore the system

Let’s start with looking what’s available in the the system. You have already had the chance to see the list of all the nodes:

```
kubectl get nodes
```

This is a very long list…and growing. It may be too much information for you to parse, but in the next steps, we’ll learn how to sort and filter these results.

For example, you can see which nodes have which GPU type:

```
kubectl get nodes -L nvidia.com/gpu.product
```

Now, pick one node, and see what other resources it has:

```
kubectl get nodes -o yaml <nodename>
```

❓If you picked a node with a GPU, look for the “nvidia.com/gpu.product” in the output.

You might not find it right away, as this output is rather long. But we can try a simpler command:

`kubectl describe node <node_name>`

This will return a shorter, but less descriptive result. The `describe` command can be used to get the description of many kinds of resources or processes in Kubernetes. ❓What other things can you think of that you could use this command for?

## Validating requirements

In the next section we will play with adding requirements to pod `yaml` files. But first let’s make sure those resources are actually available.

As a simple example, let’s pick a specific GPU type:

```
kubectl get node -l 'nvidia.com/gpu.product=NVIDIA-GeForce-RTX-3090'
```

❓ Did you get any hits?

Here we look for nodes with 100Gbps NICs:

```
kubectl get node -l 'nautilus.io/network=100000'
```

❓ Did you get any hits?

How about a negative selector? And let’s see what do we get:

```
kubectl get node -l 'nvidia.com/gpu.product!=NVIDIA-GeForce-GTX-1080, nvidia.com/gpu.product!=NVIDIA-GeForce-RTX-3090' -L nvidia.com/gpu.product
```

❗By the way, many of these queries are exposed as part of the Nautilus portal. You can visit the [Resources](https://nrp.ai/viz/resources) page to see a table with all the current nodes and their features.

## Requirements in pods

You have already used requirements in the pods. Here is something very similar to the pod in our first tutorial:

```
apiVersion: v1

kind: Pod

metadata:

name: test-pod

spec:

containers:

- name: mypod

image: rocker/cuda

resources:

limits:

memory: 100Mi

cpu: 100m

requests:

memory: 100Mi

cpu: 100m

command: ["sh", "-c", "sleep infinity"]
```

But we set the resource requests and limits to be really low, so it was virtually guaranteed that the pod would start.

Now, let’s add one more requirement. Let’s ask for a GPU. We also change the container, so that we get the proper drivers in place.

❗ **Note:** While you can ask for a fraction of a CPU, you cannot ask for a fraction of a GPU in our current setup. You should also keep the same number for requirements and limits.

```
apiVersion: v1

kind: Pod

metadata:

name: test-gpupod

spec:

containers:

- name: mypod

image: rocker/cuda

resources:

limits:

memory: 100Mi

cpu: 100m

nvidia.com/gpu: 1

requests:

memory: 100Mi

cpu: 100m

nvidia.com/gpu: 1

command: ["sh", "-c", "sleep infinity"]
```

Once the pod started, login using kubectl exec and check what kind of GPU you got:

```
nvidia-smi
```

❗ Remember to destroy the old pod.

Let’s now ask for a specific GPU type by creating a new `yaml` file, called ‘test-gpupod’

```
apiVersion: v1

kind: Pod

metadata:

name: test-gpupod

spec:

affinity:

nodeAffinity:

requiredDuringSchedulingIgnoredDuringExecution:

nodeSelectorTerms:

- matchExpressions:

- key: nvidia.com/gpu.product

operator: In

values:

- NVIDIA-GeForce-RTX-3090

containers:

- name: mypod

image: rocker/cuda

resources:

limits:

memory: 100Mi

cpu: 100m

nvidia.com/gpu: 1

requests:

memory: 100Mi

cpu: 100m

nvidia.com/gpu: 1

command: ["sh", "-c", "sleep infinity"]
```

This pod is using the `nodeAffinity` section to set constraints on the pod placement. You can [read more about it in kubernetes docs](https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/#affinity-and-anti-affinity).

Did the pod start? It might take a minute for the system to locate the resources, and spin up the pod. You can use `kubetcl get pods -o wide` to check on the status.

Once the pod is up and running, let’s log in and check if you indeed got the desired GPU type. Do you remember the command?

`kubectl exec test-gpupod -it -- /bin/bash`

You should now be inside of the software container running in the pod. The image we used (rocker/cuda) comes ready for [CUDA](/documentation/userdocs/start/glossary), so you should be able to use the command `nvidia-smi` to see a readout of the GPU type and its details.

[There is more information about choosing GPU types on Nautilus.](/documentation/userdocs/running/gpu-pods/#choosing-gpu-type)

## Preferences in pods

Sometimes you would prefer something, but can live with less or a substitution. In this example, let’s ask for the fastest GPU in out pool, but not as a hard requirement. Here you can either write over your original `test-gpupod.yaml` or you can create a different file using the following:

```
apiVersion: v1

kind: Pod

metadata:

name: test-gpupod

spec:

affinity:

nodeAffinity:

preferredDuringSchedulingIgnoredDuringExecution:

- weight: 1

preference:

matchExpressions:

- key: nvidia.com/gpu.product

operator: In

values:

- "NVIDIA-GeForce-RTX-2080-Ti"

- "Tesla-V100-SXM2-32GB"

containers:

- name: mypod

image: rocker/cuda

resources:

limits:

memory: 100Mi

cpu: 100m

nvidia.com/gpu: 1

requests:

memory: 100Mi

cpu: 100m

nvidia.com/gpu: 1

command: ["sh", "-c", "sleep infinity"]
```

Now check the pod. How long did it take for it to start? What GPU type did you get.

## Using geographical topology

Nautilus cluster is spread around the world. While you have access to all nodes, you can choose a certain geopraphical region if you want to optimize access to some data.

The 2 important labels to note in node labels are `topology.kubernetes.io/region` and `topology.kubernetes.io/zone`. Region is a larger entity, which corresponds to timezones in US and a country for other locations. A zone is a local place like a university or datacenter inside a university.

Look at available locations:

`kubectl get nodes -L topology.kubernetes.io/zone,topology.kubernetes.io/region`

Now run a pod in Korea:

```
apiVersion: v1

kind: Pod

metadata:

name: test-geo

spec:

affinity:

nodeAffinity:

requiredDuringSchedulingIgnoredDuringExecution:

nodeSelectorTerms:

- matchExpressions:

- key: topology.kubernetes.io/zone

operator: In

values:

- korea

containers:

- name: mypod

image: alpine

resources:

limits:

memory: 100Mi

cpu: 100m

requests:

memory: 100Mi

cpu: 100m

command:

- sh

- -c

- |

apk add curl;

curl ipinfo.io
```

Now look at the pod logs. Did it run in Korea?

`kubectl logs test-geo`

When you will do the [storage tutorial](/documentation/userdocs/tutorial/storage/#exploring-storageclasses), there will be a section for choosing the right storage class. Remember that you can pick a set of nodes closer to your storage location (region is close enough).

## Optional Section: reserved nodes and taints. Using tolerations

There are [several parts of the NRP Kubernetes pool that are off-limits to regular users](/documentation/userdocs/running/special). One of them is Science-DMZ nodes that don’t have access to the regular internet.

Here is a Pod yaml that will try to run on one:

```
apiVersion: v1

kind: Pod

metadata:

name: test-suncavepod

spec:

affinity:

nodeAffinity:

requiredDuringSchedulingIgnoredDuringExecution:

nodeSelectorTerms:

- matchExpressions:

- key: kubernetes.io/hostname

operator: In

values:

- igrok-la.cenic.net

containers:

- name: mypod

image: alpine

resources:

limits:

memory: 100Mi

cpu: 100m

requests:

memory: 100Mi

cpu: 100m

command: ["sh", "-c", "sleep infinity"]
```

Go ahead, request a pod like that. Did the pod start?

It should not have. There are essentially no nodes that will match the above requirements. Check for yourself:

```
kubectl get events --sort-by=.metadata.creationTimestamp
```

Now, look up the list of nodes that were supposed to match:

```
kubectl get node igrok-la.cenic.net
```

Look for the node details. `kubectl get nodes igrok-la.cenic.net -o yaml`. Search for any taints. You should see something along the lines of:

```
...

spec:

taints:

- effect: NoSchedule

key: nautilus.io/noceph

value: "true"

...
```

You have been granted permission to run on those nodes, so let’s now add the toleration that will allow you to run there (remember to remove the old Pod):

```
apiVersion: v1

kind: Pod

metadata:

name: test-scidmzpod

spec:

affinity:

nodeAffinity:

requiredDuringSchedulingIgnoredDuringExecution:

nodeSelectorTerms:

- matchExpressions:

- key: kubernetes.io/hostname

operator: In

values:

- igrok-la.cenic.net

tolerations:

- effect: NoSchedule

key: nautilus.io/noceph

operator: Exists

containers:

- name: mypod

image: alpine

resources:

limits:

memory: 100Mi

cpu: 100m

requests:

memory: 100Mi

cpu: 100m

command: ["sh", "-c", "sleep infinity"]
```

Try to submit this one.

❓ Did the Pod start?

❗ Please make sure you did not leave any pods behind.

## The end

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/tutorial/scheduling.md)

[Previous  
Scaling and Exposing](/documentation/userdocs/tutorial/basic2)  [Next  
Batch Jobs](/documentation/userdocs/tutorial/jobs)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.