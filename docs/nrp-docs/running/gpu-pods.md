# GPU Pods

**Source:** https://nrp.ai/documentation/userdocs/running/gpu-pods

# GPU Pods

Note

In this section you will request GPUs. Make sure you don’t waste those and delete your pods when not using the GPUs.

Caution

Some specific high-memory GPUs require the gpu type specified in the container `resource` requests and limits. See [Requesting special GPUs](#requesting-special-gpus)

## Running GPU pods

Use this definition to create your own pod and deploy it to kubernetes:

```
apiVersion: v1

kind: Pod

metadata:

name: gpu-pod-example

spec:

containers:

- name: gpu-container

image: tensorflow/tensorflow:latest-gpu

command: ["sleep", "infinity"]

resources:

limits:

nvidia.com/gpu: 1

requests:

nvidia.com/gpu: 1
```

This example requests 1 GPU device. You can have up to 8 per node if you’re [using jobs](/documentation/userdocs/running/jobs/), and up to 2 for pods. If you request GPU devices in your pod, kubernetes will auto schedule your pod to the appropriate node. There’s no need to specify the location manually.

You should always delete your pod when your computation is done to let other users use the GPUs.

Consider using [Jobs](/documentation/userdocs/running/jobs/) **with actual script instead of `sleep`** whenever possible to ensure your pod is not wasting GPU time. If you have never used Kubernetes before, see the [tutorial](/documentation/userdocs/tutorial/introduction/).

## Requesting special GPUs

Certain kinds of GPUs are advertised on nodes as a special resource, f.e. “nvidia.com/rtx-8000”. You have to request that resource instead of the “nvidia.com/gpu” one.

The current list is:

| GPU Type | resource |
| --- | --- |
| A40 | nvidia.com/a40 |
| A100 | nvidia.com/a100 |
| Nvidia RTX A6000 | nvidia.com/rtxa6000 |
| Quadro RTX 8000 | nvidia.com/rtx8000 |
| Grace Hopper GH200 | nvidia.com/gh200 |
| A100 MIG 1g.10gb | nvidia.com/mig-small |

Using A100s also requires [a reservation](/reservations).

For example, modifying the above example for one of these GPUs, the new yaml would be:

```
apiVersion: v1

kind: Pod

metadata:

name: gpu-pod-example

spec:

containers:

- name: gpu-container

image: tensorflow/tensorflow:latest-gpu

command: ["sleep", "infinity"]

resources:

limits:

nvidia.com/a100: 1

requests:

nvidia.com/a100: 1
```

For Grace Hopper node make sure you’re also using the image with `arm` support (`nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` is a good one) and [tolerating the arm64 architecture](/documentation/userdocs/running/special/):

```
tolerations:

- key: "nautilus.io/arm64"

operator: "Exists"

effect: "NoSchedule"
```

## Requesting many GPUs

Since 1 and 2 GPU jobs are blocking nodes from getting 4 and 8-GPU jobs, there are some nodes reserved for those. Once you submit a job with 4 or 8 GPUs request, a controller will automatically add toleration which will allow you to use the node reserved for more GPUs. You don’t need to do anything manually for that.

## Choosing GPU type

**See [requesting special GPUs](#requesting-special-gpus) for special types of GPU**

We have a variety of GPU flavors attached to Nautilus. This table describes the types of GPUs available for use, but is not up to date - it’s better to use the actual cluster information (f.e. `kubectl get nodes -L nvidia.com/gpu.product`).

Credit: [GPU types by NRP Nautilus](https://observablehq.com/d/7c0f46855b4212e0)

If you need more graphical memory, use this table or official specs to choose the type:

| GPU Type | Memory size (GB) |
| --- | --- |
| NVIDIA-GeForce-GTX-1070 | 8G |
| NVIDIA-GeForce-GTX-1080 | 8G |
| Quadro-M4000 | 8G |
| NVIDIA-A100-PCIE-40GB-MIG-2g.10gb | 10G |
| NVIDIA-GeForce-GTX-1080-Ti | 12G |
| NVIDIA-GeForce-RTX-2080-Ti | 12G |
| NVIDIA-TITAN-Xp | 12G |
| Tesla-T4 | 16G |
| NVIDIA-A10 | 24G |
| NVIDIA-GeForce-RTX-3090 | 24G |
| NVIDIA-GeForce-RTX-4090 | 24G |
| NVIDIA-TITAN-RTX | 24G |
| NVIDIA-RTX-A5000 | 24G |
| Quadro-RTX-6000 | 24G |
| Tesla-V100-SXM2-32GB | 32G |
| NVIDIA-A40 | 48G |
| NVIDIA-L40 | 48G |
| NVIDIA-RTX-A6000 | 48G |
| Quadro-RTX-8000 | 48G |
| NVIDIA-A100-SXM4-80GB | 80G |

Note

[Not all nodes are available to all users](/documentation/userdocs/running/special/). You can consult about your available resources in [Matrix](https://nrp.ai/contact) and on [resources page](https://nrp.ai/viz/resources). Labs connecting their hardware to our cluster have preferential access to all our resources.

Caution

For higher memory GPUs, use the [requesting special GPUs syntax](#requesting-special-gpus)! Affinity allows you to further refine the GPU type or choose the GPU type for generic GPUs.

To use a **specific type of GPU**, add the affinity definition to you pod yaml file. The example below specifies *1080Ti* GPU:

```
spec:

affinity:

nodeAffinity:

requiredDuringSchedulingIgnoredDuringExecution:

nodeSelectorTerms:

- matchExpressions:

- key: nvidia.com/gpu.product

operator: In

values:

- NVIDIA-GeForce-GTX-1080-Ti
```

**To make sure you did everything correctly** after you’ve submited the job, look at the corresponding pod yaml (`kubectl get pod ... -o yaml`) and check that resulting nodeAffinity is as expected.

## Selecting CUDA version

In general the higher CUDA versions support the lower and same driver version. The nodes are labelled with the major and minor CUDA and driver versions. You can check those at the [resources page](https://nrp.ai/viz/resources) or list with this command (it will also choose only GPU nodes):

Terminal window

```
kubectl get nodes -L nvidia.com/cuda.driver.major,nvidia.com/cuda.driver.minor,nvidia.com/cuda.runtime.major,nvidia.com/cuda.runtime.minor -l nvidia.com/gpu.product
```

If you’re using the container image with higher CUDA version, you have to pick the nodes supporting it. Example:

```
spec:

affinity:

nodeAffinity:

requiredDuringSchedulingIgnoredDuringExecution:

nodeSelectorTerms:

- matchExpressions:

- key: nvidia.com/cuda.runtime.major

operator: In

values:

- "12"

- key: nvidia.com/cuda.runtime.minor

operator: In

values:

- "2"
```

Also you can choose the driver above something if you know which one you need (this will pick drivers **above** 535):

```
spec:

affinity:

nodeAffinity:

requiredDuringSchedulingIgnoredDuringExecution:

nodeSelectorTerms:

- matchExpressions:

- key: nvidia.com/cuda.driver.major

operator: Gt

values:

- "535"
```

## Adding Shared Memory (shm)

Adding Shared Memory (shm)

You can add Shared Memory (shm) to your GPU pods in YAML:

```
volumeMounts:

- mountPath: /dev/shm

name: dshm

volumes:

- name: dshm

emptyDir:

medium: Memory

sizeLimit: 2Gi
```

`sizeLimit` is an optional term. Otherwise it defaults to half of the memory request. Not adding /dev/shm defaults to 64MB.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/gpu-pods.md)

[Previous  
MNIST Training (Presentation)](https://docs.google.com/presentation/d/1GMvaZr9Nm6LhYUU_E0E0LdoebPpk0dgb2Z6oS9v2Ww8/edit?usp=sharing)  [Next  
Long idle pods](/documentation/userdocs/running/long-idle)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.