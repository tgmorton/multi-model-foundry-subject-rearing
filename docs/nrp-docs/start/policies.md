# Cluster Policies

**Source:** https://nrp.ai/documentation/userdocs/start/policies

# Cluster Policies

TL;DR

* Use [**`Job`**](https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/) to run [batch jobs](/documentation/userdocs/running/jobs/) and set right resources request.
* Use [**`Deployment`**](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) if you need a [**long-running pod**](/documentation/userdocs/running/long-idle/) and set **minimal** resources request.
* Use [monitoring](/documentation/userdocs/running/monitoring/) to accurately set the resource requests.
* Avoid wasting resources: if you’ve requested something, use it, and free it up once computation is done. Admins monitor resource utilization and will ban namespaces with under-utilized resources requests.
* A single user can’t submit **more than 4 pods** violating the following limits:
  + GPU > 40%, CPU 20-200%, RAM 20-150% of requested amount.
* Users running a **Job with `sleep` command** or equivalent (script ending with “sleep”) will be banned from using the cluster.
* Clean up your data frequently. Any storage volume that was **not accessed for 6 months can be purged without notification**.

### Acceptable Use Policy

Read the [NRP Acceptable Use Policy (AUP)](/NRP-AUP.pdf).

You need to accept the AUP before using the cluster

### Resource allocation

1. Set the resource (particularly CPU, memory and ephemeral-storage) `limits` properly. Your resource `limits` must be within 20% of the resource `requests`.
2. While running a large number (> ~100) of pods or jobs, you must choose your resource `limit` = `request`.
3. If a pod goes over its memory limit, it will be killed (OOM - Out of Memory).
4. If a pod exceeds the CPU limit, it puts pressure on the node. It can affect the whole node and other use’s work.

   Tip

   It is crucial to set `limits` correctly, but it is equally important not to set `requests` too high.

   * Aim to match your `requests` closely to the average resources you expect to use, while setting `limits` slightly above your highest anticipated peak.
   * Use [monitoring](/documentation/userdocs/running/monitoring/) to fine-tune these values effectively.
5. Do not waste resources. Namespaces with consistently underutilized requests risk being banned.

### Resource usage violations

1. A user can not submit **more than 4 pods**, violating the following conditions:
   * **GPU** utilization is more than 40% of the requested GPUs.
   * **CPU** usage is within 20-200% of the requested CPUs.
   * **Memory (RAM)** usage is within 20-150% of the requested memory.
2. These restrictions **do not apply** to pods requesting **1 CPU core and 2GB memory**.
3. Keep checking the [**Violations**](https://nrp.ai/userinfo) page on the Nautilus portal to see if your pods are violating the usage policies.

### Interactive use (6 hours max runtime)

1. Pods do not stop on their own under normal conditions and won’t recover if a node fails. Because of this, we assume any pod running without a controller is **interactive**, meaning it’s used temporarily for development or debugging.
2. **Time limit:** such pods will be **destroyed in 6 hours** unless you request an exception for your namespace (in case you run JupyterHub or some other application controlling the pods for you).
3. **Resource limits:** interactive pods are limited to 2 GPUs, 32 GB RAM, and 16 CPU cores.
4. It is okay to run `sleep` in an interactive pod.

### Batch jobs

Tip

If you need to run a larger and longer computation, use one of the available [Workload Controllers](https://kubernetes.io/docs/concepts/). We recommend running those as [Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/) - this will closely watch your workload and make sure it ran to completion (exited with 0 status), shut it down to free up the resources, and restart if the node was rebooted or something else has happened.

1. Use [Job](https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/) to run [batch jobs](/documentation/userdocs/running/jobs/) and set resources carefully.
   * You can use [Guaranteed QoS](https://kubernetes.io/docs/tasks/configure-pod-container/quality-service-pod/#create-a-pod-that-gets-assigned-a-qos-class-of-guaranteed) for those.
2. While running a large number (> ~100) of jobs, you must choose your resource `limit` = `request`.
3. Using `sleep infinity` in Jobs is not allowed. Users running a Job with `sleep infinity` command or equivalent (script ending with “sleep”) will be banned from using the cluster.

### Long-running idle pods

If you need some pods to run idle for a long time, you can use the [Deployment controller](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/#creating-a-deployment).

1. Use [**`Deployment`**](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) if you need a [long-running pod](/documentation/userdocs/running/long-idle/).
2. Make sure you set minimal requests and proper limits for those to get the [Burstable QoS](https://kubernetes.io/docs/tasks/configure-pod-container/quality-service-pod/#create-a-pod-that-gets-assigned-a-qos-class-of-burstable).
3. Such a deployment can not request a GPU.
4. Deployments are automatically deleted after 2 weeks (unless the namespace is added to the exceptions list and runs a permanent service).

### Workloads purging (2 weeks max runtime)

1. A periodic process removes workloads (deployments) **older than 2** weeks to free up resources and maintain system efficiency.
2. If you need to run [some permanent services](/documentation/userdocs/running/ingress/) beyond two weeks, contact admins in [Matrix](/contact) and ask for an exception. Please provide an estimated period of service functioning and a brief description of what the service does.
3. [Long idle pods](/documentation/userdocs/running/long-idle/) can’t be added to the exceptions list, since those are considered temporary and we need to be sure those are cleaned when not needed.
4. You will receive 3 notifications if your workload is not on the exception list. After that, your workload will be deleted. Any data in persistent volumes will remain.

### Requesting GPUs

1. When you [request GPUs for your pod](/documentation/userdocs/running/gpu-pods), nobody else can use those until you stop your pod. You should only schedule GPUs that you can actually use.
2. Check the [GPU dashboard](https://grafana.nrp-nautilus.io/d/dRG9q0Ymz/k8s-compute-resources-namespace-gpus) for your namespace to make sure the utilization is above 40%, and ideally is close to 100%.
3. The only reason to request more than a single GPU is when your GPU utilization is close to 100% and you can leverage using more GPUs.
4. GPUs are a limited resource shared by many users. If you plan on deploying large jobs (>50 GPUs) please present a plan in [Matrix](/contact).

Access Request for A100 GPUs

* To prevent misuse, the default quota for A100 GPUs is set to zero for all namespaces.
* To request access, complete the [A100 access](/reservations) request form online with the details of your workflow.

### Data purging

1. Please [purge](/documentation/userdocs/storage/purging/) any data you do not need. You should clean up your storage at regular intervals.
2. NRP resources should not be used as archival storage. You can only store the data actively used for computations.
3. Any volume that was **not accessed for 6 months can be purged without notification**.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/start/policies.mdx)

[Previous  
Hierarchical resources](/documentation/userdocs/start/hierarchy)  [Next  
Deployed Services](/documentation/userdocs/start/resources)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.