# Using Nautilus

**Source:** https://nrp.ai/documentation/userdocs/start/using-nautilus

# Using Nautilus

NRP Nautilus is a Kubernetes cluster distributed all over the US and there are a few sites in Asia and Europe. Currently there are over 75 sites as part of the NRP. See all the sites on the [NRP dashboard](https://dash.nrp-nautilus.io/).

Cluster Resource Summary

The cluster has different types of resources including CPU, GPU,and FPGAs.

* **CPUs:** 16 - 384 cores / node
* **GPUs:** max 16 in a node
* **FPGAs:** max 4 in a node
* **Memory (RAM):** 16 GB - 1.6 TB / node
* **Disk Space (local scratch):** 107.3 GB - 17.8 TB / node

Check the [**Resources**](https://nrp.ai/viz/resources) page on the portal to see the detailed summary of each node.

* The page also shows the available resouces in real time.

Check the cluster summary on [**Grafana**](https://grafana.nrp-nautilus.io/d/KQsJTWPiz/cluster-summary-nrp?orgId=1&from=now-1h&to=now&timezone=browser)

## How to use the cluster

There are three main ways to use the computing resources on the NRP Nautilus cluster.

1. NRP-hosted [JupyterHub platform](https://jupyterhub-west.nrp-nautilus.io)
2. NRP-hoster [Coder platform](https://coder.nrp-nautilus.io)
3. Interfacing with Kubernetes using the `kubectl` tool
4. Using the [managed cluster services](/documentation/userdocs/start/resources)

We will discuss these options below.

### JupyterHub Platform

[JupyterHub](/documentation/userdocs/jupyter/jupyterhub-service) is arguably the most user-friendly way to interact with the NRP Nautilus cluster. It allows you to run [Jupyter notebooks](https://jupyter.org/) in a web browser, without having to worry about the underlying infrastructure. You can access the NRP-hosted JupyterHub platform by visiting the [JupyterHub](https://jupyterhub-west.nrp-nautilus.io) link and logging in with your institutional credentials. Once authenticated, you can choose the hardware specs to spawn your instance and run Jupyter notebooks as usual.

[Access the NRP-hosted JupyterHub](https://jupyterhub-west.nrp-nautilus.io)

Points to Note

* You **do** need to be part of a namespace to use the NRP-hosted JupyterHub service.
* Your institution needs to be part of **`CILogon`**.
* Look at the [**JupyterHub documentation**](/documentation/userdocs/jupyter/jupyterhub-service) to learn more about this option.

### Coder Platform

[Coder](/documentation/userdocs/coder/coder) provides an easy-to-use, JupyterHub-like experience, and you can use the NRP Nautilus cluster without Kubernetes knowledge. [Coder](https://coder.com/) also runs on the web browser. You can run your code without worrying about the underlying infrastructure. You can access the NRP-hosted Coder platform by visiting the [Coder](https://coder.nrp-nautilus.io) link and logging in with your institutional credentials using **OpenID Connect** (once the cluster admins approve your account).

[Access the NRP-hosted Coder](https://coder.nrp-nautilus.io/)

Points to Note

* You **do** need to be part of a namespace to use the NRP-hosted Coder service.
* Cluster admins need to approve your account. Ask for access in [Matrix](/contact).
* Your institution needs to be part of **`CILogon`**.
* Look at the [**Coder documentation**](/documentation/userdocs/coder/coder) to learn more about this option.

### Kubernetes

This method provides greater control over your computing resources but requires basic Kubernetes knowledge. You can create pods, jobs, and deployments while specifying the required CPU, GPU, memory, and other resources. This is particularly useful for running custom software stacks or jobs with specific resource requirements.

[Get started with Kubernetes setup](/documentation/userdocs/start/getting-started)

Points to Note

* You need to be part of at least one namespace. Follow the [Getting Started](/documentation/userdocs/start/getting-started) page
* Your institution needs to be part of **`CILogon`**.
* Need to know basic Kubernetes.
* Complete the [**Basic Kubernetes**](/documentation/userdocs/tutorial/basic) tutorial.

## Kubernetes concepts

This section highlights some of the Kubernetes concepts that are essential to start using the Nautilus cluster. You can learn more about these components in the [tutorial sections]((/documentation/userdocs/tutorial/basic/)). If you are new to Kubernetes and like to learn about the Kubernetes concepts in detail, please follow the [**Concepts page from Kubernetes Documentation**](https://kubernetes.io/docs/concepts/).

### Namespace

One of the main concepts of running workloads in Kubernetes is a namespace. A namespace creates an isolated environment in which you can run your pods. It is possible to invite other users to your namespace and it is possible to have access to multiple namespaces. Each namespace has at least one namespace admin (usually researchers, engineers, or faculties). If you are new to the cluster, please follow the instructions on the [Getting Started](/documentation/userdocs/start/getting-started#get-access-and-log-in) page.

Who can be a Namespace Admin

* A **faculty member, researcher, or postdoc**. Contact us in [Matrix](/contact).
* A student (preferably a Ph.D. or Masters student) can be a namespace admin if their supervisor sends us a request via email or [Matrix](/contact).

### Container images

A container image is a lightweight, standalone, and executable software package containing everything needed to run a piece of software: the code, runtime, libraries, dependencies, and default values for any essential settings. Images are typically stored in container registries (e.g., Docker Hub, GitLab Container Registry) and can be pulled by Kubernetes nodes when running your code.

Tip

* Create an image (e.g. Docker image) to run your code on Nautilus.
* Store it in one of the container registries. You can use [NRP-hosted GitLab](https://gitlab.nrp-nautilus.io/) to store your container image.

### Pod, Job and Deployment

A [Kubernetes pod](https://kubernetes.io/docs/concepts/workloads/pods/) is the smallest and most basic deployable unit in Kubernetes, representing a single instance of a running process. A pod can contain one or more tightly coupled containers that share the same network and storage. If you frequently deploy single-container pods, you can generally replace the word “pod” with “container.”

Caution

Single pods without any Workload Controllers will be destroyed after **6 hours**. Look at the [**cluster policies**](/documentation/userdocs/start/policies#interactive-use-6-hours-max-runtime).

* Pods are ephemeral and get deleted or evicted due lack of resources on the node or other node issues.
* Be careful while using a single pod without any controller.

A [Kubernetes job](https://kubernetes.io/docs/concepts/workloads/controllers/job/) is a resource used to run short-lived, batch, or parallel processing tasks to completion. It creates one or more pods and retries execution until the specified number of successful completions is reached. It tracks completed pods and terminates once the specified number of successes is achieved. Deleting a job will clean up all the jobs it created.

Caution

* Do not run long `sleep` inside a job. See the Look at the [**cluster policies**](/documentation/userdocs/start/policies#batch-jobs)
* Do not submit more than 400 Jobs at once.
* If you find an issue in your job, delete the job otherwise it will keep creating pods.

A [Kubernetes deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) manages the lifecycle of application workloads by ensuring the desired number of pod replicas are running and up to date. It supports rolling updates, rollbacks, and scaling to maintain application availability. Deployments automatically replace failed or outdated pods, making them ideal for managing stateless applications in a cluster.

Caution

* Deployments are automatically deleted after 2 weeks unless you ask for an exception.
* Use deployments if you need to run [long-running idle pods](/documentation/userdocs/running/long-idle/).

Note

In almost every case, **Kubernetes jobs are the preferred way to run large-scale processing on the NRP**. This is because jobs are designed to run to completion, and can be scaled up or down as needed.

| Feature | Job | Pod | Deployment |
| --- | --- | --- | --- |
| **Description** | A job is a task that runs to completion. | A pod is a group of one or more containers, with shared storage and network resources. | A deployment is a way to manage a collection of pods. |
| **Management** | Runs jobs to successful completion, handles node failures | Runs a container, no handling of node failure or conatiner failure | Manages pods, scales up or down, handles conatiner failures |
| **Paralellism** | Can run arrays of jobs, or parallel jobs (see [Kubernetes docs](https://kubernetes.io/docs/concepts/workloads/controllers/job/#parallel-jobs)) | Can run multiple containers in a pod | Can run multiple replicas of a pod |
| **Max Resources** | Can specify resources needed for the job | 2 GPUs, 32 GB RAM and 16 CPU cores | Can specify resources needed for the deployment |
| **Max Runtime** | Runs to completion | 6 hours | 2 weeks |
| **Documentation** | [Jobs](/documentation/userdocs/running/jobs) | [Pods](https://kubernetes.io/docs/concepts/workloads/pods/) | [Deployments](/documentation/userdocs/running/long-idle) |

### Resource allocation

For all the abovementioned Kubernetes objects (pods, jobs, deployments, etc) you need to ask for computing resources like (CPU, GPU, memory) before creating the objects.

* A **`request`** is the minimum amount of CPU, GPU, or memory that Kubernetes guarantees to a container. Kubernetes scheduler reserves this amount of resources to schedule a pod.
* A **`limit`** is the maximum amount of CPU or memory that a container is allowed to use.

Caution

* If a node cannot satisfy the resource requests, the pod will not be scheduled on that node.
* If a container exceeds the memory limit, it will be killed (OOMKilled - Out of Memory).
* If a container exceeds the CPU limit, Kubernetes throttles it (slows down execution).

### Storage

A [Kubernetes storage](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) provides a way to persist and manage data for containerized applications. It supports various storage options, including temporary storage (emptyDir), and persistent storage (PersistentVolumes). On Nautilus, you will be using both temporary (`ephemeral-storage`) and persistent storage for your work.

A pod can access the requested `ephemeral-storage` from the local disks of the node. Users need to create their own [PersistentVolumeClaim (PVC)](/documentation/userdocs/tutorial/storage#creating-a-persistent-volume-claim) under their namespace.

Tip

* Complete the Storage section from [the tutorial](/documentation/userdocs/tutorial/storage).
* Read more about the different [storage options](/documentation/userdocs/storage/intro) on Nautilus.

### System load

While you are welcome to use as many resources as needed, using those inefficiently causes problems for others. The CPU load of the system consists of user load, system load, and several others. If you run `top`, you’ll see something like:

Terminal window

```
%Cpu(s): 26.9 us,  1.5 sy,  0.0 ni, 71.5 id,  0.0 wa,  0.0 hi,  0.1 si,  0.0 st
```

This means 26.9% of all system CPU time is spent on user tasks (computing), 1.5% on system stuff (kernel tasks), 71.5 idle, and so on. If the system (kernel) load is more than ~15%, it indicates a problem with the user’s code. Usually, it is caused by too many threads (the system is spending too much time just switching between those), inefficient file I/O (lots of small files processed by too many workers), or something similar. In this case, the system time is wasted not on processing.

> If you were told your pod is causing too much load, look at your code and try to figure out why it is spending too much kernel time instead of computations.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/start/using-nautilus.mdx)

[Previous  
Getting Started](/documentation/userdocs/start/getting-started)  [Next  
Hierarchical resources](/documentation/userdocs/start/hierarchy)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.