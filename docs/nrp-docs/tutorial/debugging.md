# Debugging

**Source:** https://nrp.ai/documentation/userdocs/tutorial/debugging

# Debugging

# Debugging

No one writes perfect code or executes scripts without failures or errors. Understanding how to diagnose and fix these roadblocks is an important skill. Because Nautilus is highly complex and Kubernetes requires additional skills, this page is written to help you understand some basic steps you can use for debugging.

Diagnosing and fixing errors in Kubernetes differs from troubleshooting issues on a physical machine in several key ways due to the distributed and dynamic nature of containerized environments. Here are some notable differences:

* **Abstraction Layer**: Kubernetes abstracts away many underlying details of the infrastructure, such as physical servers, networking configurations, and storage devices. This abstraction simplifies management but can make it more challenging to diagnose low-level hardware issues compared to troubleshooting on a physical machine.
* **Distributed Environment**: Kubernetes operates in a distributed environment with multiple nodes, each running multiple containers. Errors may stem from interactions between various components, including containers, pods, nodes, and the Kubernetes control plane. Diagnosing and fixing issues often involve understanding the interactions and dependencies between these distributed components.
* **Dynamic Resource Allocation**: Kubernetes dynamically schedules and manages resources across the cluster, automatically distributing workloads based on resource availability and workload requirements. Diagnosing errors may involve understanding how resources are allocated and ensuring proper resource utilization across the cluster.
* **Containerized Workloads**: In Kubernetes, applications run inside containers, which are isolated environments with their own filesystem, dependencies, and runtime environments. Errors may occur due to issues specific to containerized environments, such as misconfigured container images, resource constraints, networking issues, or container runtime errors.
* **Declarative Configuration**: Kubernetes provides built-in features for container orchestration, including scheduling, scaling, health checks, rolling updates, and service discovery. Diagnosing errors may involve understanding how these orchestration features interact with application workloads and identifying any misconfigurations or errors in the orchestration settings.

## Step-by-step Diagnosis of errors in Kubernetes

Diagnosing and fixing a Kubernetes Job that is in an “error” state involves several steps to identify the root cause of the problem and take appropriate corrective actions. Here’s a general approach you can follow:

1. **Inspect Job events**: Use the `kubectl describe` command to inspect the details of the job and its associated pods. Look for any events, errors, or warnings that may provide clues about what went wrong.
2. **Check Pod Logs**: Use the `kubectl logs <pod-name>` command to retrieve logs from the pods associated with the job. Look for any error messages, stack traces, or other indicators of problems that may have occurred during pod execution.
3. **Review Container Images**: Verify that the container images specified in the job’s pod template are accessible and properly configured. Ensure that the images contain the necessary application code, dependencies, and configurations required for job execution.
4. **Inspect Resource Constraints**: Check if resource constraints such as CPU and memory limits are properly configured for the job and its pods. Insufficient resources may lead to pod failures and errors.
5. **Review Job Configuration**: Review the configuration of the job, including parameters such as command, arguments, environment variables, volume mounts, and resource requests/limits. Ensure that the job configuration is correct and matches the intended behavior.
6. **Check Cluster Resource Availability**: Verify that there are sufficient cluster resources (CPU, memory, and GPU, if applicable) available to execute the job. Resource constraints or contention may prevent the job from starting or completing successfully.
7. **Monitor External Dependencies**: If the job interacts with external dependencies or services, monitor the status and availability of these external components. Errors or failures in external dependencies may impact job execution.
8. **Retry or Resubmit Job**: If the error is transient or due to temporary issues, consider retrying or resubmitting the job. Kubernetes allows you to delete and recreate failed jobs, which may resolve the issue if it was caused by a transient failure.

## Implement Error Logging

Error handling mechanisms such as error logging can be implemented within Kubernetes Pods using application-level logging libraries or by redirecting container output to standard output (stdout) and standard error (stderr). Here’s an example of a YAML file that demonstrates error logging using Kubernetes native mechanisms:

```
apiVersion: v1

kind: Pod

metadata:

name: error-handling-pod

spec:

containers:

- name: my-container

image: my-container-image:latest

command: ["sh", "-c", "echo 'Starting application'; ./my-application 2>&1 | tee /var/log/my-application.log"]

resources:

requests:

memory: "50Mi"

cpu: "100m"

limits:

memory: "50Mi"

cpu: "100m"

volumeMounts:

- name: logs-volume

mountPath: /var/log

volumes:

- name: logs-volume

emptyDir: {}
```

In this example, you are redirecting the standard error output that exists in `/var/log` and writing it out to an empty directory called “logs-volume” which both exist only inside the pod. If the pod vanishes, so will this directory. Instead, you can write these messages to a `persistent volume claim` (PVC) that you created for this purpose. In this way, the messages will be stored even if the pod goes away. See the section on [Storage](userdocs/tutorial/storage) for detailed information on creating and using PVCs.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/tutorial/debugging.md)

[Previous  
Storage](/documentation/userdocs/tutorial/storage)  [Next  
Distributing Images Using CVMFS](/documentation/userdocs/tutorial/nrp-software)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.