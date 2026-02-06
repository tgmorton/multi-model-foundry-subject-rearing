# CPU Only Jobs

**Source:** https://nrp.ai/documentation/userdocs/running/cpu-only

# CPU Only Jobs

## Running large CPU-only jobs

Nautilus is primarily used for GPU jobs. While it’s possible to run large CPU-only jobs, you have to take certain measures to prevent taking over all cluster resources.

You can run the jobs with lower priority and allow other jobs to preempt yours. This way you should not worry about the size of your jobs and you can use the maximum number of resources in the cluster. To do that, add the `opportunistic` priority class to your pods:

```
spec:

priorityClassName: opportunistic
```

Another thing to do is to avoid the GPU nodes. This way you can be sure you’re only using the CPU-only nodes and jobs are not preventing any GPU usage. To do this, add the node antiaffinity for GPU device to your pod:

```
spec:

affinity:

nodeAffinity:

requiredDuringSchedulingIgnoredDuringExecution:

nodeSelectorTerms:

- matchExpressions:

- key: feature.node.kubernetes.io/pci-10de.present

operator: NotIn

values:

- "true"
```

You can use a combination of 2 methods or either one.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/cpu-only.md)

[Previous  
Running batch jobs](/documentation/userdocs/running/jobs)  [Next  
Scheduling](/documentation/userdocs/running/scheduling)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.