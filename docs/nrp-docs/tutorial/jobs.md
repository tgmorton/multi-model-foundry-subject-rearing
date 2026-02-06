# Running Batch Jobs

**Source:** https://nrp.ai/documentation/userdocs/tutorial/jobs

# Running Batch Jobs

# Running batch jobs

The Nautilus Cluster is designed specifically to support high-throughput batch jobs.

In Kubernetes, a batch job is a type of workload designed to run a finite number of tasks to completion, as opposed to continuously running or long-lived services. Batch jobs are ideal for executing tasks such as data processing, data analysis, batch data updates, backups, or any other task that needs to be performed periodically or on-demand.

A batch job (or simply, a job) is a daemon which watches your pod and makes sure it exited with exit status 0. If it did not for any reason, it will be restarted up to `backoffLimit` number of times.

What is the difference between a Job and a Pod?

A Job is a higher-level abstraction that manages a Pod. A Job will ensure that the Pod runs to completion, and can be scaled up or down as needed. A Pod is a group of one or more containers, with shared storage and network resources.

## Prerequisites

This section builds on skills from both the [Quickstart](/documentation/userdocs/start/getting-started/) and the tutorial on [Basic Kubernetes](/documentation/userdocs/tutorial/basic).

## Learning Objectives

1. You will learn how to create a simple job that will execute a command, then run to completion.
2. You will have a preliminary understanding of job states, such as “Completed” or “Error”.
3. You will understand how to set limits to jobs

Note

Since jobs in Nautilus are not limited in runtime, you can only run jobs with meaningful `command` field. Running in manual mode (`sleep infinity` `command` and manual start of computation) is prohibited.

Let’s run a simple job and get it’s result.

Create a a file called `job.yaml` file and submit ito the cluster:

```
apiVersion: batch/v1

kind: Job

metadata:

name: pi

spec:

template:

spec:

containers:

- name: pi

image: perl

command: ["perl",  "-Mbignum=bpi", "-wle", "print bpi(2000)"]

resources:

limits:

memory: 200Mi

cpu: 1

requests:

memory: 50Mi

cpu: 50m

restartPolicy: Never

backoffLimit: 4
```

Explore what’s running:

```
kubectl get jobs

kubectl get pods
```

When job is finished, your pod will stay in Completed state, and Job will have COMPLETIONS field 1/1. For long jobs, the pods can have Error, Evicted, and other states until they finish properly or backoffLimit is exhausted.

How would you diagnose and fix a Job that exited with the code “error”?

You could view the logs of the job with the logs command:

Terminal window

```
kubectl logs pi-<hash>
```

Note

Learn more about Job states by visiting the Kubernetes documentation about [Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/job/). There are many features of Jobs that can be utilized to run your application at-scale.

Our job did not use any storage and output the result to STDOUT, which can be seen as our pod logs:

```
kubectl logs pi-<hash>
```

The pod and job will remain for you to come and look at for `ttlSecondsAfterFinished=604800` seconds (1 week) by default, and you can adjust this value in your job definition if desired.

You can use the [more advanced example](/documentation/userdocs/running/jobs) when ready.

## The end

Please make sure you did not leave any pods and jobs behind.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/tutorial/jobs.md)

[Previous  
Scheduling](/documentation/userdocs/tutorial/scheduling)  [Next  
Images](/documentation/userdocs/tutorial/images)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.