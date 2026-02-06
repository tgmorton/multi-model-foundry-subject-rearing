# Kubeflow Training

**Source:** https://nrp.ai/documentation/userdocs/running/kubeflow

# Kubeflow Training

The Kubeflow Training Operator simplifies the management of distributed training jobs on Kubernetes. It allows users to define training jobs as Kubernetes custom resources, making it easy to scale and monitor machine learning models within a Kubernetes environment.

This guide explains how to use the Kubeflow Training Operator to submit, manage, and monitor training jobs.

---

## Key Concepts

* **TrainingJob**: A custom resource definition (CRD) used to define training jobs. It includes the details of the training task, such as the container image, number of replicas, and more.
* **Worker**: A replica that runs the training logic. Multiple workers enable distributed training.
* **PS (Parameter Server)**: Used in some distributed training frameworks to handle model parameter updates.
* **Chief/Leader**: A single pod responsible for coordinating the training (used in TensorFlow jobs).

---

## Submitting a Training Job

To submit a training job, you need to create a YAML manifest file that describes the job. Below is an example YAML file for a TensorFlow training job.

```
apiVersion: "kubeflow.org/v1"

kind: TFJob

metadata:

name: example-tfjob

spec:

tfReplicaSpecs:

Chief:

replicas: 1

restartPolicy: OnFailure

template:

spec:

containers:

- name: tensorflow

image: tensorflow/tensorflow:2.17.0

command: ["python", "/app/train.py"]

args: ["--epochs", "5"]

resources:

requests:

memory: "4Gi"

cpu: "2"

limits:

memory: "4Gi"

cpu: "2"

Worker:

replicas: 2

restartPolicy: OnFailure

template:

spec:

containers:

- name: tensorflow

image: tensorflow/tensorflow:2.17.0

command: ["python", "/app/train.py"]

args: ["--epochs", "5"]

resources:

requests:

memory: "4Gi"

cpu: "2"

limits:

memory: "4Gi"

cpu: "2"
```

Common options for `kind` are TFJob for TensorFlow and PyTorchJob for PyTorch.

For more information on specific training jobs like PyTorch or MXNet, or to explore additional features such as hyperparameter tuning, refer to the [Kubeflow documentation](https://www.kubeflow.org/docs/components/training/getting-started/).

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/kubeflow.md)

[Previous  
Dask Cluster](/documentation/userdocs/running/dask-cluster)  [Next  
High I/O jobs](/documentation/userdocs/running/io-jobs)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.