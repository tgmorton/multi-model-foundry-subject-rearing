# Local Scratch

**Source:** https://nrp.ai/documentation/userdocs/storage/local

# Local Scratch

Most nodes in the cluster have local NVMe drives, which provide faster I/O than shared filesystems. These can be used for workloads that require very intensive I/O operations ([see recommendations and an example for running these](/documentation/userdocs/running/io-jobs/)).

You can request an [ephemeral volume](https://kubernetes.io/docs/concepts/storage/volumes/#emptydir) to be attached to your pod as a fast scratch space. Note that any information stored in it will be destroyed after pod shutdown.

```
apiVersion: batch/v1

kind: Job

metadata:

name: myapp

spec:

template:

spec:

containers:

- name: demo

image: gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp

command:

- "python"

args:

- "/home/my_script.py"

- "--data=/mnt/data/..."

volumeMounts:

- name: data

mountPath: /mnt/data

resources:

limits:

memory: 8Gi

cpu: "6"

nvidia.com/gpu: "1"

ephemeral-storage: 100Gi

requests:

memory: 4Gi

cpu: "1"

nvidia.com/gpu: "1"

ephemeral-storage: 100Gi

volumes:

- name: data

emptyDir: {}

restartPolicy: Never

backoffLimit: 5
```

Please note that in case a node starves on disk, ALL pods will be evicted from the node. If you set the request to be 50G, and limit is 100G, and you use 100G, it’s likely this will destroy the node, as scheduler will put your workload on a 50G node. So make sure your request is close to the limit you set.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/storage/local.md)

[Previous  
CVMFS](/documentation/userdocs/storage/cvmfs)  [Next  
Linstor](/documentation/userdocs/storage/linstor)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.