# High I/O Jobs

**Source:** https://nrp.ai/documentation/userdocs/running/io-jobs

# High I/O Jobs

If you hit the Ceph speed bottleneck, especially if you have many small files, don’t try to push it more.

Consider using a method to open multiple parallel streams (such as using `gcloud storage` / `gsutil`), or aggregating many small files to a few large files (such as using parallel archiver tools like `7-Zip` with LZMA2, `pbzip2`, `ripunzip`/`piz`, or `zpaq`) from/to the Ceph filesystem.

Start with 1 GPU and a representative / subsampled data set and look at utilization as a function of CPU and thread core count; only go above 1 GPU once you’ve convinced yourself you are GPU bound. It is a balancing act between I/O, threads/CPU, and GPUs. The jumps in GPU performance coupled with I/O bound bottlenecks are the indication of being oversubscribed in terms of # of total GPUs requested.

You can try copying the files from ceph to the local node disk. Most of our nodes have NVME drives, which provide hundred(s) times better performance than remote Ceph storage.

#### Adjust ephemeral-size to your data volume, otherwise pod can be killed by kubernetes

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

initContainers:

- name: init-data

image: gitlab-registry.nrp-nautilus.io/prp/gsutil

args:

- gsutil

- "-m"

- rsync

- "-erP"

- /mnt/source/

- /mnt/dest/

volumeMounts:

- name: source

mountPath: /mnt/source

- name: data

mountPath: /mnt/dest

volumes:

- name: data

emptyDir: {}

- name: source

persistentVolumeClaim:

claimName: examplevol

restartPolicy: Never

backoffLimit: 5
```

#### Ram Disk

You can mount a RAM disk to your pod to reduce IO pressure. The RAM disk is dynamically sized, but does not count against your PODs memory request. The [kubernetes documentation describes setting it up](https://kubernetes.io/docs/concepts/storage/volumes/#emptydir), the example below contains just the components you need to add to your YAML file to mount a ram disk.

```
spec:

template:

spec:

containers:

volumeMounts:

- name: "ram-disk"

mountPath: "/ramdisk"

volumes:

- name: ram-disk

emptyDir:

medium: "Memory"
```

#### Large dataset sampling strategies that work (and issues to watch out for)

If you have a large dataset (>100 GB) and need to sample data from it at a high IO rate (>100 MB/sec) the suggestions below will apply to you.

* Store your dataset on the NRP/[S3](/documentation/userdocs/storage/ceph-s3/) interface. The S3 interface is scalable and universally accessible, so it make a great place to store large datasets which can be simultaneously accessed either from inside the cluster or externally.
* Download the data to the local container in a rolling window. For example, start a process which continuously downloads and maintains 30 GB of data files locally, and sample from the local data files that are available at the moment. If you run Python and boto3 to sequentially download files your download rates will usually be in the neighborhood of 100 MB/sec.
  + A good approach to doing this is to run a separate process which manages download files and deleting previous files in a rolling window. Since linux allows you to safely delete files which have open file pointers (the file remains until all file pointers are closed, e.g. any process reading a file that is deleted won’t have an issue) this process can be fully independent of other processes and thus easy to manage.
  + One thing to consider with a windowed sampling approach is that there may be an optimal ordering to download your data files such that you maintain the best distribution of samples across your local window of the dataset. Consider this if your data files have imbalanced classes.
* Spawn one or more processes to sample data from available local files. If you are running Python and Tensorflow you want these processes separate from your main training loop, otherwise you will encounter problems with the Python GIL being slowed down by the deserialization process.
* *If you run Python/Tensorflow* your sampling processes should create a RAM Disk and write the samples to the standard Tensorflow TF Records format files, then hand those file names off to `tf.data.TFRecordDataset`. If you try to pass the data to the main process via say `multiprocessing` you will lock up the Python GIL in deserialization and it will negatively impact your training loop. By writing TF Records files, the deserialization of the data happens in Tensorflow, which is in C, not in Python and thus doesn’t negatively impact your training loop due to the limitations of the Python GIL.
* Each of the processes described here (Downloading, Sampling , and Training loop) can be written in a nicely decoupled manner, making them fairly easy to write and debug. Try to avoid a lot of dependencies between the code that handles downloading data vs. code that handle sampling vs. code that runs the training loop. Write them as independent classes and processes. This approach allows an application to sample data at a rate higher than it can access directly from S3, which avoids difficult to avoid bottlenecks.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/io-jobs.md)

[Previous  
Kubeflow](/documentation/userdocs/running/kubeflow)  [Next  
CPU throttling](/documentation/userdocs/running/cpu-throttling)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.