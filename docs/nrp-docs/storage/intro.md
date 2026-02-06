# Introduction

**Source:** https://nrp.ai/documentation/userdocs/storage/intro

# Introduction

Tip

We have storage distributed across multiple geographic regions. To achieve optimal performance and speed when accessing it, ensure you select the [appropriate compute nodes](#using-the-right-region-for-your-pod) for your pod.

## Cleaning up

**Please [purge](/documentation/userdocs/storage/purging/) any data you don’t need. We’re not an archival storage, and can only store the data actively used for computations.**

**Any volume that was not accessed for 6 months can be purged without notification.**

## POSIX volumes

Most persistent data in Kubernetes comes in a form of [Persistent Volumes (PV)](https://kubernetes.io/docs/concepts/storage/persistent-volumes/), which can only be seen by cluster admins. To request a PV, you have to create a [PersistentVolumeClaim (PVC)](https://kubernetes.io/docs/concepts/storage/persistent-volumes/#persistentvolumeclaims) of a supported [StorageClass](https://kubernetes.io/docs/concepts/storage/storage-classes/) in your namespace, which will allocate storage for you.

### Provided filesystems

Credit: [Combined filesystems](https://observablehq.com/d/f2f8ca16077504f1)

### How to choose which filesystem to use

**Read the [High I/O jobs](/documentation/userdocs/running/io-jobs/) guide on optimizing your storage performance.**

**RBD (Rados Block Device)** is similar to a normal hard drive as it implements block storage on top of Ceph, and can run many kinds of file I/O operations including small files. Thus, it may accommodate conda/pip installation and code compilation. It can also provide higher IOPS than CephFS, but overall read/write performance tends to be slower than CephFS because it is less parallelized. Optimal read/write performance can be achieved when using a program or library that supports [librados](https://docs.ceph.com/en/latest/rados/api/librados-intro/) or when you code your own program using the library. Use this for housing databases or workloads that require quick response but not necessarily high read/write rates. It shares the storage pool with CephFS, therefore being the largest storage pool in Nautilus.

**CephFS** is a distibuted parallel filesystem which stores files as objects. It **cannot** handle lots of small files rapidly because it has to use metadata servers for annotating the files. Thus, conda/pip and code compilation should not be performed over CephFS. However, it has a much higher read/write performance than RBD, **especially if you are able to open multiple parallel streams to a file and aggregate many small files to a few large files**. CephFS has the largest storage pool in Nautilus, and thus it is suitable for workloads that deal with comparably larger files than RBD which requires high I/O performance, for example checkpoint files of various tasks. There is a per-file size limit of 16 TB in CephFS.

**CVMFS** provides read-only access to data on **OSDF origins** via a set of [OSDF](https://osdf.osg-htc.org/) caches, that can be mapped as a PVC to the pods. The access is read-only, and this is mostly used for rarely changing large files collections, like software packages and large training datasets.

[Linstor](https://linbit.com/linstor/) provides the fastest and the smallest latency block storage, but can’t handle large (>10TB) volumes. Can be used for VM images, high-loaded databases, etc.

**Linstor is the only block storage that can recover from a node that mounts the storage going offline - the pod will be migrated to another node. In case of Ceph it will be stuck until the node is back online. This is good for services requiring HA.**

[Comparison of Linstor vs Ceph by Linstor](https://linbit.com/blog/how-does-linstor-compare-to-ceph/)

Ceph provides an S3-compatible protocol interface (with a per-file size limit of 5 TiB). This is a native object storage protocol that can supply the maximum read/write performance. It uses the HTTP protocol instead of POSIX, if your tool supports the protocol instead of only POSIX file I/O. Many data science tools and libraries support the S3-compatible protocol as an alternative file I/O interface, and the protocol is well optimized for such purposes.

S3 storage doesn’t use PVCs, and can be accessed directly by applications.

### Creating and mounting the PVC

Use kubectl to create the PVC:

```
apiVersion: v1

kind: PersistentVolumeClaim

metadata:

name: examplevol

spec:

storageClassName: <required storage class>

accessModes:

- <access mode, f.e. ReadWriteOnce >

resources:

requests:

storage: <volume size, f.e. 20Gi>
```

After you’ve created a PVC, you can see its status (`kubectl get pvc pvc_name`). Once it has the Status `Bound`, you can attach it to your pod (claimName should match the name you gave your PVC):

```
apiVersion: v1

kind: Pod

metadata:

name: vol-pod

spec:

containers:

- name: vol-container

image: ubuntu

args: ["sleep", "36500000"]

volumeMounts:

- mountPath: /examplevol

name: examplevol

restartPolicy: Never

volumes:

- name: examplevol

persistentVolumeClaim:

claimName: examplevol
```

### Using the right region for your pod

Latency significantly affects the I/O performance. If you want optimal access speed to Ceph, add the region affinity to your pod for the correct `region` (`us-east`, `us-west`, etc):

```
spec:

affinity:

nodeAffinity:

requiredDuringSchedulingIgnoredDuringExecution:

nodeSelectorTerms:

- matchExpressions:

- key: topology.kubernetes.io/region

operator: In

values:

- us-west
```

You can list the nodes region label using: `kubectl get nodes -L topology.kubernetes.io/region,topology.kubernetes.io/zone`

### Volumes expanding

All volumes created starting from December 2020 can be expanded by simply modifying the `storage` field of the PVC (either by using `kubectl edit pvc ...`, or `kubectl update -f updated_pvc_definition.yaml`)

For older ones, all `rook-ceph-block-*` and most `rook-cephfs-*` volumes can be expanded. If yours is not expanding, you can ask cluster admins to do it in manual mode.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/storage/intro.md)

[Previous  
SmartNIC: Running](/documentation/userdocs/fpgas/esnet_running)  [Next  
Ceph FS / RBD](/documentation/userdocs/storage/ceph)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.