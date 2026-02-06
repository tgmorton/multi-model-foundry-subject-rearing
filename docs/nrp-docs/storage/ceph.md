# Ceph FS / RBD

**Source:** https://nrp.ai/documentation/userdocs/storage/ceph

# Ceph FS / RBD

No Conda or PIP on CephFS

Installing `conda` and `pip` packages on all CephFS (shared) filesystems is strictly prohibited!

## Best Practices for File Access in shared filesystems

**Use Unique Files:** Always create or use unique files rather than sharing the same file to minimize conflicts.

**Avoid Long-Lived Open Handles:** Close files as soon as you are done working with them to prevent holding locks longer than necessary.

**Favor Copy Operations:** If multiple clients need to work with the same data, consider using a copy of the file rather than the original.

## Avoiding Write Conflicts in shared filesystems

When using CephFS, it is crucial not to open the same file for write from multiple parallel clients. Doing so can lead to:

**File Locking Issues:** Concurrent writes can lock the file, potentially blocking other clients.

**Data Corruption:** Simultaneous writes to a single file can corrupt the file system or the data contained within.

## Recommended Strategies to Avoid Conflicts:

**File Versioning:** Implement a versioning system for files that multiple clients may need to update.

**Locking Mechanisms:** Utilize designated locking mechanisms to control access to shared resources if required for specific applications.

**Coordination Among Clients:** Use coordination protocols or systems to manage write access among different clients.

## Ceph filesystems data use

Credit: [Ceph data usage](https://observablehq.com/d/b9c19d9f7c57a186)

[General ceph grafana dashboard](https://grafana.nrp-nautilus.io/d/r6lloPJmz/ceph-cluster)

## Currently available storageClasses:

| StorageClass | Filesystem Type | Region | AccessModes | Restrictions | Storage Type |
| --- | --- | --- | --- | --- | --- |
| rook-cephfs | CephFS | US West | ReadWriteMany |  | Spinning drives with NVME meta |
| rook-cephfs-central | CephFS | US Central | ReadWriteMany |  | Spinning drives with NVME meta |
| rook-cephfs-east | CephFS | US East | ReadWriteMany |  | Mixed |
| rook-cephfs-south-east | CephFS | US South East | ReadWriteMany |  | Spinning drives with NVME meta |
| rook-cephfs-pacific | CephFS | Hawaii+Guam | ReadWriteMany |  | Spinning drives with NVME meta |
| rook-cephfs-haosu | CephFS | US West (local) | ReadWriteMany | Hao Su and Ravi cluster | Spinning drives with NVME, meta on NVME |
| rook-cephfs-tide | CephFS | US West (local) | ReadWriteMany | SDSU Tide cluster | Spinning drives with NVME meta |
| rook-cephfs-fullerton | CephFS | US West (local) | ReadWriteMany |  | Spinning drives with NVME meta |
| rook-cephfs-ucsd | CephFS | US West (local) | ReadWriteMany | [Read the Policy](#ucsd-nvme-cephfs-filesystem-policy) | NVME |
| rook-ceph-block (\*default\*) | RBD | US West | ReadWriteOnce |  | Spinning drives with NVME meta |
| rook-ceph-block-east | RBD | US East | ReadWriteOnce |  | Mixed |
| rook-ceph-block-south-east | RBD | US South East | ReadWriteOnce |  | Spinning drives with NVME meta |
| rook-ceph-block-pacific | RBD | Hawaii+Guam | ReadWriteOnce |  | Spinning drives with NVME meta |
| rook-ceph-block-tide | RBD | US West (local) | ReadWriteOnce | SDSU Tide cluster | Spinning drives with NVME meta |
| rook-ceph-block-fullerton | RBD | US West (local) | ReadWriteOnce |  | Spinning drives with NVME meta |
| rook-ceph-block-central | RBD | US Central | ReadWriteOnce |  | Spinning drives with NVME meta |

Ceph shared filesystem ([**CephFS**](https://docs.ceph.com/en/latest/cephfs/)) is the primary way of storing data in nautilus and allows mounting same volumes from multiple PODs in parallel (*ReadWriteMany*).

Ceph block storage allows [**RBD** (Rados Block Devices)](https://docs.ceph.com/docs/master/rbd/) to be attached to a **single pod** at a time (*ReadWriteOnce*). Provides fastest access to the data, and **is preferred for all datasets not needing shared access from multiple pods**.

## UCSD NVMe CephFS filesystem policy

Caution

**This policy applies to the `rook-cephfs-ucsd` storageclass**

The filesystem is very fast and small. We expect all data on it to be used for currently running computation and then promptly deleted. We reserve the right to purge any data that’s staying there longer than needed at admin’s discretion without any notifications.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/storage/ceph.md)

[Previous  
Intro](/documentation/userdocs/storage/intro)  [Next  
Ceph S3](/documentation/userdocs/storage/ceph-s3)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.