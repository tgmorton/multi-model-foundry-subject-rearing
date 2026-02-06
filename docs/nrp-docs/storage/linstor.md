# Linstor

**Source:** https://nrp.ai/documentation/userdocs/storage/linstor

# Linstor

[Linstor](https://linbit.com/linstor/) is currently the fastest distributed block storage in the cluster, and can be used for tasks requiring minimal latency, such as VM images, docker build space, databases, etc. Also it doesn’t lock the volumes like Ceph does, making it a good option for critical highly available storage volumes.

It uses the [DRBD](https://linbit.com/drbd/) kernel module that handles the replication, and provides nearly native drive performance for I/O operations.

Note

* Linstor allocates the space for the requested amount of storage. You should only request the space as needed and extend when necessary.
* `linstor-ha` provides the highest level of redundancy and reliability at a cost of higher replication factor, and can only be used for smaller most critical use cases. (jupyter hub volume is a good example)

### Linstor storage pools data use

Credit: [Linstor data use](https://observablehq.com/d/4d813a19acd33267)

[Linstor Grafana dashboard](https://grafana.nrp-nautilus.io/d/f_tZtVlMz/linstor-drbd)

### Currently available Storage Classes:

| StorageClass | Region | AccessModes | Storage Type | Replication factor |
| --- | --- | --- | --- | --- |
| linstor-ha | 3 replicas spread through different regions | ReadWriteOnce | Mixed | 3x |
| linstor-unl | US Central / UNL | ReadWriteOnce | Spinning drives RAID 10 | 1x |
| linstor-sdsu | US West / SDSU | ReadWriteOnce | NVME | 2x |
| linstor-igrok | US West / UCSD | ReadWriteOnce | NVME | 2x |
| linstor-ucsc | US West / UCSC | ReadWriteOnce | SSDs RAID 10 | 1x |

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/storage/linstor.md)

[Previous  
Local scratch](/documentation/userdocs/storage/local)  [Next  
Nextcloud](/documentation/userdocs/storage/nextcloud)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.