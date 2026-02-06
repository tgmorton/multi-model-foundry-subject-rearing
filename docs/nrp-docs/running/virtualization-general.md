# Virtualization- General

**Source:** https://nrp.ai/documentation/userdocs/running/virtualization-general

# Virtualization- General

## Storage

Virtual machines are more sensitive to the storage speed than regular container, making it important to choose the right storage type and region for your VM image.

[Linstor](/documentation/userdocs/storage/linstor/) provides the fastest available storage option for VMs.

It’s also important to minimize the latency to the storage. We currently have Linstor available in San Diego and UNL area. So please choose the region closes to your intendent computation place, and position your VMs as close to the storage as possible.

## Using GPUs in VMs

Nodes labeled with `nautilus.io/vfio` are configured to passthrough GPUs for kubevirt VMs.

To see the list of such nodes, run `kubectl get nodes -l nautilus.io/vfio=true`

You can check the types of exposed GPUs in each node with `kubectl describe node <node-name>`, `Capacity` section will have a number of `nvidia.com/...` gpus available.

To use the GPU in VM, add it to the `gpus` section of your VM definition with the appropriate type:

```
domain:

devices:

gpus:

- deviceName: nvidia.com/GA102GL_A10

name: gpu1
```

You can proceed with running our examples for [Windows](/documentation/userdocs/running/virtualization-windows/) and [Ubuntu linux](/documentation/userdocs/running/virtualization-ubuntu/)

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/virtualization-general.md)

[Previous  
Postgres cluster](/documentation/userdocs/running/postgres)  [Next  
Live ubuntu](/documentation/userdocs/running/virtualization-ubuntu)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.