# Virtualization - Ubuntu

**Source:** https://nrp.ai/documentation/userdocs/running/virtualization-ubuntu

# Virtualization - Ubuntu

Guide: <https://kubevirt.io/2020/KubeVirt-installing_Microsoft_Windows_from_an_iso.html>

## Running the ubuntu live vm

Here’s the working example of an ubuntu VM with cloud-init, added SSH key and emptyDir scratch disk:

```
apiVersion: kubevirt.io/v1

kind: VirtualMachine

metadata:

name: myvm

spec:

running: false

template:

spec:

accessCredentials:

- sshPublicKey:

propagationMethod:

configDrive: {}

source:

secret:

secretName: pub-keys

architecture: amd64

domain:

cpu:

cores: 8

devices:

autoattachGraphicsDevice: true

autoattachSerialConsole: true

disks:

- disk:

bus: virtio

name: harddrive

- disk:

bus: virtio

name: virtiocontainerdisk

bootOrder: 1

- disk:

bus: virtio

name: cloudinit

machine:

type: q35

resources:

limits:

cpu: 800m

memory: 8Gi

requests:

cpu: 800m

memory: 8Gi

volumes:

- containerDisk:

image: quay.io/containerdisks/ubuntu:22.04

name: virtiocontainerdisk

- emptyDisk:

capacity: 16Gi

name: harddrive

- cloudInitConfigDrive:

userData: |-

#cloud-config

disk_setup:

/dev/vda:

table_type: gpt

layout: True

overwrite: True

fs_setup:

- device: /dev/vda

partition: 1

filesystem: xfs

mounts:

- [ vda, /opt/data ]

name: cloudinit
```

You also need to create the secret with your SSH key to login with:

```
apiVersion: v1

data:

key1: <base64-encoded key>

kind: Secret

metadata:

name: pub-keys

type: Opaque
```

Use [virtctl](https://kubevirt.io/user-guide/operations/virtctl_client_tool/) to start the VM:

Terminal window

```
virtctl start myvm
```

Open VNC or SSH:

Terminal window

```
virtctl vnc myvm

virtctl ssh ubuntu@myvm
```

## Example of an ubuntu vm with large local disk on linstor

```
apiVersion: kubevirt.io/v1

kind: VirtualMachine

metadata:

name: ubuntu-vm

spec:

dataVolumeTemplates:

- metadata:

name: ubuntu-2204

spec:

source:

registry:

url: "docker://quay.io/containerdisks/ubuntu:22.04"

pvc:

accessModes:

- ReadWriteOnce

resources:

requests:

storage: 150Gi

storageClassName: linstor-igrok

running: true

template:

metadata:

creationTimestamp: null

spec:

accessCredentials:

- sshPublicKey:

source:

secret:

secretName: pub-keys

affinity:

nodeAffinity:

requiredDuringSchedulingIgnoredDuringExecution:

nodeSelectorTerms:

- matchExpressions:

- key: topology.kubernetes.io/region

operator: In

values:

- us-west

architecture: amd64

domain:

cpu:

cores: 8

devices:

autoattachGraphicsDevice: true

autoattachSerialConsole: true

disks:

- bootOrder: 1

disk:

bus: virtio

name: datavolumedisk1

machine:

type: q35

resources:

limits:

ephemeral-storage: 100Gi

memory: 8Gi

cpu: 8

requests:

ephemeral-storage: 100Gi

memory: 8Gi

cpu: 8

volumes:

- dataVolume:

name: ubuntu-2204

name: datavolumedisk1

- cloudInitConfigDrive:

userData: |-

#cloud-config

name: cloudinit
```

## Fixing the network issue

If you hit [this problem](https://github.com/kubevirt/kubevirt/issues/9993), add static MAC address to your VM:

```
networks:

- name: default

pod: {}

domain:

devices:

interfaces:

- name: default

masquerade: {}

macAddress: "02:00:00:00:00:02"
```

## Using linux GUI

For better mouse tracking in VNC, add the tablet device:

```
domain:

devices:

inputs:

- bus: usb

name: tablet1

type: tablet
```

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/virtualization-ubuntu.md)

[Previous  
General](/documentation/userdocs/running/virtualization-general)  [Next  
Windows](/documentation/userdocs/running/virtualization-windows)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.