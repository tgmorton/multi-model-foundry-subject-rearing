# Virtualization - Windows

**Source:** https://nrp.ai/documentation/userdocs/running/virtualization-windows

# Virtualization - Windows

[2022 guide windows 11](https://kubevirt.io/2022/KubeVirt-installing_Microsoft_Windows_11_from_an_iso.html)

## Creating a Windows 11 VM install image from existing image

First step is to download the Windows 11 ISO from <https://www.microsoft.com/en-us/software-download/windows11>.

Then, you need to follow [these steps](/documentation/userdocs/storage/ceph-s3#host-a-file-for-public-download) to host your Windows 11 ISO file in an S3 Bucket.

After you have your ISO file download link ready, create this DataVolume:

```
apiVersion: cdi.kubevirt.io/v1beta1

kind: DataVolume

metadata:

name: windev

spec:

source:

http:

url: "<YOUR_WINDOWS_ISO_DOWNLOAD_LINK>"

pvc:

volumeMode: Filesystem

accessModes:

- ReadWriteOnce

resources:

requests:

storage: 10Gi

storageClassName: linstor-unl
```

After running `kubectl apply -f datavolume.yaml`, this will create a new PVC.

Once it’s ready:

```
➜ kubectl get datavolume windev

NAME   PHASE       PROGRESS   RESTARTS   AGE

windev Succeeded   100.0%                3d19h
```

… please make sure that you delete your file and/or bucket. Then, create a drive for the VM:

```
apiVersion: v1

kind: PersistentVolumeClaim

metadata:

name: winhd

spec:

accessModes:

- ReadWriteOnce

resources:

requests:

storage: 50Gi

storageClassName: linstor-unl
```

… and a VM with both:

```
apiVersion: kubevirt.io/v1

kind: VirtualMachine

metadata:

name: winvm

spec:

running: false

template:

metadata:

labels:

kubevirt.io/size: small

kubevirt.io/domain: winvm

spec:

domain:

cpu:

cores: 8

devices:

disks:

- bootOrder: 2

cdrom:

bus: sata

name: cdromiso

- disk:

bus: sata

bootOrder: 1

name: harddrive

- cdrom:

bus: sata

name: virtiocontainerdisk

interfaces:

- name: default

model: e1000

masquerade: {}

inputs:

- bus: usb

name: tablet1

type: tablet

sound:

name: sound

tpm: {}

resources:

requests:

cpu: 8

memory: 8Gi

limits:

cpu: 8

memory: 8Gi

clock:

timer:

hpet:

present: false

hyperv: {}

pit:

tickPolicy: delay

rtc:

tickPolicy: catchup

utc: {}

features:

acpi: {}

apic: {}

hyperv:

relaxed: {}

spinlocks:

spinlocks: 8191

vapic: {}

smm: {}

firmware:

bootloader:

efi:

secureBoot: true

uuid: 5d307ca9-b3ef-428c-8861-06e72d69f223

networks:

- name: default

pod: {}

volumes:

- name: cdromiso

dataVolume:

name: windev

- name: harddrive

persistentVolumeClaim:

claimName: winhd

- name: virtiocontainerdisk

containerDisk:

image: kubevirt/virtio-container-disk
```

Note

The usb tablet input and sound device are required for proper audio and mouse tracking

Now, install virtctl: <https://github.com/kubevirt/kubevirt/releases/tag/v0.42.1>

Then start it:

`virtctl start winvm`

After that you should be able to connect to it using vnc: `virtctl vnc winvm` after instaling one of VNC apps on mac:

<https://github.com/kubevirt/kubevirt/blob/master/pkg/virtctl/vnc/vnc.go#L223>

<https://kubevirt.io/user-guide/#/usage/graphical-and-console-access>

## Our main example

```
apiVersion: kubevirt.io/v1

kind: VirtualMachine

metadata:

name: winvm

spec:

running: false

template:

metadata:

labels:

kubevirt.io/size: small

kubevirt.io/domain: winvm

spec:

affinity:

nodeAffinity:

requiredDuringSchedulingIgnoredDuringExecution:

nodeSelectorTerms:

- matchExpressions:

- key: kubernetes.io/hostname

operator: In

values:

- hcc-nrp-shor-c5830.unl.edu

domain:

cpu:

cores: 16

devices:

gpus:

- deviceName: "nvidia.com/A40_Tesla_A40"

name: gpu1

disks:

- bootOrder: 2

cdrom:

bus: sata

name: cdromiso

- disk:

bus: sata

bootOrder: 1

name: harddrive

- cdrom:

bus: sata

name: virtiocontainerdisk

interfaces:

- name: default

model: e1000

masquerade: {}

inputs:

- bus: usb

name: tablet1

type: tablet

sound:

name: sound

tpm: {}

resources:

requests:

cpu: 16

memory: 16Gi

limits:

cpu: 16

memory: 16Gi

clock:

timer:

hpet:

present: false

hyperv: {}

pit:

tickPolicy: delay

rtc:

tickPolicy: catchup

utc: {}

features:

acpi: {}

apic: {}

hyperv:

relaxed: {}

spinlocks:

spinlocks: 8191

vapic: {}

smm: {}

firmware:

bootloader:

efi:

secureBoot: true

uuid: 5d307ca9-b3ef-428c-8861-06e72d69f223

networks:

- name: default

pod: {}

volumes:

- name: cdromiso

dataVolume:

name: windev

- name: harddrive

persistentVolumeClaim:

claimName: winhd

- name: virtiocontainerdisk

containerDisk:

image: kubevirt/virtio-container-disk
```

Caution

If you want to request a GPU with KubeVirt, please ensure there are available GPUs specifically designated for KubeVirt. **Note: KubeVirt GPUs aren’t managed by Kubernetes (k8s) in the standard way.**

To check for available GPUs, you can run the following commands:

Terminal window

```
$ kubectl get nodes -l nautilus.io/vfio=true

NAME                           STATUS   ROLES    AGE    VERSION

<node-name>                    Ready    <none>   <age>  <version>
```

Then, describe the node to verify GPU availability:

Terminal window

```
$ kubectl describe node <node-name> | grep -i nvidia.com/

Taints:             nvidia.com/gpu=Exists:PreferNoSchedule

nvidia.com/GA102_GEFORCE_RTX_3090:  4

nvidia.com/gpu:                     0

nvidia.com/GA102_GEFORCE_RTX_3090:  4

nvidia.com/gpu:                     0

nvidia.com/GA102_GEFORCE_RTX_3090  4                   4

nvidia.com/gpu                     0                   0
```

In the node description, look for GPU availability under a specific name that aligns with KubeVirt’s allocation method, rather than `nvidia.com/gpu`.

#### Adding virtvnc

In your namespace create a deployment:

```
apiVersion: apps/v1

kind: Deployment

metadata:

name: virtvnc

spec:

replicas: 1

revisionHistoryLimit: 1

selector:

matchLabels:

app: virtvnc

template:

metadata:

labels:

app: virtvnc

spec:

containers:

- env:

- name: NAMESPACE

valueFrom:

fieldRef:

apiVersion: v1

fieldPath: metadata.namespace

image: gitlab-registry.nrp-nautilus.io/prp/virtvnc

imagePullPolicy: Always

livenessProbe:

failureThreshold: 30

httpGet:

path: /

port: 8001

scheme: HTTP

initialDelaySeconds: 30

periodSeconds: 10

successThreshold: 1

timeoutSeconds: 5

name: virtvnc

resources:

requests:

cpu: 1m

memory: 100Mi

limits:

cpu: 2m

memory: 200Mi

serviceAccountName: virtvnc
```

Service account and roles:

```
apiVersion: v1

kind: ServiceAccount

metadata:

name: virtvnc

---

apiVersion: rbac.authorization.k8s.io/v1

kind: Role

metadata:

name: virtvnc

rules:

- apiGroups:

- subresources.kubevirt.io

resources:

- virtualmachineinstances/console

- virtualmachineinstances/vnc

verbs:

- get

- apiGroups:

- kubevirt.io

resources:

- virtualmachines

- virtualmachineinstances

- virtualmachineinstancepresets

- virtualmachineinstancereplicasets

- virtualmachineinstancemigrations

verbs:

- get

- list

- watch

---

apiVersion: rbac.authorization.k8s.io/v1

kind: RoleBinding

metadata:

name: virtvnc

roleRef:

apiGroup: rbac.authorization.k8s.io

kind: Role

name: virtvnc

subjects:

- kind: ServiceAccount

name: virtvnc
```

Service and ingress:

```
apiVersion: networking.k8s.io/v1

kind: Ingress

metadata:

annotations:

ingress.kubernetes.io/auth-secret: virtvnc-login

ingress.kubernetes.io/auth-type: basic

ingress.kubernetes.io/config-backend: |

http-request del-header Authorization

name: virtvnc

spec:

ingressClassName: haproxy

rules:

- host: <some_name>.nrp-nautilus.io

http:

paths:

- backend:

service:

name: virtvnc

port:

number: 8001

path: /

pathType: ImplementationSpecific

tls:

- hosts:

- <some_name>.nrp-nautilus.io

---

apiVersion: v1

kind: Service

metadata:

labels:

app: virtvnc

name: virtvnc

spec:

ports:

- port: 8001

protocol: TCP

targetPort: 8001

selector:

app: virtvnc

type: ClusterIP
```

Secret to hold the password for your virtvnc:

Terminal window

```
kubectl create secret generic virtvnc-login -n <namespace> --from-literal=auth=<my_login>::<my_password>
```

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/virtualization-windows.md)

[Previous  
Live ubuntu](/documentation/userdocs/running/virtualization-ubuntu)  [Next  
Ray Cluster](/documentation/userdocs/running/ray-cluster)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.