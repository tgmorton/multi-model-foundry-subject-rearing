# Storage

**Source:** https://nrp.ai/documentation/userdocs/tutorial/storage

# Storage

# Storage

In Kubernetes, a Persistent Volume Claim (PVC) is a resource that allows a user to request storage from a storage class defined in the cluster. StorageClasses enable the cluster to abstract the details of storage provisioning and management, allowing users to request storage without needing to know the specifics of the underlying infrastructure.

## Prerequisites

This section builds on skills from the tutorial on [Basic Kubernetes](/documentation/userdocs/tutorial/basic). You will need a basic understanding of Kubernetes concepts such as Pods, Persistent Volume Claims (PVCs), and Persistent Volumes (PVs).

## Learning objectives

1. You will have a basic understanding of storage types in Kubernetes.
2. By completing this lesson, you will understand how to request a persistent volume claim (PVC).
3. You will understand how to connect your PVC to another pod and make it available to your software container.

## Storage Types

In a Kubernetes cluster, there are several types of storage options available to manage data persistence for applications and services:

* **Local Storage**: Kubernetes allows pods to use storage directly attached to the node where they are scheduled. This is typically in the form of local disks. While local storage is fast, it is not portable across nodes and will be subject to data loss if the node fails.
* **Persistent Volumes** (PV): PVs are cluster-wide storage resources provisioned by the (cluster) administrator. They are not bound to any particular pod, and pods can claim them using Persistent Volume Claims (PVCs). In Nautilus, PVs are created dynamically when a Persistent Volume Claim is made.
* **Persistent Volume Claim** (PVC): PVCs are requests for storage by applications. They are used by developers to request specific storage resources (size, access mode, etc.) without needing to know the underlying storage implementation.
* **Object Storage**: Kubernetes can also integrate with object storage systems like Amazon S3, Google Cloud Storage, and others using plugins or external solutions like MinIO. These provide scalable and durable storage for various types of data.

There are other types of storage options in other Kubernetes clusters, but they are not implemented in Nautilus.

## Create an emptyDir

In Kubernetes, an `emptyDir` is a type of volume that is initially empty and created when a Pod is assigned to a node. It’s intended to be used as temporary storage within a pod. An `emptyDir` volume exists as long as the Pod that uses it is running on a node. When the Pod is removed from the node for any reason, the data in the emptyDir is deleted permanently.

Let’s explore the `emptyDir`by creating a simple example.

You can copy-and-paste the lines below into a new file called `strg1.yaml`.

###### strg1.yaml:

```
apiVersion: apps/v1

kind: Deployment

metadata:

name: test-storage

labels:

k8s-app: test-storage

spec:

replicas: 1

selector:

matchLabels:

k8s-app: test-storage

template:

metadata:

labels:

k8s-app: test-storage

spec:

containers:

- name: mypod

image: alpine

resources:

limits:

memory: 100Mi

cpu: 100m

requests:

memory: 100Mi

cpu: 100m

command: ["sh", "-c", "apk add dumb-init && dumb-init -- sleep 100000"]

volumeMounts:

- name: mydata

mountPath: /mnt/myscratch

volumes:

- name: mydata

emptyDir: {}
```

:question: Examine the sample `yaml`code above. Why do you think we specified a “deployment” instead of a simple pod? Are there other things that are different about this deployment? ❗ Hint: examine the `image` we are using? :question: What image is it? Would we expect this image to behave like other images?

### Start the deployment

Just like other processes, we can start our deployment by using the `create` command and specifying a file, like so:

`kubectl create -f strg1.yaml`

Now, try logging into the Pod created by the deployment. You will have to discover its name by querying the cluster with `kubectl get pods -o wide`.

:question: Were you able to log in?

If you used the command:

`kubectl exec -it test-storage-<hash> -- /bin/bash`

:question: What was the outcome? :question: Were you able to log into the pod?

❗ In fact, the `yaml` file is using the Linux distribution known as “Alpine”.

The Alpine Linux distribution is a lightweight and security-oriented Linux distribution commonly used in containerized environments, including Kubernetes. Alpine Linux is known for its minimalistic design, small footprint, and focus on security. It provides a simple and efficient base for containerized applications, offering a smaller attack surface and reduced resource usage compared to other Linux distributions.

But instead of the full-featured `bash`shell (aka Command Line Interpreter or CLI), Alpine uses a lightweight version called `ash` (short for Almquist Shell).

It aims to provide essential shell functionalities while keeping its codebase small and efficient. It lacks some of the advanced features found in Bash but offers POSIX compliance and basic scripting capabilities.

For Alpine, we need to use a different interpreter, so use this command instead:

`kubectl exec -it test-storage-<hash> -- /bin/ash`

to log into the Pod.

Once you are inside the Pod, you can create a directory, such as

`mkdir /mnt/myscratch/<username>`

then store some files in it (hint: you can create them on the fly, using the `cat` command to redirect the standard input).

Also put some files in some other (unrelated) directories, if you wish.

Now, while still logged into the Pod, kill the container using the command `kill 1`. Since this is a deployment, we’d expect the container to respawn, so we can just wait for a new one to be created, then log back in.

:question: What happened to files?

You can now delete the deployment.

## Creating a Persistent Volume Claim

In addition to the computing cluster we’ve been exploring, Nautilus also has a distributed storage system (a Ceph Storage Cluster). This storage cluster provides persistent storage for Nautilus.

Integrating a Ceph storage cluster with a Kubernetes cluster offers several advantages for managing storage in containerized environments, such as scaleability, high availability, fault tolerance, dynamic provisioning, performance and data mobility for containerized applications.

To get storage, we need to create an abstraction called `PersistentVolumeClaim`. By doing so, we “claim” some storage space and a “Persistent Volume” is created dynamically. PVCs are scoped to a particular namespace in Kubernetes. This means that PVCs created within one namespace are not directly accessible or visible to other namespaces by default.

Create the file:

###### pvc.yaml:

```
apiVersion: v1

kind: PersistentVolumeClaim

metadata:

name: test-vol

spec:

storageClassName: rook-ceph-block

accessModes:

- ReadWriteOnce

resources:

requests:

storage: 1Gi
```

We’re creating a 1GB volume and formatting it with XFS.

Look at it’s status with `kubectl get pvc -o wide`. The `STATUS` field should be equals to `Bound` - this indicates successful allocation.

Now we can attach it to our pod. Create one called pvc-pod.yaml:

```
apiVersion: v1

kind: Pod

metadata:

name: test-pod

spec:

containers:

- name: mypod

image: ubuntu:latest

command: ["sh", "-c", "sleep infinity"]

resources:

limits:

memory: 100Mi

cpu: 100m

requests:

memory: 100Mi

cpu: 100m

volumeMounts:

- mountPath: /examplevol

name: examplevol

volumes:

- name: examplevol

persistentVolumeClaim:

claimName: test-vol
```

In volumes section we’re attaching the requested persistent volume to the pod (by its name!), and in volumeMounts we’re mounting the attached volume to the container in specified folder.

## Exploring storageClasses

Attaching persistent storage is usually done based on storage class. You can explore the different storage classes by reading the [documentation](/documentation/userdocs/storage/intro). Not all storage classes are available to you as a User or even as a namespace administrator.

Note that the one we used is the default - it will be used if you define none.

❗ Not all Linux distributions share the same functionalities, though many of the basics may be the same (e.g. POSIX-compliance). It’s important to choose the right distribution based on your needs. It’s also important to remember to balance your requests against your actual needs, keeping in mind that optimizing system resource requirements is important when large numbers of tasks are executed concurrently. Minimizing system resource requirements improves scalability, speed, reliability and stability.

❗ Remember that you can choose compute nodes location closer to your preferred storage class as described in the [scheduling tutorial](/documentation/userdocs/tutorial/scheduling/#using-geographical-topology).

## Cleaning up

After you’ve deleted all the pods and deployments, delete the volume claim:

`kubectl delete pvc test-vol`

Please make sure you did not leave any running pods, deployments, volumes.

## End

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/tutorial/storage.md)

[Previous  
Images](/documentation/userdocs/tutorial/images)  [Next  
Debugging](/documentation/userdocs/tutorial/debugging)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.