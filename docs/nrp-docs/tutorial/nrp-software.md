# Container images on CVMFS

**Source:** https://nrp.ai/documentation/userdocs/tutorial/nrp-software

# Container images on CVMFS

[CVMFS](https://cernvm.cern.ch/fs/) is a distributed filesystem that allows you to mount software repositories and datasets on your device. NRP hosts an application repository `/cvmfs/nrp-software.opensciencegrid.org` to give users access to a reproducible, ready-to-use environment to share and use data through CVMFS. In this tutorial, we’ll go through steps distributing and using container images to the CVMFS repo hosted by NRP.

**Access to container images distributed to CVMFS is available globally. For example, the CVMFS data repository is also availabale on the [OSG OSPool](https://osg-htc.org/services/ospool/), not just limited to the NRP environment. And other repositories in CVMFS can also be accessed from NRP environment.**

## Learning Objectives

1. How to create your own data images.
2. How to distribute images into CVMFS.
3. How to use the images distributed in CVMFS

## Prerequisites

In order to complete this tutorial, you should have gone through the [Quickstart](/documentation/userdocs/start/getting-started/), and finished these tutorials:

1. [Basic Kubernetes](/documentation/userdocs/tutorial/basic)
2. [Storage](/documentation/userdocs/tutorial/storage)
3. [Docker Images](/documentation/userdocs/tutorial/images)

You will also need knowlege regarding [Creating a pull request from a fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork) on GitHub.

## Creating data container images

Here we focus on how to create a customized data container image. We suppose that you have setup your enviroment according to the [Docker Images](/documentation/userdocs/tutorial/images) tutorial.

### 1. Put installation instructions in a `Dockerfile`

A `Dockerfile` is a plain text file with keywords and commands that can be used to create a new container image. Here is an example to build an image based on [python:3](https://hub.docker.com/_/python) with the `jupyter` package, and a data file called `data.txt`.

```
FROM python:3

RUN pip install jupyter

ADD data.txt /data.txt
```

* `FROM`, indicates which container image we’re starting with. We use [python:3](https://hub.docker.com/_/python) as the base image.
* `RUN`, indicates installation commands we want to run while building the image. Here we use `pip` to install the `jupyter` package.
* `ADD`, indicates the local or remote files or directories to be included in the image. Here we include a local file `data.txt` in the image.

### 2. Build the image

Run the following command to build the image in the same directory with the `Dockerfile` and `data.txt` files:

Terminal window

```
docker build . -t my-data-container:latest
```

### 3. Push the image to a container registry

In Step 2, we built a local image. If you have an account on [Docker Hub](https://docs.docker.com/docker-hub/), you can run the following commands to tag the image and push it there:

Terminal window

```
docker image tag my-data-container:latest DOCKER_USERNAME/my-data-container:latest

docker push DOCKER_USERNAME/my-data-container:latest
```

You may also use the GitLab container registry provided by NRP. For details, please refer to [Building in GitLab](/documentation/userdocs/development/gitlab)

## Distributing container images on CVMFS

Image distribution on CVMFS works with unpacked layers or image root file systems. We host a CVMFS repo to synchronize and distribute container images. Any image publically available in Docker and other registries can be included for automatic syncing into the CVMFS repository. The result is an unpacked image under `/cvmfs/nrp-software.opensciencegrid.org`.

To get your images included, please either create a git pull request against `images.txt` in the [cvmfs-oci](https://github.com/nrp-nautilus/cvmfs-oci) repository, or [contact NRP admins](/contact) and we can help you.

The `images.txt` file is a list of container images, one image per line, in the format of `[registry_host/][namespace/]repository_name[:tag]`. For examaple:

```
gitlab-registry.nrp-nautilus.io/nrp/scientific-images/python:latest
```

When this line is added to the image.txt and the pull request is merged, the image will be unpacked into CVMFS at `/cvmfs/gitlab-registry.nrp-nautilus.io/nrp/scientific-images/python:latest`.

If new versions of images have been pushed to the registry, they will be detected and updated in CVMFS accordingly.

## Accessing unpacked container images

### Creating a PVC volume with the spec `storageClassName: cvmfs`

To access images distributed in CVMFS, you need to attach the CVMFS volume which can mount all repos. First, create the PVC (taken from <https://github.com/cvmfs-contrib/cvmfs-csi/tree/master/example> ):

```
apiVersion: v1

kind: PersistentVolumeClaim

metadata:

name: cvmfs

spec:

accessModes:

- ReadOnlyMany

resources:

requests:

# Volume size value has no effect and is ignored

# by the driver, but must be non-zero.

storage: 1

storageClassName: cvmfs
```

### Using the `cvmfs` PVC in pods

#### Option 1, mount the entire CVMFS

Create a pod with the following yaml file:

```
apiVersion: v1

kind: Pod

metadata:

name: cvmfs-all-repos

spec:

containers:

- name: idle

image: busybox

imagePullPolicy: IfNotPresent

command: [ "/bin/sh", "-c", "trap : TERM INT; (while true; do sleep 1000; done) & wait" ]

volumeMounts:

- name: my-cvmfs

mountPath: /my-cvmfs

# CVMFS automount volumes must be mounted with HostToContainer mount propagation.

mountPropagation: HostToContainer

volumes:

- name: my-cvmfs

persistentVolumeClaim:

claimName: cvmfs
```

In this example, the `nrp-software.opensciencegrid.org` repo is accessible at `/my-cvmfs/nrp-software.opensciencegrid.org` in the pod. Notice that repo is mounted unless the mount point is accessed.

#### Option 2, mount the `nrp-software.opensciencegrid.org` repo specifically

If you need to mount the `nrp-software.opensciencegrid.org` repo specifically, add the `subPath` key to the pod’s `volumeMounts` section:

```
volumeMounts:

- name: my-cvmfs

# It is possible to mount a single CVMFS repository by specifying subPath.

subPath: nrp-software.opensciencegrid.org

mountPath: /my-nrp-software-cvmfs

mountPropagation: HostToContainer
```

In this example, the repo is accessible at `/my-nrp-software-cvmfs`, but other repos in CVMFS are not available in the pod.

### Mount the `cvmfs` PVC in customized JupyterHub deployment

If you have a [customized JupyterHub deployment](/documentation/userdocs/jupyter/jupyterhub), you can make CVMFS available in user spawned instances. Suppose a `cvmfs` PVC has been created in the namespace where the Jupyterhub is depoloyed, and you want to mount the CVMFS repo `nrp-software.openscience.org` at /nrp-software in every user’s pod, you can insert the following example into the JupyterHub’s value template (e.g. config.yaml):

```
singleuser:

storage:

extraVolumes:

- name: nrp-software

persistentVolumeClaim:

claimName: cvmfs

extraVolumeMounts:

- name: nrp-software

mountPath: /nrp-software

subPath: nrp-software.opensciencegrid.org

mountPropagation: HostToContainer
```

And then update the helm chart by command `helm upgrade --cleanup-on-fail --install jhub jupyterhub/jupyterhub --namespace <namespace> --version=<version> --values config.yaml`. When the JupyterHub is deployed, the images in `nrp-software.opensciencegrid.org` CVMFS repo is mounted as /nrp-software when users spawn new pods.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/tutorial/nrp-software.md)

[Previous  
Debugging](/documentation/userdocs/tutorial/debugging)  [Next  
MNIST Training (Presentation)](https://docs.google.com/presentation/d/1GMvaZr9Nm6LhYUU_E0E0LdoebPpk0dgb2Z6oS9v2Ww8/edit?usp=sharing)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.