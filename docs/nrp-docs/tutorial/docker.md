# Docker and Kubernetes

**Source:** https://nrp.ai/documentation/userdocs/tutorial/docker

# Docker and Kubernetes

# Docker and Nautilus

## Introduction

Learning to use Docker for Nautilus enables you to take advantage of containerization technology to streamline the deployment, management, and scalability of your applications on the cluster. It also aligns with modern DevOps practices and helps you leverage the full capabilities of the Nautilus infrastructure.

## Prerequisites

You will need the ability to install applications on your computer running a supported operating system (OS), such as Windows, Mac OS X or Linux (Chromebooks are not supported). You should also be familiar with the basic concepts of software containers (images) and [Docker Hub](https://hub.docker.com).

#### Installing docker

Docker container image is an easy way to package some files together and run as an isolated environment on virtually any computer in the world - from your own laptop to a supercomputer.

The most popular, but not the only one, tool to run containers is [Docker](https://www.docker.com). There is a [nice guide](https://www.docker.com/get-started) on installing docker on your local machine and [getting started with it](https://www.docker.com/101-tutorial).

#### Building image

There are millions of already existing containers available for free on [docker hub](https://hub.docker.com) (use the search field to search for the software you need). In case you want something that doesn’t exist there, or want to extend the already existing container, it’s easy to do so.

To build a new container, you first need to create a Dockerfile. Let’s say we want a container which has python3 in it, and also has the geojson library installed to parse some geojson files and the vim text editor.

First we search by keyword: <https://hub.docker.com/search?q=python&type=image>. Usually the most popular images appear on top. The [python repository](https://hub.docker.com/_/python) looks good to us, and we go to see which tags are available from the repository.

*Container tag* is a flavor of a container - it can tell us which version of the software it contains, which variant of software setup is used if there are many, etc. The default tag is `latest` if you don’t specify any, but it’s advised to specify at least the version so that your workflow is not suddenly broken by `:latest` image version update. The tag is specified after the container name, f.e. `python:3.9`.

Let’s say we browsed the [python tags](https://hub.docker.com/_/python?tab=tags) page and found that we want to base our own image on the [3.9 version](https://hub.docker.com/layers/python/library/python/3.9/images/sha256-bdc5af48b5f7bf9fe141057d17dbdfbfadc1c348dbb7292f8856d993bd66af63?context=explore).

On the specific tag page we can see the Dockerfile commands which were used to build the image (not for all images). We could copy the commands and modify those to our own needs, but instead we might simply derive our container from the one already existing. To do that we start our Dockerfile with `FROM` directive:

```
FROM python:3.9

RUN apt-get update && \

apt-get install -y vim && \

pip install geojson
```

The reason to put 3 comands in a single RUN is that we want those to be in a single container *layer*, which will speed up downloads and minimise the image size. Here’s the [best practices guide](https://www.docker.com/blog/intro-guide-to-dockerfile-best-practices/).

Save this as `Dockerfile` and run the command (in case you have docker installed locally):

```
docker build . -t my-first-container:latest
```

You will see docker pulling the existing container *images* from docker hub, and then executing the RUN directive you added to Dockerfile. Each such directive creates a separate *layer* in the image, and layers are cached - this way you can have multiple python containers, and they all will share the base layers.

Now type `docker images` and see that your image `my-first-container` was created and is saved in your local docker *repository*.

#### Running image

Now we can actually run the image. A single image can be ran multiple times, and all changes to the files will only be visible while the container is running once it stops, all new data **will be lost** unless it’s placed in a [persistent volume](https://docs.docker.com/storage/volumes/).

Run:

```
docker run -it --rm -v $PWD:/code my-first-container bash
```

This will start an instance of our container locally on your machine, and also the `-v $PWD:/code` mounted the current directory into /code inside the container. You can then execute `cd /code` inside the container and run some python code. Also note that geojson is installed:

```
root@19d104e34ca6:/# pip list

Package    Version

---------- -------

geojson    2.5.0

pip        21.0.1

setuptools 53.0.0

wheel      0.36.2
```

To exit, execute `exit`. `--rm` told docker to kill and remove the container once we exist from it. You can run in detached mode by running instead:

```
docker run -d -v $PWD:/code my-first-container sleep infinity
```

This will start the container, but not exec into it. To exec, run `docker ps` to see running containers (not either the Container ID or the Name of your new container), and then exec into it:

```
docker exec -it <name_or_id> bash
```

This will execute bash inside the container. YOu can enter and exit from it multiple times. To destroy, run:

```
docker rm -f <name_or_id>
```

#### Exporting image

For now the container image we’ve built is stored in our local docker registry. To make it available to the kubernetes cluster we need to put it somewhere globally available. You can register on docker hub and [upload there](https://docs.docker.com/docker-hub/), or use [our GitLab instance](/documentation/userdocs/development/gitlab/) and create a repository. Our GitLab is free, has unlimited space and is much faster than Docker Hub. Also for larger images it’s possible to build those directly in our cluster and not on your local computer, which will significantly increase the speed and upload time. But you’re free to upload the images from your local machine if you wish.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/tutorial/docker.md)

[Previous  
Introduction](/documentation/userdocs/tutorial/introduction)  [Next  
Basic Kubernetes](/documentation/userdocs/tutorial/basic)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.