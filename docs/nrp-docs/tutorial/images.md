# Docker Images

**Source:** https://nrp.ai/documentation/userdocs/tutorial/images

# Docker Images

# **Images**

Images are fundamental components in Nautilus as they package the application code, dependencies, and runtime environment required to run containers within Kubernetes clusters. Images serve as the building blocks for containerized applications, allowing developers to deploy their software in a consistent and portable manner.

## Learning Objectives

1. By reviewing these materials, you will have a basic understanding of how to locate standard software packages to run on Nautilus.
2. You will have a basic understanding of how to modify those packages using Docker.

## Prerequisites

In order to complete this tutorial, you should have the following:

1. An account on [Docker Hub](https://hub.docker.com/signup).
2. [Docker Desktop](https://www.docker.com/get-started/) installed on your local machine.
3. An account on [Nautilus GitLab repository](https://gitlab.nrp-nautilus.io/users/sign_up).
4. You should be familiar with the concept of software container images, Kubernetes and have completed the basic Kubernetes tutorials.

## Finding container images and building your own

Multiple pre-built publicly available container images already exist on [Docker Hub](https://hub.docker.com). If there’s one that fits your needs, you can use one directly from there. You will need to sign up for an account, and it would be helpful for you to install Docker Desktop on your local machine.

If there’s an image that you want to use, but also need a couple more libraries added to it, you can extend the image by building your own based on remote one.

First, create an account in our [GitLab](https://gitlab.nrp-nautilus.io) instance to simplify building and storing the containers.

## Extending an existing image

Find the image you need. Let’s say we want to extend the [python:3](https://hub.docker.com/_/python) image by adding a jupyter package to it, and also installing the vim text editor.

Follow the [GitLab development guide](/documentation/userdocs/development/gitlab/) (make it public to simplify accessing your image), and also add this `Dockerfile` to the project before committing your changes:

```
FROM python:3

RUN pip install jupyter && \

apt-get update && \

apt-get install -y vim
```

Use the [basic section](/documentation/userdocs/tutorial/basic) to run the pod with your new image.

Exec into the pod and check jupyter and vim are installed:

```
root@test-pod:/# vim -h | head -n 1

VIM - Vi IMproved 8.1 (2018 May 18, compiled Jun 15 2019 16:41:15)

root@test-pod:/# jupyter

usage: jupyter [-h] [--version] [--config-dir] [--data-dir] [--runtime-dir] [--paths] [--json] [subcommand]

jupyter: error: one of the arguments --version subcommand --config-dir --data-dir --runtime-dir --paths is required
```

## Changing an existing image

Sometimes you see an image which almost work for you, but you’d like to change versions of the software of the software installed in it. Some images provide the source of their Dockerfile, which is usually well written and you can use it as is, but modify the versions.

For example, we take the [mltshp/mltshp-web](http://hub.docker.com/r/mltshp/mltshp-web) image, and go to it’s [Dockerfile](https://hub.docker.com/r/mltshp/mltshp-web/dockerfile). Then we copy it’s contents and use the above steps to build our own image with needed changes.

## The end

Please make sure you deleted the project in `Settings -> General -> Advanced` when done.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/tutorial/images.md)

[Previous  
Batch Jobs](/documentation/userdocs/tutorial/jobs)  [Next  
Storage](/documentation/userdocs/tutorial/storage)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.