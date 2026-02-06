# Introduction

**Source:** https://nrp.ai/documentation/userdocs/tutorial/introduction

# Introduction

In these tutorials you will learn the basic skills needed to access resources associated with the Nautilus Cluster. These resources can include GPU and CPU cycles, storage and even software. For novice users, we recommend that you step through all the tutorials, especially tutorials on launching pods, requesting a storage volume, and using Docker.

## What is Nautilus?

Nautilus is a distributed, hyper-converged cluster where storage, compute and networking are tightly integrated into a single unified system. What distinguishes Nautilus from other such systems is that Nautilus is distributed across dozens of locations. Individual Nautilus nodes are comprised of high-performance GPU servers combined with high-speed data transfer-and-storage nodes, and additional storage nodes that serve as data reservoirs…all stitched together with very fast networks. Unlike standalone GPU resources, jobs sent to Nautilus can run anywhere in this vast cluster, giving the user flexibility to run single tasks or up to many tasks across multiple nodes and locations while Kubernetes manages the workload.

## Why use Nautilus?

Nautilus has a huge variety of both standard GPUs (for example: NVIDIA RTX2080-Ti, RTX3090, RTX4090 and A10) and high-precision, high-memory GPUs (for example: NVIDIA A100 Tensor Core). It also has the ability to download large datasets very quickly and store them near any computation you might launch. Since Nautilus is a hyper-converged cluster which supports a high-degree of virtualization, the user does not need to worry about where their data reside or where their jobs will run. Furthermore, because software runs in containers, it is scaleable, repeatable and customizable without the hassle of system configuration dependencies. Software containers provide portability, efficiency, scaleability, consistency, speed, and security. Nautilus makes managing the data, software and hardware required for training of machine learning easier.

### But how much does it cost?

Perhaps most importantly, for individuals associated with institutions that participate in the InCommon identity and access management service, and who are conducting sponsored research in artificial intelligence or machine learning, Nautilus is FREE to use (subject to [policy restrictions](/documentation/userdocs/start/policies/) and fair use).

## Steps to Successful Nautilus Use

There are several key steps which you must go through to get access to Nautilus. They are:

1. Establish an account
2. Configure your local system to interact with Nautilus
3. Be promoted to User or Admin
4. Read and understand the policies
5. For beginners, complete the [Tutorials](./basic)

Caution

Familiarity with the Unix command line is the single biggest predictor of success with Nautilus. For most users who are unfamiliar with the Unix shell, this will prevent them from being able to use Nautilus, as it has no graphical user interface or desktop application. Users who do not understand the basic principles of Unix, lack the ability to navigate a Unix file system through a terminal window, do not understand Unix commands, have no experience with scripting or scripting languages, or have never written code before are highly discouraged from attempting to use Nautilus without first having a basic grasp of these skills.

## What if I don’t think I have time to learn a bunch of new skills?

In addition to being able to launch jobs using Kubernetes on the cluster, Nautilus provides a simpler interface called a [Jupyterlab Notebook](https://jupyter.org). This is for people who just want a easy way to interact with the cluster without any local installations or configurations. JupyterLab is a web-based interactive development environment for notebooks, code, and data. Its flexible interface allows users to configure and arrange workflows in data science, scientific computing, and machine learning. There are many places where you can try [JupyterLab](https://jupyter.org/try) online, but since our Jupyterlab is backed up by the Nautilus cluster, your Notebook and its code can run on as many as eight GPUs working as a single computational engine. Python is the main language used on JupyterLab. While Python is an essential skill for machine learning, it’s also easy to learn. If you want to try Nautilus, but without learning Unix, you can start by [learning Python](https://www.python.org/about/gettingstarted/).

And by the way, no need to install Python locally, just reach out to the community on [Matrix](https://nrp.ai/contact) to request access to the Jupyterlab running on Nautilus. It will run in your browser without any local software installations.

## Getting Help

Nautilus is a community-based cluster, managed by a small group of NSF-funded administrators. Every effort has been made to provide a world-class computational platform, but Nautilus is also largely self-service and community supported. Users are expected to learn-by-doing, and generally rely on community-based resources for troubleshooting. A diverse and active user community can be found by joining the Matrix channel. Having a Nautilus account is not required for Matrix, but having Matrix is required for joining Nautilus. Find out more about this by [joining the chat channel](https://nrp.ai/contact) today.

# Now, let’s get started…

[Get Access](/documentation/userdocs/start/getting-started/)

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/tutorial/introduction.mdx)

[Previous  
Asking for Support](/documentation/userdocs/start/support)  [Next  
Docker and Containers](/documentation/userdocs/tutorial/docker)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.