# Basic Kubernetes (k8s) Tutorial

**Source:** https://nrp.ai/documentation/userdocs/tutorial/basic

# Basic Kubernetes (k8s) Tutorial

In this tutorial, you will be introduced to basic k8s commands, how to launch simple pods and deployments, as well as interact with the cluster to query its status and the status of your processes running in the cluster. You will also see your first example of a YAML file.

Question?

Throughout these tutorials, you may see a “Question?“. This is an opportunity for you to pause and reflect, or answer a quick self-assessment that can lead to deeper understanding of the materials.

## Prerequisites

This section assumes you’ve completed the [quickstart](/documentation/userdocs/start/getting-started/) section.

## Learning Objectives

1. You will understand the basic format of k8s commands.
2. You will be able to use k8s commands to interact with the cluster.
3. You will have a basic understanding of YAML and you will be able to use that to create a simple pod.
4. You will have an understanding of the difference between an pod and a deployment.
5. You will have a basic understanding of the “stateless” nature of a pod and how deployments can be used to specify an ideal state for your pods or software containers (running inside your pods).

Setting your namespace

Since you only have permissions to run in namespaces of which you are a member, you must specify your namespace. For the purposes of this tutorial, it is recommended that you set your namespace globally with this command:

Terminal window

```
kubectl config set-context nautilus --namespace=<the_namespace>
```

Some users may operate concurrently across several namespaces, and k8s allows you to specify which in each `kubectl` command by adding `-n <a_namespace>`, regardless of whether you have set your context as above. Most users save the keystrokes by setting the context for themselves and deviate only when necessary, rather than specifying each time.

## Explore the system

Now that you understand how to set your namespace, let’s begin to explore the system.

### List cluster nodes

The Nautilus Cluster is widely geographically distributed and highly heterogenous. You can get a sense of the types of nodes in the system by typing:

Terminal window

```
kubectl get nodes
```

**Please note**: You likely won’t have access to all the nodes listed, as some are reserved.

### List processes running in your namespace

There are three categories of processes we will examine:

* [pods](/documentation/userdocs/start/glossary)
* [deployments](/documentation/userdocs/start/glossary)
* [services](/documentation/userdocs/start/glossary)

Listing the categories running in k8s follows a similar format.

Terminal window

```
kubectl get <category>
```

Right now you probably don’t have anything running in the namespace, and these commands will return `No resources found in ... <namespace>.` but you will revisit these commands as we step through the tutorials as a way of checking on status.

#### List pods

List all the pods in your namespace

Terminal window

```
kubectl get pods
```

#### List deployments

List all the deployments in your namespace

Terminal window

```
kubectl get deployments
```

#### List services

List all the services in your namespace

Terminal window

```
kubectl get services
```

## Setup your first pod with YAML

Let’s create a simple generic pod, and then login into it.

### A simple YAML file

Create the `pod1.yaml` file with the following contents by copy-pasting:

pod1.yaml

```
apiVersion: v1

kind: Pod

metadata:

name: test-pod

spec:

containers:

- name: mypod

image: ubuntu

resources:

limits:

memory: 100Mi

cpu: 100m

requests:

memory: 100Mi

cpu: 100m

command: ["sh", "-c", "echo 'Im a new pod' && sleep infinity"]
```

**Reminder**: Indentation is important in YAML, just like in Python

A simple way to create a file un Unix

There are many ways to create a file and put data into it in Unix. One of the easiest ways is to redirect the standard input with the following command:

Create a new file

```
cat > <name_of_new_file>
```

You can then paste any copied text or type directly into the command line interface. Once you are done, you can `control-c` out of the input stream. This will do the job without leaving the command line interface. Try it by copying the `pod1.yaml` sample above, and then pasting it into a new file using the Unix `cat` command.

### Creating YAML files dynamically

Alternatively, if you don’t want to create a file and are using Unix-like system, you can create YAML files dynamically like this:

Terminal window

```
kubectl create -f - << EOF

<contents you want to deploy>

EOF
```

### Launch a simple pod

Making sure you are in the same file directory as your `pod1.yaml` file, type the following command:

Terminal window

```
kubectl create -f pod1.yaml
```

After a few moments (as the pod is creating itself), see if you can find it:

Terminal window

```
kubectl get pods
```

**Please Note**: You may see the other pods too.

If it is not yet in Running state, you can check what is going on with a list of the events in your namespace:

Terminal window

```
kubectl get events --sort-by=.metadata.creationTimestamp
```

Events and other useful information about the pod can be seen in `describe`:

Terminal window

```
kubectl describe pod test-pod
```

:question: Where did the name *test-pod* come from? Examine the `pod1.yaml` file to find the answer.

If the pod is in Running state, we can check it’s logs

Terminal window

```
kubectl logs test-pod
```

Let’s log into it

Terminal window

```
kubectl exec -it test-pod -- /bin/bash
```

There’s a relationship between the operating system and the command line interpreter

The last part of this command, specifying `bash` can change, depending on the operating system we choose. Keep this in mind when you deploy operating systems other than Ubuntu.

Question?

Did you manage to log into your pod?

If yes, you are now inside the (container in the) pod!

Question?

Does it feel any different than a regular, dedicated node?

Try to create some directories and some files with content (using the `cat` command, if you like). “Hello world” will do, but feel free to be creative.

### Let’s examine the pod’s networking

Networking inside a Kubernetes pod is crucial for facilitating communication between containers within the same pod and enabling connectivity with other pods, as well as external services. Understanding how networking works inside a pod is essential for building and deploying applications effectively in a Kubernetes environment. So, let’s examine how the network is configured inside our simple pod by logging into it.

Question?

Do we have the necessary tools to examine the network installed in our pod?

Remember that pods running inside k8s are “stateless” and since we didn’t specify any software packages to be included in our [simple pod](#a-simple-yaml-file) example when we launched it, we have some work to do before we can look at the network.

The package `ifconfig` is not included in our initial pod; so let’s install it.

First, let’s make sure our installation tools are updated.

Terminal window

```
apt update
```

Now, we can use apt to install the necessary network tools.

Terminal window

```
apt install net-tools
```

Now check the networking:

Terminal window

```
ifconfig -a
```

Question?

What did you discover? Does the output look like you’d expect?

Finally, let’s exit out of the pod and move on by entering the `exit` command in the command line interface of the pod, or using the keyboard shortcut ‘Control-D’.

### Testing statelessness

To demonstrate that pods really are stateless, we are going to shutdown our simple pod, and repeat the creation of our simple pod, then take a look at it again.

### Shutdown the originall

Let’s manually shut down the pod using `kubectl`

Terminal window

```
kubectl delete -f pod1.yaml
```

This may take a moment, as the system will remove the pod gracefully. After a few moments, check that it is actually gone:

Terminal window

```
kubectl get pods
```

Question?

Is it gone?

If yes, let’s create it again:

Terminal window

```
kubectl create -f pod1.yaml
```

Accessing prior commands from the CLI

Most command line interfaces store the most recent commands you have entered. You can access them easily by using your keyboard’s “up arrow” which is a quick and easy way to repeat commands.

### Looking at Pod1.yaml (again)

Give the system a moment to create the new pod.

Question?

Does it have the same IP? We can check by using the following command:

```
kubectl get pod -o wide test-pod
```

Log back into the pod:

```
kubectl exec -it test-pod -- /bin/bash
```

Now, let’s look for the files you created with the `cat` command. Are they where you left them? What is the status of the files your created?

Question?

How does this exercise demonstrate “statelessness” and what are the implications for how I prepare for the inevitable and normal restarting of a pod?

### Cleaning up

Since Nautilus is a shared platform and a community resource, it’s very important that we don’t leave any stray processes running. So, everytime we’re finished using a pod, we should make sure to clean up after ourselves.

So, let’s delete explicitly the pod, using the following command:

Terminal window

```
kubectl delete pod test-pod
```

## Let’s transform our pod into a deployment

You saw that when a pod was terminated, it was gone. While above we did it by ourselves, the result would have been the same if a node died or was restarted. This is normal and expected in a k8s cluster, and is actually a great feature of Kubernetes: it is highly resilient. Kubernetes constantly monitors the health of pods and containers. In the event of a failure or unresponsiveness, Kubernetes automatically restarts containers or entire pods to maintain the desired state specified by the user.

In order to specify to the cluster your “desired state”, the use of Deployments is recommended.

You can copy-and-paste the lines below into a new file on your local system (using the `cat` command, if you like).

###### Deployment 1

dep1.yaml

```
apiVersion: apps/v1

kind: Deployment

metadata:

name: test-dep

labels:

k8s-app: test-dep

spec:

replicas: 1

selector:

matchLabels:

k8s-app: test-dep

template:

metadata:

labels:

k8s-app: test-dep

spec:

containers:

- name: mypod

image: ubuntu

resources:

limits:

memory: 500Mi

cpu: 500m

requests:

memory: 100Mi

cpu: 50m

command: ["sh", "-c", "sleep infinity"]
```

Now let’s start the deployment:

Terminal window

```
kubectl create -f dep1.yaml
```

See if you can find it:

Terminal window

```
kubectl get deployments
```

The Deployment is just a conceptual service

It describes to the cluster the *ideal state* of your pods. It doesn’t actually do more than that. In this case, the *ideal state* is a single replica of the container called “mypod”.

See if you can find the associated pod:

Terminal window

```
kubectl get pods
```

Once you have found the name assigned to it by the cluster, let’s log into it.

Terminal window

```
kubectl get pod -o wide test-dep-<hash>

kubectl exec -it test-dep-<hash> -- /bin/bash
```

You are now inside the (container in the) pod!

### Testing statelessness with deployments

Create directories and files as before.

Try various commands as before.

Let’s now delete the pod!

Terminal window

```
kubectl delete pod test-dep-<hash>
```

Is it really gone?

Terminal window

```
kubectl get pods
```

What happened to the deployment?

Terminal window

```
kubectl get deployments
```

Get into the new pod

Terminal window

```
kubectl get pod -o wide test-dep-<hash>

kubectl exec -it test-dep-<hash> -- /bin/bash
```

Question?

Was anything preserved? How might you make sure the settings, software packages, and data you need in your pod is preserved?

Using the right image helps with reliability and resiliency

Ensuring that the software container you are deploying has all the necessary elements to function involves creating a well-packaged and self-contained container image. Unlike the simple examples above, you can make a robust and reliable deployment by specifying an appropriate base image, and including any system libraries, and any language-specific runtime or dependencies in your software container.

Question?

Examining the [`dep1.yaml`](#deployment-1) example above, what would you change to specify a different image? Keep that in mind as we progress through the tutorials. If you are curious and want to read ahead, we cover that topic in the section called “[Images](/documentation/userdocs/tutorial/images)”.

Let’s now delete the deployment:

Terminal window

```
kubectl delete -f dep1.yaml
```

Verify everything is gone:

Terminal window

```
kubectl get deployments

kubectl get pods
```

## Next steps

In the next [tutorial](../basic2), we will explore how to run a simple web server in a pod, and how to expose it to the outside world and scale it.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/tutorial/basic.md)

[Previous  
Docker and Containers](/documentation/userdocs/tutorial/docker)  [Next  
Scaling and Exposing](/documentation/userdocs/tutorial/basic2)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.