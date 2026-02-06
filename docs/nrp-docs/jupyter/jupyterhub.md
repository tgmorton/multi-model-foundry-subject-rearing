# Deploy JupyterHub

**Source:** https://nrp.ai/documentation/userdocs/jupyter/jupyterhub

# Deploy JupyterHub

## Why Deploy Your Own JupyterHub?

JupyterHub provides a multi-user Jupyter notebook environment that allows you to:

* **Share computational resources** with your team, class, or research group
* **Standardize environments** across multiple users with consistent packages and configurations
* **Control access** to specific users or institutions through authentication
* **Customize the environment** with your own software stack, packages, and tools
* **Scale resources** based on your specific needs (CPU, memory, GPU)
* **Integrate with NRP infrastructure** for seamless access to cluster resources

This guide is based on the [Zero to Jupyter](https://zero-to-jupyterhub.readthedocs.io/en/stable/) guide with configurations specific to the Nautilus cluster. You must be the admin of the namespace you’re deploying to.

JupyterHub Culling Policy

**Deploying JupyterHub without culling is against cluster policies.**

All JupyterHub deployments **MUST** include culling configuration with a maximum idle time of **6 hours or less**. This is required to ensure fair resource allocation and prevent resource waste on the NRP cluster.

See the [culling configuration section](#culling-configuration) below for implementation details.

Note

There’s work going on making JupyterHub scalable and HA in this issue: <https://github.com/jupyterhub/jupyterhub/issues/1932>

## Initial Setup

Start from choosing the name for your project. It will look like `your_name.nrp-nautilus.io`

### Register CiLogon application

Register your application at <https://cilogon.org/oauth2/register>.

Set the callback url to `https://your_name.nrp-nautilus.io/hub/oauth_callback`

Client Type: `Confidential`

Scopes: `org.cilogon.userinfo,openid,profile,email`

Refresh Tokens: `No`

Save the client ID and Secret.

### Create the namespace

Create a namespace for your project on [Nautilus portal](https://nrp.ai/namespaces) and annotate it with all information.

## Configuration

### Install helm and download the helm chart

Follow the [install guide](https://zero-to-jupyterhub.readthedocs.io/en/stable/jupyterhub/installation.html), and use [this template](../values) for the config:

1. Run `openssl rand -hex 32` and replace the `secret_token` in the yaml file with the generated key
2. Minimally set the `client_id`, `client_secret`, `admin_users`, `secret_token`, `oauth_callback_url`, `ingress.hosts` fields.
3. Add security with either `allowed_idps` or `allowed_users`. Do NOT leave your JupyterHub instance open for anyone to sign in, this may result in locking of your namespace. You may find your IDP at <https://cilogon.org/idplist>
4. `helm repo add jupyterhub https://jupyterhub.github.io/helm-chart/ && helm repo update`
5. `helm upgrade --cleanup-on-fail --install jhub jupyterhub/jupyterhub --namespace <namespace> --version=3.3.7 --values config.yaml`

Once the pods start, you should be able to see the installation under your selected name.

### Automatic deployment

You can put your JupyterHub configuration in GitLab and automatically redeploy the application on repository changes. Please refer to [this guide](/documentation/userdocs/development/k8s-integration/) for details.

## Customization

### Understanding the Values Template

The [template values file](../values) provides a comprehensive starting point with:

* **Pre-configured profiles** for different scientific domains (Python, R, Julia, TensorFlow, PyTorch, etc.)
* **Resource limits** and guarantees for CPU and memory
* **Storage configuration** with Ceph block storage
* **Authentication settings** for CILogon integration
* **Ingress configuration** for external access

### Adding Your Own Container Image

To add your own custom container image to JupyterHub, you need to modify the `profileList` section in your `values.yaml`:

```
profileList:

- display_name: "My Custom Environment"

kubespawner_override:

image_spec: "your-registry.com/your-org/your-image:tag"

default: false  # Set to true if you want this as the default
```

**Key fields to modify:**

* **`display_name`**: What users see in the profile selector
* **`image_spec`**: Full path to your container image
* **`default`**: Whether this profile is selected by default

### Creating Custom Container Images

#### Option 1: Extend Existing Images

Start with a base Jupyter image and add your packages:

```
FROM jupyter/minimal-notebook:latest

# Install system dependencies

USER root

RUN apt-get update && apt-get install -y \

build-essential \

&& rm -rf /var/lib/apt/lists/*

# Switch back to jovyan user

USER jovyan

# Install Python packages

RUN pip install --no-cache-dir \

pandas \

matplotlib \

scipy \

scikit-learn \

your-custom-package
```

#### Option 2: Build from Scratch

Create a completely custom image:

```
FROM python:3.11-slim

# Install system dependencies

RUN apt-get update && apt-get install -y \

build-essential \

git \

&& rm -rf /var/lib/apt/lists/*

# Install JupyterHub requirements

RUN pip install --no-cache-dir \

jupyterhub \

notebook \

jupyterlab

# Install your scientific packages

RUN pip install --no-cache-dir \

numpy \

pandas \

matplotlib \

your-research-packages

# Create jovyan user (JupyterHub standard)

RUN useradd -m -s /bin/bash jovyan

USER jovyan

WORKDIR /home/jovyan

# Start Jupyter

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888"]
```

#### Building and Pushing Your Image

Terminal window

```
# Build the image

docker build -t your-registry.com/your-org/your-image:tag .

# Push to registry

docker push your-registry.com/your-org/your-image:tag
```

**Available registries on NRP:**

* **GitLab Container Registry**: `gitlab-registry.nrp-nautilus.io/your-project/your-image:tag`
* **Docker Hub**: `your-username/your-image:tag`
* **Quay.io**: `quay.io/your-org/your-image:tag`

### Extending Existing Images

The first method to create your own custom software environment for the JupyterHub instance is to create your own Docker image to be used with the instance. The easiest way to get started is to use a pre-existing image, such as `jupyter/minimal-notebook:latest` for a minimal Jupyter image or using a [Scientific Image](https://nrp.ai/documentation/userdocs/running/sci-img/), then extending either image with the packages you need. If you have an existing image that you would like to make usable within JupyterHub, you will need to install the `jupyterhub` and `notebook` Python packages in your image.

The general format would be:

```
FROM <registry_url>/<organization>/<your_project>:<optional_tag>

# Install Packages

pip install --no-cache-dir <packages>
```

An example of this would be:

```
FROM jupyter/minimal-notebook:latest

# Install packages

pip install --no-cache-dir pandas matplotlib scipy
```

Once you create your Dockerfile, you can build your image locally and push it to a container registry such as `gitlab-registry.nrp-nautilus.io` or have the [GitLab build and push the image for you automatically.](/documentation/userdocs/development/gitlab/)

### Allowing custom Anaconda environments

Sometimes you may want to be able to create custom environments on the fly in your JupyterHub instance and allow them to persist across sessions. This can be useful for development, lab workflows, or exploration assignments in classes.

For this, you will need to complete two steps.

1. Make sure your Jupyter image has `nb_conda_kernels` installed in the environment.
2. Create `.condarc` file in your `$HOME` path and add the config below.

   ```
   envs_dirs:

   - /home/jovyan/my-conda-envs/
   ```

After those steps are complete, Anaconda environments can be created within the Jupyter session and persist across sessions after they close.

### Adding images to your configuration

In other cases, you may want to set specific environments to use for different tasks, assignments, or labs to avoid many redundent environments.

The [example values](../values) from earlier in this guide already has a few environments defined that provide a broad range of applications to use. To add your image to the list of available images, you will need to add the values below to your `profileList:`

```
- display_name: Name To Show

kubespawner_override:

image_spec: <registry_url>/<organization>/<your_project>:<optional_tag>
```

If it is to be the default image, add `default: True`.

### Shared Storage

If you are working with others on the same project or distributing data out for a class, you can add a `PersisitentVolumeClaim` as a shared location across all of the pods in the JupyterHub instance.

For example, using the example from the [Zero to JupyterHub guide](https://z2jh.jupyter.org/en/stable/jupyterhub/customizing/user-storage.html), we can example the `storage:` section of our example values to:

```
storage:

type: dynamic

extraLabels: {}

# Change starts here

extraVolumes:

- name: jupyterhub-shared

persistentVolumeClaim:

claimName: jupyterhub-shared-volume

extraVolumeMounts:

- name: jupyterhub-shared

mountPath: /home/shared

# Change Ends

capacity: 5Gi

homeMountPath: /home/jovyan

dynamic:

storageClass: rook-ceph-block

pvcNameTemplate: claim-{username}{servername}

volumeNameTemplate: volume-{username}{servername}

storageAccessModes: [ReadWriteOnce]
```

This would mount the shared storage to `/home/shared` using the `jupyterhub-shared-volume` PVC. Please note, that for the PVC to be used across multiple pods, the volume would need to have an appropriate access mode such as `ReadOnlyMany` or `ReadWriteMany`.

### Authentication

Note

This section will be referencing the [example values](../values) for a JupyterHub deployment.

#### Limit access to your University

As a basic step to help limit access to your JupyterHub instance and not leaving it wide open, you should set the `allowed_idps` to be that of only your university and any other universities that your collaborators are from.

The available Identity providers (idps) are listed in this list from CILogon: <https://cilogon.org/idplist/>

Once on that page, search for your university by name or url. Once found, the idps value for the configuration will be under “EntityID”. The [example values](../values) uses the University of Nebraska-Lincoln as the example. The “EntityID” for that university is `https://shib.unl.edu/idp/shibboleth`.

Under each identity provider, you will also need to add the `allowed_domains`, which will often be your university’s URL. For example, the University of Nebraska-Lincoln’s `allowed_domains` would be `unl.edu`.

#### Admin Users

Admin users can be set in the `admin_users` list under `JupyterHub`. Admins are identified by their email address used to log into Nautilus.

```
JupyterHub:

admin_access: true

admin_users: ["[email protected]","[email protected]"]
```

With the example config, admin users can access another user’s notebooks. If you want to disable this, set `admin_access` to `false`

#### Allowed Users

Allowed users can be set in the `allowed_users` list under `JupyterHub`. Admins are identified by their email address used to log into Nautilus.

```
JupyterHub:

allowed_users: ["[email protected]","[email protected]"]
```

With the example config, admin users can access another user’s notebooks. If you want to disable this, set `admin_access` to `false`

## Culling Configuration

**Required:** All JupyterHub deployments must include culling configuration to automatically shut down idle user servers. This is mandatory for compliance with NRP cluster policies.

### Basic Culling Setup

Add the following configuration to your `values.yaml` file at the root level:

```
cull:

enabled: true

users: false

removeNamedServers: false

timeout: 3600      # 1 hour in seconds - Must be ≤ 21600 (6 hours)

every: 600         # Check every 10 minutes

concurrency: 10    # Number of parallel culling operations

maxAge: 0          # No maximum age limit
```

### Culling Parameters

* **`enabled`**: Set to `true` to enable culling
* **`users`**: Set to `false` to only cull servers, not user accounts
* **`removeNamedServers`**: Set to `false` to preserve named servers
* **`timeout`**: Maximum idle time before culling (in seconds). **Must be ≤ 21600 (6 hours)**
* **`every`**: How often to check for idle servers (in seconds)
* **`concurrency`**: Number of parallel culling operations
* **`maxAge`**: Maximum age of servers regardless of activity (0 = no limit)

### Verification

After deployment, verify culling is working by checking the culling service logs:

Terminal window

```
kubectl logs -n <namespace> deployment/jupyterhub -c hub | grep cull
```

You should see periodic messages about culling operations.

## Good Practices

When setting up a custom JupyterHub there are a couple of good practices you can implement to help keep the environment sustainable and secure.

### Limit who has access

When deploying a JupyterHub instance, you should lock down the service to who should have access rather than leaving the instance open.

At minimum, it should be [limited to your University](#limit-access-to-your-university).Further steps can be taken to limit it to [specific individuals](#allowed-users), which is recommended for labs and small classes.

### Use Git

While developing your configuration, Git will be a useful tool to keep track of any changes you make. If something goes wrong with your configuration, using the history in Git will help revert any changes made that caused the issues. When working with the Nautilus cluster, you can use the [hosted Gitlab instance](https://gitlab.nrp-nautilus.io/) to keep track of the changes and then [automatically deploy the changes](#automatic-deployment).

### Documentation

As you start and continue to use your JupyterHub instance, it is strongly encouraged to keep a running docment of how the instance is setup and any workflows or assignments that run on it. This will help others use and maintain the instance, along with helping with future development and debugging.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/jupyter/jupyterhub.md)

[Previous  
ML/Jupyter Pod](/documentation/userdocs/jupyter/jupyter-pod)  [Next  
Using Coder](/documentation/userdocs/coder/coder)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.