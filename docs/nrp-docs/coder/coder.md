# Using Coder

**Source:** https://nrp.ai/documentation/userdocs/coder/coder

# Using Coder

# Coder Environment Documentation

Caution

If your workspace was created before 11/01/2024, please do not update your workspace version before backing up your data! Otherwise, your data might be lost.

TL;DR

1. **Coder** provides a quick-start development environment similar to JupyterHub, running directly on our cluster. No Kubernetes knowledge required!
2. **FPGA Requests**: Easily request and use Xilinx Alveo U55C FPGAs for specialized workloads.
3. **Preconfigured with Xilinx Vivado/Vitis License Server**, including **P4 license support**, so you can compile and deploy a variety of FPGA workloads right out of the box.
4. **Git Integration**: All workspaces come with SSH keys for seamless GitLab connections.
5. **Important**: Deleting any workspace also deletes its associated Persistent Volume Claim (PVC), so be sure to backup any critical data.

---

## Overview

We offer a **Coder** environment hosted at [coder.nrp-nautilus.io](https://coder.nrp-nautilus.io) on our cluster, providing an easy-to-use, JupyterHub-like experience without needing Kubernetes expertise. You can create an account, sign in with your institutional credentials via OpenID Connect, and start using Coder immediately. Users are allowed up to **5 active workspaces** and can easily request Xilinx Alveo U55C FPGAs for enhanced computing power.

Each workspace includes a persistent home folder, initially limited to 5GB, with the option to request additional storage. Workspaces are launched from a selection of templates tailored to different development needs.

## Coder CLI

For easier management, you can install the **Coder CLI** on your local machine [here](https://coder.com/docs/install). The CLI enables you to manage your account, handle workspaces, and SSH into any workspace directly from your terminal.

## Workspace Templates

### 1. General Template

This template provides a setup similar to JupyterHub, with configuration options for:

* Region
* CPU Cores
* Memory
* GPU Type
* FPGA Requests (highlighted options in bold)

Each image includes features like noVNC Desktop GUI, JupyterLab, an in-browser Terminal 💻, VSCode integration, and Cursor integration.

### 2. Selkies

Preloaded with **Selkies-GStreamer Streaming Service** for fast, GPU-accelerated remote desktop access. This setup includes **VSCode Integration**, **PyCharm**, Wine for Windows applications, and other useful tools.

Selkies Desktop Credentials are:

Username: ubuntu Password: mypasswd

### 3. CUDA/PyTorch/TensorFlow

Preloaded with an extensive selection of machine learning libraries, plus GPU support for ML/DL tasks.

### 4. U55C FPGA Vitis Workflow

Ideal for FPGA development, preloaded with multiple versions of Vivado and Vitis, along with access to the **Xilinx License Server** (includes **VitisNetP4 license for P4 workloads**). Allows for one or multiple FPGA requests.

### 5. ESnet FPGA SmartNICs

Includes all tools from the U55C FPGA Vitis Workflow, plus ESnet SmartNIC tools for configuring Xilinx Alveo U55C FPGAs as P4-programmable SmartNICs with 100Gbps per port ⚡.

### 6. Other Environments

Additional images are available for various development needs, such as Node.js, RStudio, and Golang.

## Workspace Management

After setting up a workspace, you can always adjust configurations later through the settings menu.

**Important**: Deleting any workspace also deletes its associated Persistent Volume Claim (PVC), so be sure to backup any critical data.

### Git Integration

All workspaces come preconfigured with an SSH key, available in your user settings, simplifying connections to GitLab. This makes backing up code a straightforward task.

---

For further assistance, feel free to contact the Nautilus Support team.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/coder/coder.mdx)

[Previous  
Deploy JupyterHub](/documentation/userdocs/jupyter/jupyterhub)  [Next  
Deploying Coder](/documentation/userdocs/coder/deploy)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.