# AMD/Xilinx FPGAs

**Source:** https://nrp.ai/documentation/userdocs/fpgas/vivado-vitis

# AMD/Xilinx FPGAs

# Using FPGAs with Vivado and Vitis on Coder

For users looking to work with FPGAs, the **U55C FPGA Vitis Workflow template** in our Coder environment provides an efficient setup with all necessary tools, including **Vivado** and **Vitis**. This guide covers key steps and specific configurations needed to get started. For general setup instructions, workspace management, and other workspace templates, see the [Coder Environment Documentation](/documentation/userdocs/coder/coder).

---

## Overview

The **U55C FPGA Vitis Workflow template** provides a complete setup for FPGA development, including:

* **Vivado and Vitis versions**: Multiple preconfigured versions for flexibility across different projects.
* **Xilinx License Server**: Includes the **VitisNetP4 license**, supporting P4 workloads.
* **FPGA Requests**: Specify the number of FPGAs directly within the template for development and deployment.

### Steps

1. **Sign in to Coder**: Log in to [coder.nrp-nautilus.io](https://coder.nrp-nautilus.io) with institutional credentials.
2. **Select the U55C FPGA Vitis Workflow Template**: This template is specifically configured for FPGA projects using Vivado and Vitis.
3. **Configure Resources**: Allocate the required FPGA resources within the template settings.

### Workflow

Access tools like **Vivado** and **Vitis** through the **noVNC Desktop GUI** included in the workspace. Workspaces come with SSH keys preconfigured, making it easy to link your **GitLab** for version control. See [Coder Environment Documentation](userdocs/coder/coder/) for general tips and important notes on workspace management.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/fpgas/vivado-vitis.md)

[Previous  
FABRIC Integration](/documentation/userdocs/networks/fabric)  [Next  
ESnet SmartNIC](/documentation/userdocs/fpgas/esnet)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.