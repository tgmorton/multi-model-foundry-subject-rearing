# Cloud AI 100

**Source:** https://nrp.ai/documentation/userdocs/ai/qaic

# Cloud AI 100

# Qualcomm Cloud AI 100 Ultra Documentation

This documentation provides guidelines for deploying and managing workloads on the Qualcomm Cloud AI 100 Ultra cards available in the Nautilus cluster.

## Overview

The Nautilus cluster is equipped with **8 Qualcomm Cloud AI 100 Ultra AI-Accelerator cards**. Each card contains **4 SoCs (devices)**, resulting in a total of **32 devices (32 QIDs)**. Every device includes **16 cores (NSPs)** and can load large language models (LLMs) of up to approximately **25B parameters**.

## Resource Allocation in Kubernetes

To leverage these accelerators in your deployments, set the resource requests and limits using the resource type `qualcomm.com/qaic`. Each device is exposed as one allocatable resource. With 32 available devices, you can allocate up to 32 `qualcomm.com/qaic` resources.

### Example YAML Configuration

The following YAML snippet demonstrates how to configure resource requests and limits in your Kubernetes deployment:

```
resources:

requests:

cpu: "4"

memory: "16Gi"

qualcomm.com/qaic: 1

limits:

cpu: "8"

memory: "32Gi"

qualcomm.com/qaic: 1
```

Adjust the CPU and memory values as required for your workload.

## In-House Docker Image with vLLM Support

An in-house Docker image has been built with the latest SDK and vLLM support, optimized for the Qualcomm Cloud AI 100 Ultra cards.

**Docker Image:** `gitlab-registry.nrp-nautilus.io/cloud-ai-100/qaic-docker-images:vllm-latest`

This image includes all necessary tools and libraries to efficiently utilize the Qualcomm accelerators.

## Additional Resources

For more detailed instructions on installation and performance tuning, please refer to the official Qualcomm documentation:

* [Qualcomm Cloud AI SDK vLLM Documentation](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/vLLM/vLLM/)
* [Efficient Transformers Documentation](https://quic.github.io/efficient-transformers)

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/ai/qaic.md)

[Previous  
LLM in JupyterHub](/documentation/userdocs/ai/llm-jupyterhub)  [Next  
Vector database](/documentation/userdocs/ai/vector-database)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.