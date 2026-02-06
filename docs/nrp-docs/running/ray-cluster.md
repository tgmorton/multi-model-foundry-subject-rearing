# Ray Cluster

**Source:** https://nrp.ai/documentation/userdocs/running/ray-cluster

# Ray Cluster

[Ray](https://docs.ray.io/en/latest/index.html) is an open-source unified framework for scaling AI and Python applications like machine learning. It provides the compute layer for parallel processing so that you don’t need to be a distributed systems expert. To run Ray applications on multiple nodes, you must first deploy a Ray cluster. This document is adapted from the official guide of [RayCluster Quickstart](https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/raycluster-quick-start.html#kuberay-raycluster-quickstart) for deploying and using a Ray Cluster on Nautilus in your namespace.

Note

This document assumes you’re deploying in your default namespace. If you are in multiple namespaces and you are deploying Ray in a non-default namespace, you will need to append `-n <namespace-name>` to the `helm` and `kubectl` commands below.

1. Deploy a RayCluster custom resource

   Once the KubeRay operator is running, create a RayCluster Custom Resource (CR):

   ```
   helm install raycluster kuberay/ray-cluster
   ```

   View the RayCluster CR:

   Terminal window

   ```
   kubectl get rayclusters

   # NAME                 DESIRED WORKERS   AVAILABLE WORKERS   CPUS   MEMORY   GPUS   STATUS   AGE

   # raycluster-kuberay   1                 1                   2      3G       0      ready    121m
   ```

   The KubeRay operator will detect the RayCluster object, then start your Ray cluster by creating head and worker pods. To vew Ray cluster’s pods, run the following command:

   Terminal window

   ```
   kubectl get pods --selector=ray.io/cluster=raycluster-kuberay
   ```

   A `raycluster-kuberay-head` pod and a `raycluster-kuberay-worker` pod should be listed in the output:

   Terminal window

   ```
   # NAME                                          READY   STATUS    RESTARTS   AGE

   # raycluster-kuberay-head-8gjxh                 1/1     Running   0          124m

   # raycluster-kuberay-worker-workergroup-w74gh   1/1     Running   0          124m
   ```
2. Run an application on a RayCluster

   Once the RayCluster has been deployed, users in the namespace can run Ray jobs. The most straightforward way is to exec directly into the head pod.

   Identify the RayCluster’s head pod:

   Terminal window

   ```
   export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)

   echo $HEAD_POD

   # raycluster-kuberay-head-8gjxh
   ```

   Execute a Ray job to print the cluster resources:

   Terminal window

   ```
   kubectl exec -it $HEAD_POD -- python -c "import ray; ray.init(); print(ray.cluster_resources())"

   # 2024-07-24 20:51:00,788 INFO worker.py:1405 -- Using address 127.0.0.1:6379 set in the environment variable RAY_ADDRESS

   # 2024-07-24 20:51:00,788 INFO worker.py:1540 -- Connecting to existing Ray cluster at address: 10.244.110.211:6379...

   # 2024-07-24 20:51:00,797 INFO worker.py:1715 -- Connected to Ray cluster. View the dashboard at http://10.244.110.211:8265

   # {'node:10.244.231.23': 1.0, 'memory': 3000000000.0, 'object_store_memory': 751175270.0, 'CPU': 2.0, 'node:10.244.110.211': 1.0, 'node:__internal_head__': 1.0}
   ```

   You may also submit a Ray job to the RayCluster via ray job submission SDK. For more information, please refer to <https://docs.ray.io/en/latest/cluster/kubernetes/getting-started/raycluster-quick-start.html#method-2-submit-a-ray-job-to-the-raycluster-via-ray-job-submission-sdk>
3. Access the Ray Dashboard

   Execute this command to create a tunnel:

   Terminal window

   ```
   kubectl port-forward service/raycluster-kuberay-head-svc 8265:8265
   ```

   Visit <http://localhost:8265> in your browser for the Dashboard.
4. Cleanup

   When the RayCluster is not needed anymore, delete the RayCluster CR:

   Terminal window

   ```
   # Uninstall the RayCluster Helm chart

   helm uninstall raycluster

   # release "raycluster" uninstalled
   ```

   It might take several seconds for the RayCluster’s pods to terminate. Confirm that the pods are gone by running:

   Terminal window

   ```
   kubectl get pods
   ```

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/ray-cluster.mdx)

[Previous  
Windows](/documentation/userdocs/running/virtualization-windows)  [Next  
Dask Cluster](/documentation/userdocs/running/dask-cluster)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.