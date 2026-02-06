# Deployed Services

**Source:** https://nrp.ai/documentation/userdocs/start/resources

# Deployed Services

Although you can run your own containers, there are services and resources already deployed by cluster admins that you can use without creating those yourself.

Caution

Most services require additional registration.

Stable - Service is **generally supported**. You can report issues with the service.

Unsupported - Service is provided **as is**. You can report the issues but those will be checked with lowest priority. The service might be removed soon.

Experimental - The service is added **for testing**. Likely to have bugs or change at any time. Also might be removed without any notification.

**Please report broken services in [Matrix Support](/documentation/userdocs/start/support/).**

## Computations

* [Stable JupyterHub](https://jupyterhub-west.nrp-nautilus.io) [(Docs)](/documentation/userdocs/jupyter/jupyterhub-service)
* [Stable Coder: remote development environment](https://coder.nrp-nautilus.io) [(Docs)](/documentation/userdocs/coder/coder)
* [Stable Kubevirt: Virtual Machines](/documentation/userdocs/running/virtualization-general)
* [Unsupported WebODM (Web Open Drone Map): Drone Images stitching](https://webodm.nrp-nautilus.io)

## Data sharing and collaboration tools

* [Stable GitLab: code and containers repository](https://gitlab.nrp-nautilus.io)
* [Stable Overleaf: LaTeX collaboration](https://overleaf.nrp-nautilus.io)
* [Stable EtherPad: notebooks](https://etherpad.nrp-nautilus.io)
* [Stable BentoPDF: In-Browser PDF Editing and Manipulation](https://bentopdf.nrp-nautilus.io)
* [Stable Jitsi: Video conferencing](https://jitsi.nrp-nautilus.io)
* [Stable Nextcloud: File sharing](https://nextcloud.nrp-nautilus.io)
* [Stable Yopass: one-time secret sharing](https://credentials.nrp-nautilus.io)
* [Unsupported SyncThing: File sync](https://syncthing.net) ([Contact us to set up](https://nrp.ai/contact))
* [Unsupported Hedgedoc: collaborative markdown editor](https://hedgedoc.nrp-nautilus.io)

## Artificial intelligence and LLM

* [Stable NRP-Managed LLMs and services](/documentation/userdocs/ai/llm-managed)

## Network monitoring

* [Stable PerfSONAR](https://perfsonar.nrp-nautilus.io/maddash-webui/)
* [Unsupported Traceroute tool](https://traceroute.nrp-nautilus.io)

## Others

* [Stable NetBox - infrastructure tracking](https://netbox.nrp-nautilus.io)

## Kubernetes operators (CRDs)

### Databases

* Stable [PostgreSQL (zalan.do)](/documentation/userdocs/running/postgres)
* Stable [Elasticsearch](https://www.elastic.co/guide/en/cloud-on-k8s/current/k8s-elasticsearch-specification.html)
* Stable [Prometheus](https://prometheus-operator.dev) (users are allowed to create [`ServiceMonitors`](https://prometheus-operator.dev/docs/api-reference/api/#monitoring.coreos.com/v1.ServiceMonitor))
* Experimental [Nebula graph](https://www.nebula-graph.io)
* Experimental [ScyllaDB](https://operator.docs.scylladb.com/stable/resources/scyllaclusters/index.html)

### Compute

* Stable [Ray](/documentation/userdocs/running/ray-cluster)
* Stable [Dask](/documentation/userdocs/running/dask-cluster)
* Stable [Kubeflow](/documentation/userdocs/running/kubeflow)

### Others

* Stable [Cert-manager](/documentation/userdocs/running/ingress#auto-renewing-the-certificate)
* Stable [Sealed secrets](https://github.com/bitnami-labs/sealed-secrets?tab=readme-ov-file#usage)
* Stable [Multus](https://github.com/k8snetworkplumbingwg/multus-cni)
* Stable [Admiralty federation](https://admiralty.io)
* Experimental [Clabernetes Containerlab](https://containerlab.dev/manual/topo-def-file/)
* Experimental [keda.sh autoscaling](https://keda.sh)

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/start/resources.mdx)

[Previous  
Cluster Policies](/documentation/userdocs/start/policies)  [Next  
Glossary](/documentation/userdocs/start/glossary)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.