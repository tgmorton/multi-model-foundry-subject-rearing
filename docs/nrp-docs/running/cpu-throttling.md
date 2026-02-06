# CPU Throttling

**Source:** https://nrp.ai/documentation/userdocs/running/cpu-throttling

# CPU Throttling

The jobs are running in linux cgroups, and CPU limits are enforced by those. There’s a number of issues still pending that result in decreased performance if limits are not set right.

In general case, when the application can limit itself in the number of cores used, the request can be set to that number and limit to any higher number. This will accomodate all spikes that may occur. You should avoid setting the limit much higher than request and then constantly consuming more cores than request, as this will make the OS unstable in case there’s not enough cores left for the system. The scheduling desicions are always based on the request, not limit.

[Google best practices for requests and limits](https://cloud.google.com/blog/products/containers-kubernetes/kubernetes-best-practices-resource-requests-and-limits)

The grafana pod monitoring dashboard shows throttling for pods, and there’s an open ticket discussing what it means and how to make it more informative, but also highlighting the pending problems with throttling: <https://github.com/kubernetes-monitoring/kubernetes-mixin/issues/108>.

In general requesting full cores and using [Guaranteed QoS](https://kubernetes.io/docs/tasks/configure-pod-container/quality-service-pod/) (request == limit) seems to help avoid throttling.

Some more links to issues still open: [1](https://github.com/kubernetes/kubernetes/issues/67577), [2](https://github.com/kubernetes/kubernetes/issues/97445)

[CPU manager blog post](https://kubernetes.io/blog/2018/07/24/feature-highlight-cpu-manager/)

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/cpu-throttling.md)

[Previous  
High I/O jobs](/documentation/userdocs/running/io-jobs)  [Next  
JupyterHub Service](/documentation/userdocs/jupyter/jupyterhub-service)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.