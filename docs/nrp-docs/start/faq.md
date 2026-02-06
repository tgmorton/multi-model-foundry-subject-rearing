# FAQ

**Source:** https://nrp.ai/documentation/userdocs/start/faq

# FAQ

**My pod is stuck Terminating.**

> This happens for a few reasons, such as:
>
> * The node running your pod went offline. The pod will finish terminating once the node is back online.
> * The storage attached to the pod can’t be unmounted.
> * Due to the high load on the node, your pod termination process could not be completed. In all these cases, you should ask a cluster admin in [Matrix](/contact) chat to look at your pod, or just wait for somebody to fix it.
>
>   Do Not Force delete your pod
>
>   Do not use `kubectl delete --grace-period=0 --force` to delete stuck pods!  
>   It will keep resources attached to the node for an indefinite period and will require the rebooting of the node (if you happen to know which node was it).

---

**I tried to use `nvprof` in my GPU pod and got an error.**

> There is a [vulnerability](https://nvidia.custhelp.com/app/answers/detail/a_id/4738) in NVIDIA drivers still not fixed, and this feature is disabled by default. Enabling it requires too much effort, so for now we keep it default. Hopefully it will be fixed soon.

---

**How do I acknowledge support from NRP / Nautilus in research papers?**

1. Please acknowledge the NRP NSF grants in the format specified by the [AUP](https://nrp.ai/NRP-AUP.pdf).
2. Please cite the following paper when acknowledging the National Research Platform (NRP):

[The National Research Platform: Stretched, Multi-Tenant, Scientific Kubernetes Cluster](https://dl.acm.org/doi/10.1145/3708035.3736060)

**BibTeX:**

```
@inproceedings{10.1145/3708035.3736060,

author = {Weitzel, Derek and Graves, Ashton and Albin, Sam and Zhu, Huijun and Wuerthwein, Frank and Tatineni, Mahidhar and Mishin, Dmitry and Khoda, Elham and Sada, Mohammad and Smarr, Larry and DeFanti, Thomas and Graham, John},

title = {The National Research Platform: Stretched, Multi-Tenant, Scientific Kubernetes Cluster},

year = {2025},

isbn = {9798400713989},

publisher = {Association for Computing Machinery},

address = {New York, NY, USA},

url = {https://doi.org/10.1145/3708035.3736060},

doi = {10.1145/3708035.3736060},

abstract = {The National Research Platform (NRP) represents a distributed, multi-tenant Kubernetes-based cyberinfrastructure designed to facilitate collaborative scientific computing. Spanning over 75 locations in the U.S. and internationally, the NRP uniquely integrates varied computational resources, ranging from single nodes to extensive GPU and CPU clusters, to support diverse research workloads including advanced AI and machine learning tasks. It emphasizes flexibility through user-friendly interfaces such as JupyterHub and low level control of resources through direct Kubernetes interaction. Critical operational insights are discussed, including security enhancements using Kubernetes-integrated threat detection, extensive monitoring, and comprehensive accounting systems. This paper highlights the NRP’s growing importance and scalability in addressing the increasing demands for distributed scientific computational resources.},

booktitle = {Practice and Experience in Advanced Research Computing 2025: The Power of Collaboration},

articleno = {69},

numpages = {5},

keywords = {Distributed Computing, Kubernetes, High Throughput Computing, Artificial Intelligence},

location = {},

series = {PEARC '25}

}
```

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/start/faq.mdx)

[Previous  
Glossary](/documentation/userdocs/start/glossary)  [Next  
Asking for Support](/documentation/userdocs/start/support)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.