# Special Use

**Source:** https://nrp.ai/documentation/userdocs/running/special

# Special Use

Our cluster combines various hardware resources from multiple universities and other organizations.

Caution

By default you can only use the nodes **having NO Taints** (see the [resources page](https://nrp.ai/viz/resources) of the portal).

#### All Taints

Please use caution when applying tolerations and only tolerate taints for which you have explicit authorization from cluster administrators. Tolerating the wrong taints may cause your workloads to land on unintended or restricted nodes, leading to failures or policy violations.

Here is the taint system and their descriptions. To run on a node with a taint, you need to [use the node toleration in your pod](https://kubernetes.io/docs/concepts/scheduling-eviction/taint-and-toleration/). Users **may only tolerate values they are authorized for by cluster admins**.

| Taint Key | Description |
| --- | --- |
| **nautilus.io/reservation** | For user-facing reservations. Users can only tolerate a value here if they are part of an approved group. If unsure, toleration is not allowed. |
| **nautilus.io/hardware** | For special hardware nodes that users should not land on by default but are permitted to tolerate if needed. |
| **nautilus.io/system** | For system services or any infrastructure nodes that users should never schedule onto. Toleration is not permitted. |
| **nautilus.io/issue** | For nodes with temporary issues that should not accept user workloads. Numeric values indicate GitLab issues, strings indicate other issues. Users cannot tolerate these. |

---

##### Taint name mappings from old to new system

| Old Taint | New Taint |
| --- | --- |
| nautilus.io/5g=true | nautilus.io/system=5g:NoSchedule |
| nautilus.io/ceph=true | nautilus.io/system=storage:NoSchedule |
| nautilus.io/ceph-external=true | nautilus.io/system=storage:NoSchedule |
| nautilus.io/jump=true | nautilus.io/system=jump:NoSchedule |
| nautilus.io/perfsonar=true | nautilus.io/system=perfsonar:NoSchedule |
| nautilus.io/linstor-server=true | nautilus.io/system=storage:NoSchedule |
| nautilus.io/bluefield2=true | nautilus.io/reservation=bluefield2:NoSchedule |
| nautilus.io/csusb=true | nautilus.io/reservation=csusb:NoSchedule |
| nautilus.io/csu-tide=true | nautilus.io/reservation=csu-tide:NoSchedule |
| nautilus.io/fpga-tutorial=true | nautilus.io/reservation=fpga-tutorial:NoSchedule |
| nautilus.io/genai-lab=true | nautilus.io/reservation=genai-lab:NoSchedule |
| nautilus.io/mizzou=true | nautilus.io/reservation=mizzou:NoSchedule |
| msu-cache=true | nautilus.io/reservation=msu-cache:NoSchedule |
| nautilus.io/prism-center=true | nautilus.io/reservation=prism-center:NoSchedule |
| nautilus.io/qaic=undefined | nautilus.io/reservation=qaic:NoSchedule |
| nautilus.io/reservation=cogrob | nautilus.io/reservation=cogrob:NoSchedule |
| nautilus.io/reservation=wifire | nautilus.io/reservation=wifire:NoSchedule |
| nautilus.io/nrp-llm=true | nautilus.io/reservation=nrp-llm:NoSchedule |
| nautilus.io/sdsc-llm=true | nautilus.io/reservation=sdsc-llm:NoSchedule |
| nautilus.io/sense=true | nautilus.io/reservation=sense:NoSchedule |
| nautilus.io/stashcache=true | nautilus.io/reservation=osdf:NoSchedule |
| nautilus.io/suncave=true | nautilus.io/reservation=suncave:NoSchedule |
| nautilus.io/suncave-head=true | nautilus.io/reservation=suncave-head:NoSchedule |
| um-cache=true | nautilus.io/reservation=um-cache:NoSchedule |
| nautilus.io/arm64=true | nautilus.io/hardware=arm64:NoSchedule |
| nautilus.io/large-gpu=true | nautilus.io/hardware=large-gpu:NoSchedule |
| nautilus.io/disk-swap=true | nautilus.io/issue=disk-swap:NoSchedule |
| nautilus.io/gitlab-issue=1234 | nautilus.io/issue=1234:NoSchedule |
| nautilus.io/slow-network=true | nautilus.io/issue=slow-network:NoSchedule |
| nautilus.io/testing=true | nautilus.io/issue=testing:NoSchedule |
| nautilus.io/upgrading=true | nautilus.io/issue=upgrading:NoSchedule |
| node.kubernetes.io/unreachable=undefined | node.kubernetes.io/unreachable=undefined:NoSchedule |

[Observable notebook with taints summary](https://observablehq.com/d/1cb451398e09c3ff)

#### Reservations

**Groups may request exclusive access to entire nodes** if their workloads justify it. Such nodes can be reserved by setting the following `taint` and corresponding `toleration`:

```
spec:

tolerations:

- key: "nautilus.io/reservation"

operator: "Equal"

value: "group1"

effect: "NoSchedule"
```

Please fill out the [node reservation form](https://nrp.ai/reservations/) if your group has a use case that would benefit from whole-node reservations.

In addition, our cluster contains several sets of nodes dedicated to certain groups.

Users can target **ONLY THE GROUP NODES** by using `affinity`, for example:

```
spec:

affinity:

nodeAffinity:

requiredDuringSchedulingIgnoredDuringExecution:

nodeSelectorTerms:

- matchExpressions:

- key: nautilus.io/reservation

operator: In

values:

- group1
```

For large jobs, this helps avoid consuming all shared cluster resources. Optionally, a higher priority can be used (contact the admins before using one).

#### Other taints

Some nodes in the cluster don’t have access to public Internet, and can only access educational network. They still can pull images from Docker Hub using a proxy.

If your workload is not using the public Internet resources, you might tolerate the `nautilus.io/science-dmz` and get access to additional nodes.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/special.md)

[Previous  
GatewayAPI](/documentation/userdocs/running/gateway)  [Next  
Faster images download](/documentation/userdocs/running/fast-img-download)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.