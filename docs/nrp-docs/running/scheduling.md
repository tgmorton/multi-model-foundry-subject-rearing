# Scheduling

**Source:** https://nrp.ai/documentation/userdocs/running/scheduling

# Scheduling

Caution

Current support for everything on this page is experimental. We might change it at anytime. We welcome friendly users to test this function and would like feedback.

Danger

The content of the jobs (yaml file) will be initially visible to all users. It’s not possible to limit it by namespace. We will probably limit access to lookout in the future or implement the limited visibility per group.

## Introduction

[Armada](http://armadaproject.io/) multi-Kubernetes cluster batch job meta-scheduler designed to handle massive-scale workloads. Built on top of Kubernetes, Armada enables organizations to distribute millions of batch jobs per day across tens of thousands of nodes spanning multiple clusters, making it an ideal solution for high-throughput computational workloads.

## Installation

To start using Armada [download the client application](https://media.nrp.ai/etc/armadactl) for Arm MacOS. Linux and Windows versions will be released soon. You can build one from [the armadactl-cache-token merge request source](https://github.com/dejanzele/armada/tree/feat/armadactl-cache-token) until the feature request is merged into the project.

[Download Armadactl for MacOS/arm](https://media.nrp.ai/etc/armadactl)

Put it in your PATH folder (f.e. `~/bin/`) and make runnable (`chmod +x ~/bin/armadactl`)

Save the `.armadactl.yaml` config file in your home folder:

[Download .armadactl.yaml config file](/.armadactl.yaml)

You’re ready to use armada.

## Usage

Start from listing the queues:

Terminal window

```
armadactl get queues
```

It will list all queues matching the list of all namespaces in the cluster. You can submit to the ones you normally access in the cluster.

Submit a job:

test-job.yaml

```
queue: <your_namespace>

jobSetId: job-set

jobs:

- namespace: <your_namespace>

priorityClassName: armada-default

podSpec:

terminationGracePeriodSeconds: 0

restartPolicy: Never

containers:

- name: tester

image: alpine:latest

command:

- sh

args:

- -c

- echo $(( (RANDOM % 60) + 10 ))

resources:

limits:

memory: 128Mi

cpu: 2

requests:

memory: 128Mi

cpu: 2
```

Terminal window

```
armadactl submit test-job.yaml
```

Login to the [lookout page](https://armada-lookout.nrp-nautilus.io) and see the status of your job. Also you should see the pod running in your namespace.

### Priorities

Allowed priorities in armada:

| priorityClass | preemptible | value |
| --- | --- | --- |
| armada-default | false | 100 |
| armada-preemptible | true | 50 |
| armada-high-priority | false | 1000 |

Danger

Preempted jobs will **not** be automatically resheduled.

Danger

Preemption is only working inside armada. Once the pods are in k8s cluster, those won’t be preempted by normal pods or preempt other cluster pods.

## Python client

To use python client, you can get the OIDC token from the [lookout page](https://armada-lookout.nrp-nautilus.io) in web browser. After logging in, check the site storage developer tab. There will be the id\_token stored for authentik. It’s good for 30 minutes.

Example of the python client:

Terminal window

```
pip install armada-client
```

```
import os

import uuid

import grpc

from armada_client.client import ArmadaClient

from armada_client.k8s.io.api.core.v1 import generated_pb2 as core_v1

from armada_client.k8s.io.apimachinery.pkg.api.resource import (

generated_pb2 as api_resource,

)

def create_dummy_job(client: ArmadaClient):

"""

Create a dummy job with a single container.

"""

pod = core_v1.PodSpec(

containers=[

core_v1.Container(

name="container1",

image="index.docker.io/library/ubuntu:latest",

args=["sleep", "10s"],

securityContext=core_v1.SecurityContext(runAsUser=1000),

resources=core_v1.ResourceRequirements(

requests={

"cpu": api_resource.Quantity(string="120m"),

"memory": api_resource.Quantity(string="510Mi"),

},

limits={

"cpu": api_resource.Quantity(string="120m"),

"memory": api_resource.Quantity(string="510Mi"),

},

),

)

],

)

return [client.create_job_request_item(priority=0, pod_spec=pod, namespace="your_namespace")]

token = "your_id_token"

HOST = "armada.nrp-nautilus.io"

PORT = "50051"

queue = f"your_namespace"

job_set_id = f"simple-jobset-{uuid.uuid1()}"

class BearerAuth(grpc.AuthMetadataPlugin):

def __call__(self, context, callback):

callback((("authorization", f"Bearer {token}"),), None)

channel = grpc.secure_channel(f"{HOST}:{PORT}",

grpc.composite_channel_credentials(

grpc.ssl_channel_credentials(),

grpc.metadata_call_credentials(BearerAuth())

)

)

client = ArmadaClient(channel)

job_request_items = create_dummy_job(client)

client.submit_jobs(

queue=queue, job_set_id=job_set_id, job_request_items=job_request_items

)

print("Completed Workflow")
```

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/scheduling.mdx)

[Previous  
Running CPU only jobs](/documentation/userdocs/running/cpu-only)  [Next  
Client scripts](/documentation/userdocs/running/scripts)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.