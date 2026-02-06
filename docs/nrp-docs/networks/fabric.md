# Joint Experiments with FABRIC Testbed

**Source:** https://nrp.ai/documentation/userdocs/networks/fabric

# Joint Experiments with FABRIC Testbed

# Seamless Experimentation Across FABRIC and Nautilus

## 1. Prerequisites

* FABRIC account with project access
* Nautilus cluster access through a namespace
* Basic understanding of Jupyter notebooks and FABRIC

## 2. FABRIC Setup

### 2.1 Install Dependencies

This tutorial assumes you’re running in an environment with the FABRIC fablib configuration and SSH authentication already set up.

```
from kubernetes import client, config

from kubernetes.client.rest import ApiException

from kubernetes.stream import stream

import os

import time

import random

import string
```

### 2.2 Configure FABRIC Environment

```
from fabrictestbed_extensions.fablib.fablib import FablibManager as fablib_manager

fablib = fablib_manager()

fablib.show_config();
```

```
# manually select a site

site = 'UCSD'
```

```
# generate a name for the slice

slice_name = "my-slice-" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

print(f"slice_name= {slice_name}")
```

To connect from FABRIC to Nautilus, we utilize Facility Ports, which allow us to establish a connection from the FABRIC dataplane through the UCSD site to Nautilus, specifically to the Nautilus nodes at UCSD.

* For more information on Facility Ports, visit: [FABRIC Facility Ports](https://learn.fabric-testbed.net/knowledge-base/fabric-facility-ports/)
* For more details about the Nautilus nodes, check: [Nautilus Resources](https://nrp.ai/viz/resources)

In this setup, we will be using the FABRIC facility ports **NRP-UCSD** and **FP1-UCSD**.

Additionally, we have employed the ESnet SENSE Orchestrator to establish a 1Gbps L2 path between the FABRIC UCSD site and the Nautilus node (**node-2-7.sdsc.optiputer.net**) through a map of VLAN tags listed below. In the time of writing the documentation, these paths were current. For up-to-date paths, please contact the admins via [Matrix](https://nrp.ai/contact) before reproducing.

* For more about SENSE, visit: [SENSE Orchestrator](https://sense.es.net/)

```
vlan_map = {

"3110": (3110, "FP1-UCSD"),

"3111": (3607, "FP1-UCSD"),

"3113": (3113, "FP1-UCSD"),

"3114": (3114, "FP1-UCSD"),

"3116": (3116, "FP1-UCSD"),

"3118": (3119, "FP1-UCSD"),

"3119": (3135, "FP1-UCSD"),

"3123": (3123, "NRP-UCSD"),

"3124": (3124, "NRP-UCSD"),

"3125": (3125, "NRP-UCSD"),

"3126": (3126, "NRP-UCSD"),

"3127": (3127, "NRP-UCSD"),

"3128": (3118, "NRP-UCSD"),

"3129": (3129, "NRP-UCSD")

}
```

**Please pick a VLAN tag from the above dictionary for your UCSD slice.**

If the selected VLAN tag is already in use, you will need to choose a different one.

```
vlan_choice = "3114"
```

### 2.3 Create the FABRIC UCSD Slice

```
slice = fablib.new_slice(name=slice_name)

image = 'docker_ubuntu_20'

facility_port=vlan_map[vlan_choice][1]

facility_port_site='UCSD'

facility_port_vlan=vlan_choice

node = slice.add_node(name=f"Node1", site='UCSD',cores=1, ram=8, disk=100, image=image)

node_iface = node.add_component(model='NIC_Basic', name="nic1").get_interfaces()[0]

facility_port = slice.add_facility_port(name=facility_port, site=facility_port_site, vlan=facility_port_vlan)

facility_port_interface =facility_port.get_interfaces()[0]

print(f"facility_port.get_site(): {facility_port.get_site()}")

facility_port.get_name()

net = slice.add_l2network(name=f'net_facility_port', interfaces=[])

net.add_interface(node_iface)

net.add_interface(facility_port_interface)

slice.submit()
```

```
slice = fablib.get_slice(name=slice_name)

node1 = slice.get_node(name="Node1")
```

### 2.4 Configure the Interface

```
command = "sudo ip addr add 192.168.1.1/24 dev enp7s0 && sudo ip link set enp7s0 up && ip addr show enp7s0"

stdout, stderr = node1.execute(command)
```

```
command = "sudo apt-get update && sudo apt-get install iputils-ping -y"

stdout, stderr = node1.execute(command)
```

## 3. Kubernetes Setup

### 3.1 Configure Kubernetes Client

```
namespace = 'nrp-fabric-integration'
```

```
pod_name = "my-pod-" + ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))

print(f"pod_name= {pod_name}")
```

```
# Path to your Kubernetes config file (optional)

#kube_config_path = '/path/to/your/kubeconfig'  # Adjust this to your config file path, or set to None
```

```
# Define the token (used if no config file is provided)

token = 'PLEASE GET NEW TOKEN FROM ADMINS'
```

In the event that the token doesn’t work, please contact the NRP admins via [Matrix](https://nrp.ai/contact)

```
# Ensure kube_config_path is defined before checking

kube_config_path = kube_config_path if 'kube_config_path' in locals() else None

# Check if kube_config_path exists and load configuration

if kube_config_path and os.path.exists(kube_config_path):

config.load_kube_config(config_file=kube_config_path)

print("Kubeconfig loaded successfully.")

else:

# Fallback to token if no config file is found

print("No kubeconfig found. Using token for authentication.")

configuration = client.Configuration()

configuration.host = 'https://67.58.53.148:443'  # Your server URL

configuration.api_key = {'authorization': f'Bearer {token}'}  # Use the token for authorization

configuration.verify_ssl = False

api_client = client.ApiClient(configuration)
```

### 3.2 Create Pod with VLAN Attachment

```
# Define the node you want the pod to land on

node = 'node-2-7.sdsc.optiputer.net'  # The node you want the pod to land on
```

```
# Define Node Affinity

node_affinity = client.V1NodeAffinity(

required_during_scheduling_ignored_during_execution=client.V1NodeSelector(

node_selector_terms=[

client.V1NodeSelectorTerm(

match_expressions=[

client.V1NodeSelectorRequirement(

key='kubernetes.io/hostname',  # Standard node name label

operator='In',

values=[node]  # The node you want the pod to land on

)

]

)

]

)

)

# Define the affinity

affinity = client.V1Affinity(

node_affinity=node_affinity

)
```

```
# Define the security context with network capabilities

security_context = client.V1SecurityContext(

capabilities=client.V1Capabilities(

add=["NET_RAW", "NET_ADMIN"]

)

)
```

```
# Define pod spec

pod_spec = client.V1PodSpec(

affinity=affinity,

containers=[

client.V1Container(

name='my-container',

image='alpine:latest',  # Specify the container image you want

command=["sleep", "3600"],# Just keeps the pod running for an hour

security_context=security_context

)

]

)
```

```
# Define the pod

pod = client.V1Pod(

metadata=client.V1ObjectMeta(

name=pod_name,

annotations={

"k8s.v1.cni.cncf.io/networks": f"ens-{vlan_map[vlan_choice][0]}"

}

),

spec=pod_spec

)
```

```
# Create the pod in the specified namespace

try:

if kube_config_path and os.path.exists(kube_config_path):

api_instance = client.CoreV1Api()

else:

api_instance = client.CoreV1Api(api_client)

api_response = api_instance.create_namespaced_pod(namespace=namespace, body=pod)

print(f"Pod created. Name: {api_response.metadata.name}")

except ApiException as e:

print(f"Exception when creating pod: {e}")
```

```
pod = api_instance.read_namespaced_pod(name=pod_name, namespace=namespace)
```

```
time.sleep(20)

command = ["/bin/sh", "-c", "ip addr show net1"]

# Execute the command in the pod

resp = stream(api_instance.connect_get_namespaced_pod_exec,

pod_name, namespace,

command=command,

stderr=True, stdin=False, stdout=True, tty=False)

print(resp)
```

## 4. Connectivity Test

### 4.1 Configure IP Address on Pod

```
command = [

"/bin/sh", "-c",

"ip addr flush dev net1 && ip addr add 192.168.1.2/24 dev net1 && ip addr show net1"

]

# Execute the command in the pod

resp = stream(api_instance.connect_get_namespaced_pod_exec,

pod_name, namespace,

command=command,

stderr=True, stdin=False, stdout=True, tty=False)

print(resp)
```

### 4.2 Test from Nautilus to FABRIC

```
command = [

"/bin/sh", "-c",

"ping -I net1 -c 4 192.168.1.1"

]

# Execute the command in the pod

resp = stream(api_instance.connect_get_namespaced_pod_exec,

pod_name, namespace,

command=command,

stderr=True, stdin=False, stdout=True, tty=False)

print(resp)
```

### 4.3 Test from FABRIC to Nautilus

```
command = "ping -I enp7s0 -c 4 192.168.1.2"

stdout, stderr = node1.execute(command)
```

## 5. Interactive Access

### 5.1 SSH into FABRIC Node

```
slice.get_nodes()[0].get_ssh_command()
```

### 5.2 Kubectl access to the Nautilus Pod

```
# Generate kubectl exec command with token authentication

server_url= 'https://67.58.53.148:443'

print("\nTo access the pod, run:")

print(f"""

kubectl exec -it {pod_name} -n {namespace} \\

--token={token} \\

--server={server_url} \\

--insecure-skip-tls-verify=true \\

-- /bin/sh

""")
```

## 6. Cleanup

```
# To delete the slice

slice = fablib.get_slice(name=slice_name)

slice.delete()
```

The pod has `sleep 3600`, which means it will automatically delete within an hour from starting.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/networks/fabric.mdx)

[Previous  
Vector database](/documentation/userdocs/ai/vector-database)  [Next  
Vivado and Vitis](/documentation/userdocs/fpgas/vivado-vitis)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.