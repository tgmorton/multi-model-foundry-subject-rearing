# Python k8s API

**Source:** https://nrp.ai/documentation/userdocs/running/kubernetes

# Python k8s API

Kubernetes offers API clients that allow you to build integrations under-the-hood of applications. The following is an example Python client for Nautilus that can submit it’s own jobs and all the other functions of `kubectl` The config mounts the Ceph Shared FileSystem as well as the node’s local scratch space. Be sure to configure resources appropriately to your application.

```
import os

from kubernetes import client, config, utils

from kubernetes.client.rest import ApiException

from skimage.transform import resize

from uuid import uuid4

import time

from pathlib import Path

def test():

try:

config.load_kube_config()

v1 = client.CoreV1Api()

ret = v1.list_namespaced_pod('ncmir-mm')

except:

ValueError('Could not connect to kube cluster')

class Constants(object):

NAMESPACE = YOUR_NAMESPACE

class KubernetesApiClient(object):

def __init__(self):

# load

print("\n Loading Nautilus Client... \n")

def create_batch_api_client(self):

return client.BatchV1Api(client.ApiClient())

def create_job_object(self, job_name, container_image, args=[],cmd = ['/bin/bash'],

min_cpu=1, min_ram = 4, max_cpu=2, max_ram=12):

res = client.V1ResourceRequirements(

requests={"cpu":"1","memory":"8Gi","ephemeral-storage": "4Gi"},

limits = {"cpu":"4","memory":"24Gi","ephemeral-storage": "16Gi"})

volume_mount_2 = client.V1VolumeMount(

mount_path='/ceph',

name='ceph'

)

volume_mount_1 = client.V1VolumeMount(

mount_path='/mnt/data',

name='data'

)

#env = client.V1EnvVar(name='GOOGLE_APPLICATION_CREDENTIALS',value=google_app_credentials_path)

container = client.V1Container(

name=job_name,

command = cmd,

image=container_image,

args=args,

volume_mounts=[volume_mount_1,volume_mount_2],

env=[],

image_pull_policy="Always",

resources = res)

volume_1 = client.V1Volume(

name='data'

)

flex_2 = client.V1FlexVolumeSource(

driver='ceph.rook.io/rook',

fs_type='ceph',

options = {'fsName': 'nautilusfs',

'clusterNamespace': 'rook',

'path': 'YOUR_CEPHFS_MOUNT',

'mountUser': 'YOUR_NAMESPACE',

'mountSecret': 'YOUR_CEPHFS_SECRET'}

)

volume_2=client.V1Volume(

name = 'ceph',

flex_volume=flex_2

)

template = client.V1PodTemplateSpec(

metadata=client.V1ObjectMeta(labels={"app": "sample"}),

spec=client.V1PodSpec(restart_policy="Never",

containers=[container],

volumes=[volume_1,volume_2])

)

spec = client.V1JobSpec(

template=template,

backoff_limit=6,

ttl_seconds_after_finished=60)

job = client.V1Job(

api_version="batch/v1",

kind="Job",

metadata=client.V1ObjectMeta(name=job_name),

spec=spec)

return job

def submit_job(jobname, image = 'YOUR_IMAGE',args = []):

api_client = KubernetesApiClient()

job_api_client = api_client.create_batch_api_client()

job = api_client.create_job_object(jobname, image, args)

try:

api_response = job_api_client.create_namespaced_job(

namespace=Constants.NAMESPACE,

body=job)

print(str(api_response.status))

except ApiException as e:

print(e) # Handle the exception.

return job_api_client
```

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/kubernetes.md)

[Previous  
Globus-connect](/documentation/userdocs/running/globus-connect)  [Next  
Federation](/documentation/userdocs/running/federation)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.