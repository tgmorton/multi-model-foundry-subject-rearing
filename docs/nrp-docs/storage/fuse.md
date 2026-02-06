# Title

**Source:** https://nrp.ai/documentation/userdocs/storage/fuse

# Title

#### Mounting FUSE filesystem

Currently kubernetes requires elevated privileges to use FUSE mounts. The improvements are tracked in [this issue](https://github.com/kubernetes/kubernetes/issues/7890). Normally we don’t allow users to run with privileged access in the cluster, and this methoud can only be used in exceptional cases.

Here we demonstrate connecting to nautilus western S3 pool, but it can be adjusted for any FUSE mount.

1. Create the secret with your S3 credentials:

rclone.conf:

```
[nautilus_s3]

type = s3

provider = Ceph

access_key_id = <S3 key>

secret_access_key = <S3 secret>

endpoint = https://s3-west.nrp-nautilus.io
```

`kubectl create secret generic s3 --from-file=rclone.conf`

2. Minimal pod example to mount S3:

   ```
   apiVersion: v1

   kind: Pod

   metadata:

   name: fuse-pod

   spec:

   containers:

   - name: vol-container

   image: ubuntu

   command:

   - bash

   - -c

   - apt-get update && apt-get install -y vim fuse rclone curl && rclone mount nautilus_s3:<your_bucket_in_s3> /mnt

   securityContext:

   capabilities:

   add:

   - SYS_ADMIN

   resources:

   requests:

   memory: 1Gi

   cpu: "1"

   smarter-devices/fuse: "1"

   limits:

   memory: 1Gi

   cpu: "1"

   smarter-devices/fuse: "1"

   volumeMounts:

   - name: secret-volume

   mountPath: /root/.config/rclone

   volumes:

   - name: secret-volume

   secret:

   secretName: s3
   ```

The fuse device will be provided by the special kubernetes plugin via the `smarter-devices/fuse` resource request, and `SYS_ADMIN` capability is needed to make the mount.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/storage/fuse.md)

[Previous  
Syncthing](/documentation/userdocs/storage/syncthing)  [Next  
Moving data](/documentation/userdocs/storage/move-data)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.