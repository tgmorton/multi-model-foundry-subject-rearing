# Moving Data

**Source:** https://nrp.ai/documentation/userdocs/storage/move-data

# Moving Data

## Into the cluster

If you want to send data between the cluster storage and some other storage outside (including your local computer), you have several options, and choosing one depends on the volume of your data, number of files, where you data is now (and whether there’s access to this location from the cluster).

Kubernetes pods have local addresses and are not accessible from outside, so your options are pretty much limited to either accessing the cluster storage itself (S3), or pulling the data into the pod. If you’re not using the services provided by the cluster (S3 or Nextcloud), you’ll have to mount [a persistent storage volume](/documentation/userdocs/storage/ceph/) into your pod.

### ”kubectl cp” command

The most straightforward way to copy data is using the `kubectl` tool.

```
kubectl -n my_namespace cp ~/tmp/file.dat my_super_pod:/tmp/file.dat
```

```
kubectl -n my_namespace cp my_super_pod:/tmp/file.dat ~/tmp/file.dat
```

**You should NOT use this method for any large amount of data.** The data is going through our api (management) server, which does not have a fast connection, and you’ll be affecting the cluster performance if you send more than a couple megabytes through it.

### Using S3 object storage provided by the cluster

This is the most scalable way, and you can transfer the largest volume and number of files using it. Refer to our [S3 documentation](/documentation/userdocs/storage/ceph-s3/) on how to request an account, and setup one of clients outside and inside to access the data.

### Using the Nextcloud instance

Using our [Nextcloud](/documentation/userdocs/storage/nextcloud/) provides a convenient way to sync the data from your local machine, but this way is not too scalable and fast. Also you’ll still have to copy the data to the pod from Nextcloud to use it. (The page provides a setup example for rclone which uses the Nextcloud WebDAV interface).

### Pulling data from inside the pod

If your data is located in a storage which can be accessed from outside (any cloud provider, a server, etc.), it might be easier to pull the data into your pod. Depending on the data size, you can either:

* Run an [idle pod](/documentation/userdocs/running/long-idle/), `kubectl exec` into it and manually run a command. You’ll have to set up the credentials to access the remote storage.
* Or, run a [batch job](/documentation/userdocs/running/jobs/) which will do this for you. In this case the pod should have credentials set up at the time it starts. This should be better for large datasets, since you don’t have to keep your shell open, and will auto restart if the pod is killed for some reason.

The tools you can use include [scp](https://en.wikipedia.org/wiki/Secure_copy) (needs you to set up an ssh key or type the password by hand), [rclone](https://rclone.org/) (supports MANY data storages, and you can copy the config file generated locally), wget/curl (for pulling data from HTTP servers), any other tool for accessing your dataset you might find.

#### Using secrets

If you need to provide credentials to your data puller as a file, the best way is to use the [kubernetes Secret](https://kubernetes.io/docs/concepts/configuration/secret/). Create a secret in your namespace from a file:

```
kubectl create -n my_namespace secret generic my-secret --from-file=secret.key
```

Check it’s created:

```
kubectl get -n my_namespace secrets
```

Then mount one to your pod as a folder:

```
volumeMounts:

- mountPath: /secrets

name: sec

volumes:

- name: sec

secret:

secretName: my-secret
```

Or use as a file:

```
volumeMounts:

- mountPath: /secrets/secret.key

subPath: secret.key

name: sec

volumes:

- name: sec

secret:

secretName: my-secret
```

Or even use as an environmental variable:

```
kubectl create -n my_namespace secret generic my-secret --from-literal=pass=my-super-pass
```

```
- env:

- name: DB_PASS

valueFrom:

secretKeyRef:

key: pass

name: my-secret
```

## Inside the cluster

To copy the data from one volume to another (which can be located in a different region), you can use a pod (interactively) or a job (in batch mode).

Here’s the example of a job using the [gsutil](gitlab-registry.nrp-nautilus.io/prp/gsutil:latest) image to copy data in parallel between two mounted PVCs:

```
apiVersion: batch/v1

kind: Job

metadata:

name: copy-data

spec:

template:

spec:

containers:

- image: gitlab-registry.nrp-nautilus.io/prp/gsutil:latest

command:

- bash

- -c

- "gsutil -m rsync -e -r -P /from /to/"

imagePullPolicy: Always

name: backup

volumeMounts:

- mountPath: /from

name: source

- mountPath: /to

name: target

resources:

limits:

cpu: "4"

memory: 4G

requests:

cpu: "4"

memory: 4G

nodeSelector:

topology.kubernetes.io/region: us-central

restartPolicy: Never

volumes:

- name: target

persistentVolumeClaim:

claimName: <claim to>

- name: source

persistentVolumeClaim:

claimName: <claim from>

readOnly: true

backoffLimit: 1
```

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/storage/move-data.md)

[Previous  
Mounting FUSE](/documentation/userdocs/storage/fuse)  [Next  
Purging](/documentation/userdocs/storage/purging)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.