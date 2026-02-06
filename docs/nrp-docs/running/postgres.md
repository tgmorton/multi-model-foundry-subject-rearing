# Postgres Cluster

**Source:** https://nrp.ai/documentation/userdocs/running/postgres

# Postgres Cluster

### Using Zalando Postgres Operator in Kubernetes

This guide provides instructions for end users on how to deploy a PostgreSQL cluster using the [Zalando Postgres Operator](https://postgres-operator.readthedocs.io/en/latest/) in a Kubernetes environment.

### 1. Deploying a PostgreSQL Cluster

You can deploy your PostgreSQL cluster by creating a `Postgresql` custom resource (CR).

#### Example: `postgres-cluster.yaml`

```
apiVersion: "acid.zalan.do/v1"

kind: "postgresql"

metadata:

name: "my-postgres-cluster"

spec:

teamId: "my-team"

volume:

size: 10Gi

numberOfInstances: 3

users:

myapp:  # database users

- superuser

- createdb

databases:

mydatabase: myapp  # database name: owner

postgresql:

version: "14"  # Postgres version

resources:

requests:

cpu: "500m"

memory: "500Mi"

limits:

cpu: "1"

memory: "1Gi"
```

#### Apply the Cluster Manifest:

Terminal window

```
kubectl apply -n default -f postgres-cluster.yaml
```

**Replace “default” here and later with your actual namespace.**

This command will create a PostgreSQL cluster with the following configuration:

* 3 instances of PostgreSQL
* A database named `mydatabase` owned by `myapp` user
* 10Gi of storage per instance
* Resource requests and limits configured for each instance
* PostgreSQL version 14

### 2. Accessing the PostgreSQL Cluster

Once the cluster is running, you can connect to PostgreSQL using a Kubernetes service that is automatically created by the operator. The service will follow this format:

* **Cluster Service**: `my-postgres-cluster` (default name)
* **Primary Endpoint**: `my-postgres-cluster.default.svc.cluster.local`
* **Replica Endpoints**: `my-postgres-cluster-repl.default.svc.cluster.local`

Get password for user `postgres`:

Terminal window

```
kubectl get secret postgres.my-postgres-cluster.credentials.postgresql.acid.zalan.do -o 'jsonpath={.data.password}' | base64 -d
```

You can connect to the primary endpoint like this:

Terminal window

```
kubectl run -i --tty --rm debug --image=postgres -- bash

psql -h my-postgres-cluster -U postgres
```

The user for your created database (`mydatabase`) will have a different password stored in `myapp.my-postgres-cluster.credentials.postgresql.acid.zalan.do`.

### 3. Scaling the PostgreSQL Cluster

To scale the cluster, you can modify the `numberOfInstances` field in your `postgres-cluster.yaml` file. For example, to scale to 5 instances:

```
numberOfInstances: 5
```

Apply the changes:

Terminal window

```
kubectl apply -f postgres-cluster.yaml
```

The operator will handle scaling up or down the number of PostgreSQL instances automatically.

### 4. Monitoring the Cluster

The Zalando Postgres Operator provides built-in support for monitoring the health and performance of the PostgreSQL cluster. You can check the status of your cluster by running:

Terminal window

```
kubectl get postgresql
```

You should see output like:

Terminal window

```
NAME                TEAM   VERSION   STATUS    INSTANCES   AGE

my-postgres-cluster my-team 14       Running   3           5m
```

### 4. Choosing the storage for the cluster

Linstor storageClass is preferred as it provides the best performance for postgres. Refer to [linstor storage docs](/documentation/userdocs/storage/linstor/). Example of adding the linstor storage:

```
volume:

size: 10Gi

storageClass: linstor-igrok
```

This is how you can manage and deploy a PostgreSQL cluster using the Zalando Postgres Operator in Kubernetes. For more advanced configurations, refer to the [full documentation of the operator](https://postgres-operator.readthedocs.io/en/latest/user/).

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/postgres.md)

[Previous  
Scientific images](/documentation/userdocs/running/sci-img)  [Next  
General](/documentation/userdocs/running/virtualization-general)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.