# Deploying Coder

**Source:** https://nrp.ai/documentation/userdocs/coder/deploy

# Deploying Coder

# Guide to Deploying Your Own Coder Instance on Nautilus

This guide walks you through deploying a Coder instance in your own namespace on Nautilus, where you can have admin control, customize templates, and manage users. The instructions here are based on Coder’s official [Kubernetes installation guide](https://coder.com/docs/install/kubernetes) with some adjustments for Nautilus.

NOTE: You will need to request an exception for your namespace from admins for the gatekeeper ingress wildcard rule, and the deployment run time (otherwise your coder deployment will be killed after two weeks)

### 1. Choose Your Project Name

Your project will be accessible at `https://your_name.nrp-nautilus.io`.

---

### 2. Create the Namespace

Log into the [Nautilus portal](https://nrp.ai/namespaces) to create a namespace for your project and ensure it’s annotated with any required information.

---

### 3. Install Helm and Download the Coder Helm Chart

If Helm is not already installed, follow the [official Helm installation guide](https://helm.sh/docs/intro/install/) to set it up.

---

### 4. Set Up PostgreSQL

Create a PostgreSQL instance in your namespace by following the [PostgreSQL setup guide](https://nrp.ai/documentation/userdocs/running/postgres/).

* **Requirements**:
  + Namespace: Your project namespace
  + Username: `coder`
  + Database: `coder`
  + Storage: 10Gi

---

### 5. Create the PostgreSQL Secret

After PostgreSQL setup, create a Kubernetes secret for Coder’s database connection.

Replace `my-postgres-cluster.default.svc.cluster.local` with the endpoint for your PostgreSQL instance.

Terminal window

```
kubectl create secret generic coder-db-url -n your_namespace \

--from-literal=url="postgres://coder:[email protected]:5432/coder?sslmode=disable"
```

---

### 6. Install Coder with Helm

Add the Coder Helm repository:

Terminal window

```
helm repo add coder-v2 https://helm.coder.com/v2
```

---

### 7. Configure Your Deployment (values.yaml)

Create a `values.yaml` file with the configuration settings:

```
coder:

env:

- name: CODER_ACCESS_URL

value: https://your_name.nrp-nautilus.io

- name: CODER_WILDCARD_ACCESS_URL

value: '*.your_name.nrp-nautilus.io'

- name: CODER_PG_CONNECTION_URL

valueFrom:

secretKeyRef:

key: url

name: coder-db-url

#  - name: CODER_OIDC_ISSUER_URL

#    value: https://cilogon.org

#  - name: CODER_OIDC_CLIENT_ID

#    valueFrom:

#      secretKeyRef:

#        key: client

#        name: coder-cilogon

#  - name: CODER_OIDC_CLIENT_SECRET

#    valueFrom:

#      secretKeyRef:

#        key: secret

#        name: coder-cilogon

#  - name: CODER_GITAUTH_0_ID

#    value: gitlab

#  - name: CODER_GITAUTH_0_TYPE

#    value: gitlab

#  - name: CODER_GITAUTH_0_CLIENT_ID

#    valueFrom:

#      secretKeyRef:

#        key: client

#        name: coder-gitlab

#  - name: CODER_GITAUTH_0_CLIENT_SECRET

#    valueFrom:

#      secretKeyRef:

#        key: secret

#        name: coder-gitlab

#  - name: CODER_GITAUTH_0_AUTH_URL

#    value: https://gitlab.nrp-nautilus.io/oauth/authorize

#  - name: CODER_GITAUTH_0_TOKEN_URL

#    value: https://gitlab.nrp-nautilus.io/oauth/token

#  - name: CODER_GITAUTH_0_VALIDATE_URL

#    value: https://gitlab.nrp-nautilus.io/oauth/token/info

#  - name: CODER_OIDC_ALLOW_SIGNUPS

#    value: "false"

ingress:

className: haproxy

enable: true

host: your_name.nrp-nautilus.io

tls:

enable: true

secretName: ssl-key

wildcardSecretName: ssl-key

wildcardHost: '*.your_name.nrp-nautilus.io'

resources:

limits:

cpu: 6

memory: 4Gi

requests:

cpu: 6

memory: 4Gi

service:

sessionAffinity: None

type: ClusterIP
```

For OAuth integration, uncomment and configure the OAuth settings as needed.

---

### 8. Deploy Coder

Use Helm to deploy Coder:

Terminal window

```
helm install coder coder-v2/coder \

--namespace your_namespace \

--values values.yaml \

--version 2.15.1
```

---

### 9. Log in to Coder 🎉

Visit `https://your_name.nrp-nautilus.io` to access your Coder instance.

---

### 10. Add Custom Templates

You can add your own templates or use the examples we have at [NRP Coder Templates](https://gitlab.nrp-nautilus.io/prp/coder-templates).

---

You’re all set to use your customized Coder instance!

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/coder/deploy.md)

[Previous  
Using Coder](/documentation/userdocs/coder/coder)  [Next  
NRP Managed LLM](/documentation/userdocs/ai/llm-managed)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.