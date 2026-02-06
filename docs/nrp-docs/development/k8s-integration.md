# K8s Gitlab Integration

**Source:** https://nrp.ai/documentation/userdocs/development/k8s-integration

# K8s Gitlab Integration

This page covers integrating GitLab with the Nautilus cluster to automatically deploy from GitLab to Kubernetes via CI/CD jobs.

1. In your project, go to `Operate -> Kubernetes clusters`, click the dropdown in the top right and select `Connect a cluster (certificate - deprecated)`
2. In the namespace create a GitLab service account: `kubectl create sa gitlab -n <your_namespace>`
3. Create the rolebinding for the service account:

   ```
   kubectl create -f - << EOF

   apiVersion: rbac.authorization.k8s.io/v1

   kind: RoleBinding

   metadata:

   name: gitlab

   namespace: <your_namespace>

   roleRef:

   apiGroup: rbac.authorization.k8s.io

   kind: ClusterRole

   name: admin

   subjects:

   - kind: ServiceAccount

   name: gitlab

   namespace: <your_namespace>

   EOF
   ```
4. Create a secret for the service account:

   ```
   kubectl -n <your_namespace> apply -f - << EOF

   apiVersion: v1

   kind: Secret

   metadata:

   name: gitlab-secret

   annotations:

   kubernetes.io/service-account.name: gitlab

   type: kubernetes.io/service-account-token

   EOF
   ```
5. Get the secret and Certificate Authority (CA) for the service account:

   `kubectl get secret -n your_namespace | grep gitlab`

   `kubectl get secret -n your_namespace <gitlab-secret-...> -o yaml`

   `echo <the token value> | base64 -d` - this will give you the service token field value

   `echo <the CA value> | base64 -d` - CA

   API URL - get from your cluster config file (`https://67.58.53.148:443`)
6. Uncheck `GitLab-managed cluster`, enter the namespace into `Project namespace prefix (optional, unique)`
7. Click `Add kubernetes cluster`

Now your cluster config will be available to tools like `kubectl` and `helm` to access your namespace. You can use [this project](https://gitlab.nrp-nautilus.io/prp/jupyterlab-west) as an example of how to automatically deploy a Helm application to your namespace and [this one](https://gitlab.nrp-nautilus.io/prp/nautilus-admission/-/blob/master/.gitlab-ci.yml#L28-39) to automatically update the deployment image.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/development/k8s-integration.md)

[Previous  
Private Repos](/documentation/userdocs/development/private-repos)  [Next  
Networking](/documentation/admindocs/participating/network)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.