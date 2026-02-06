# Getting Started with NRP Nautilus

**Source:** https://nrp.ai/documentation/userdocs/start/getting-started

# Getting Started with NRP Nautilus

### Get access and log in

If you are a new user and want to access the NRP Nautilus cluster, follow the steps below.

New namespace management page

The namespaces management page is now at <https://nrp.ai/namespaces>. If you can’t find the namespace in the old portal, try the new one.

1. Point your browser to the [NRP Nautilus portal](https://nrp.ai).
2. On the portal page, click on the **`Log In`** button at the top right corner.
3. You will be redirected to the **`CILogon`** page, where you have to **Select an Identity Provider**.
4. Select your institution (for example: University Name) from the menu and click the **`Log On`** button.

   * If your institution is using **`CILogon`** as a federated certification authority, it will be on the menu. Select the name of your institution and use either a personal account or an institutional G-suite account. This is usually your institutional account name (email) and a password associated with it.
   * If your institution is not in the list, you may be able to use the **`Microsoft`** entry. If your university uses Microsoft’s Active Directory for user management (common if your campus uses Office 365 or related products), then you can login with your institutional credentials at the Microsoft login prompt.
   * If your institution is not in the **`CILogon`** dropdown, and Microsoft also doesn’t work, you can select **`Google`** or **`Github`**.

   Note for ORCID users

   We require email to be visible. If you are using **`ORCID`** as your institution, please change the email setting to make it visible to everyone.
5. After a successful authentication, you will log in to the portal and your account will be created if it was not created before.
6. If you are a **student**, please contact your research supervisor and ask them to add you to their [namespace](/documentation/userdocs/start/using-nautilus/#namespace). Once you are added to a namespace, your status will change to a cluster **`user`** and you will get access to all namespace resources.
7. If you are a **faculty member, researcher, or postdoc** starting a new project and need your own [namespace](/documentation/userdocs/start/using-nautilus/#namespace) — either for yourself or your research group — you can request to be promoted to a namespace **`admin`** in [Matrix](/contact). Also let us know the desired name for your group (usually the department name). As an **`admin`**, you will have the ability to:

   * Create multiple namespaces.
   * Invite other users to your namespace(s).

   Admin Responsibilities

   As a namespace **`admin`**, you will be the one responsible for

   * all activity happening in your namespaces,
   * keeping the user list up-to-date.

   We’re handling the namespaces management as a [hierarchical model](/viz/namespaces) where we delegate the management of certain parts to the admins who can then further delegate the responsibilities..

   With your request to become an admin, please let us know your current affiliation and where you see yourself fitting in this structure. You can simply own a single namespace under your university branch for small experiments or manage multiple different namespaces for large departments or groups of students.

   [Read more](/documentation/userdocs/start/hierarchy)
8. Access your namespace(s).

   * If you have become a namespace **`admin`**, you can start creating your own namespaces at this time by going to the [**Namespaces manager**](https://nrp.ai/namespaces) section on the portal. You can add other users on the same page after they have logged in to the portal.
   * If you got added to a namespace as a **`user`**, you can verify it by going to the [**Namespaces manager**](https://nrp.ai/namespaces) tab after logging in to the portal. The namespaces you’re a member of will be marked in bold.
9. **Make sure you read the cluster policies before starting to use it.**

   Read the Cluster Policies

   * Read the [**NRP Acceptable Use Policy (AUP)**](/NRP-AUP.pdf).
   * Read the [**Cluster Policies**](/documentation/userdocs/start/policies/).

### Get access and log in (deprecated)

Note

We’re in the process of transitioning to the [new authentication and namespace management system](#get-access-and-log-in).

Read previous instructions

If you are a new user and want to access the NRP Nautilus cluster, follow the steps below.

1. Point your browser to the [NRP Nautilus portal](https://nrp.ai).
2. On the portal page, click on the **`Login`** button at the top right corner.
3. You will be redirected to the **`CILogon`** page, where you have to **Select an Identity Provider**.
4. Select your institution (for example: University Name) from the menu and click the **`Log On`** button.

   * If your institution is using **`CILogon`** as a federated certification authority, it will be on the menu. Select the name of your institution and use either a personal account or an institutional G-suite account. This is usually your institutional account name (email) and a password associated with it.
   * If your institution is not using **`CILogon`**, you can select **`Google`**.

   Note for ORCID users

   We require email to be visible. If you are using **`ORCID`** as your institution, please change the email setting to make it visible to everyone.
5. After a successful authentication, you will log on to the portal with a **`guest`** status.

   Note

   On the first login, you become a **`guest`**. As a **`guest`**, you cannot access any computing resources to run your code. You have to be part of at least one namespace (usually your research group) as a **`user`** or **`admin`**.
6. If you are a **student**, please contact your research supervisor and ask them to add you to their [namespace](/documentation/userdocs/start/using-nautilus/#namespace). Once you are added to a namespace, your status will change to a cluster **`user`** and you will get access to all namespace resources.
7. If you are a **faculty member, researcher, or postdoc** starting a new project and need your own [namespace](/documentation/userdocs/start/using-nautilus/#namespace) —either for yourself or your research group—you can request to be promoted to a namespace **`admin`** in [Matrix](/contact). As an **`admin`**, you will have the ability to:

   * Create multiple namespaces.
   * Invite other users to your namespace(s).

   Admin Responsibilities

   As a namespace **`admin`**, you will be the one responsible for

   * all activity happening in your namespaces,
   * keeping the user list up-to-date.

   Removing a Namespace Admin

   To remove a namespace admin, please reach out to the cluster operators [via Matrix](https://nrp.ai/contact).
8. Access your namespace(s).

   * If you have become a namespace **`admin`**, you can start creating your own namespaces at this time by going to the [**Namespaces manager**](https://nrp.ai/namespaces) section on the portal. You can add other users on the same page after they have logged in to the portal.
   * If you got added to a namespace as a **`user`**, you can verify it by going to the [**Namespaces manager**](https://nrp.ai/namespaces) tab after logging in to the portal. You should be able to “Select your namespace” from a drop-down list.
9. **Make sure you read the cluster policies before starting to use it.**

   Read the Cluster Policies

   * Read and accept the [**NRP Acceptable Use Policy (AUP)**](/NRP-AUP.pdf).
   * Read the [**Cluster Policies**](/documentation/userdocs/start/policies/).

### Cluster access via kubectl

1. [Install](https://kubernetes.io/docs/tasks/tools/install-kubectl/) the Kubernetes command-line tool, `kubectl`.
2. [Install kubelogin](https://github.com/int128/kubelogin?tab=readme-ov-file#setup) plugin

   Required

   You **must** install the `kubelogin` plugin, or your kubeconfig file **will not work**.

   [Download kubelogin](https://github.com/int128/kubelogin?tab=readme-ov-file#setup)

   Using kubectl in WSL (Windows Subsystem for Linux)

   If you’re using **WSL**, you might run into browser launch or keyring errors during authentication with `kubelogin`. Here are fixes for common issues:

   * **Browser won’t launch or opens incorrectly**: Add  
     `--browser-command="/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"`  
     to open Chrome from Windows.
   * **Port 8000 already in use or fails to bind**: Add  
     `--listen-port=18000`  
     to use a different port.
   * **Keyring write error (e.g. `/run/user/1000/bus` not found)**: Add  
     `--token-cache-storage=disk`  
     to store the token on disk instead of using the Linux keyring.

   You can add these arguments to your config file as follows:

   Terminal window

   ```
   args:

   - oidc-login

   - get-token

   - --browser-command="/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"

   - --listen-port=18000

   - --token-cache-storage=disk

   - ...
   ```

   For more context and discussion, see [this related GitHub issue](https://github.com/int128/kubelogin/issues/535).

   Using kubectl in console with no browser

   If you only have a linux console and browser on another machine, you can use device flow. **Add these arguments** to the config file:

   Terminal window

   ```
   args:

   - oidc-login

   - get-token

   - --grant-type=device-code

   - --skip-open-browser

   - ...
   ```
3. Save the [config file](/config) as **`config`** (without any extension) and put it in your `$HOME/.kube` folder.

   [Download Config File](/config)

   This folder may not exist on your machine, to create it execute:

   Terminal window

   ```
   mkdir ~/.kube
   ```

   Finally, you should have the Nautilus Kubernetes config on your machine (laptop, desktop) as `~/.kube/config`
4. Make sure you are using the correct config file.

   * Run the following command to list available Kubernetes contexts:

     Get Contexts

     ```
     $ kubectl config get-contexts

     CURRENT   NAME       CLUSTER    AUTHINFO   NAMESPACE

     *         nautilus   nautilus   oidc
     ```
   * If you have access to multiple Kubernetes clusters, you need to choose `use-context nautilus` by doing

     Set Context

     ```
     kubectl config use-context nautilus
     ```
5. Issue the kubectl command. It will authenticate via CiLogon by opening the browser window, and close one in the end.

   ```
   kubectl get nodes
   ```
6. Verify cluster access using `kubectl`. Run the following command on your terminal.

   Get Pods

   ```
   kubectl get pods -n <YOUR_NAMESPACE>
   ```

   If you see the message “No resources found in your namespace” it means there are no pods in your namespace yet. This indicates that you have access to the resources of your namespace.
7. If you know you’re a member of a namespace, you can set it as default.

   ```
   kubectl config set contexts.nautilus.namespace <YOUR NAMESPACE>
   ```

### Updating namespace membership

By default the access token expires in half an hour and is automatically updated after that. If you need to update one sooner (for example, you were added to a new namespace), you can invalidate the token:

Invalidate access token

```
kubectl oidc-login clean
```

After that any `kubectl` command will trigget getting the new token. (F.e. `kubectl get nodes`)

Read it before using the cluster

* Containers are stateless.
  + **All your data will be gone forever when the container restarts**, unless you store it in a persistent volume.
* Container restart is normal in the Kubernates (k8s) cluster. Expect it.
* Never force Delete Pods.
* Users running a **`Job`** with **`sleep` command** or equivalent (any command that never ends by itself) will be banned from using the cluster.

### Use the Cluster

1. Read the [Using Nautilus](/documentation/userdocs/start/using-nautilus) page to learn how to use the cluster.
2. To learn more about Kubernetes you can look at [our tutorial](/documentation/userdocs/tutorial/basic/).
3. Other helpful resources:

   * [`kubectl` Tool Overview](https://kubernetes.io/docs/reference/kubectl) and [Cheatsheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
   * [Kubernetes Basics](https://kubernetes.io/docs/tutorials/kubernetes-basics/) from Kubernetes project
   * [Kubernetes Official Tutorials](https://kubernetes.io/docs/tutorials/)

   Please note that not all examples will work in our cluster because of security policies. You are limited to seeing what’s happening in your own namespace, and nobody else can see your running pods.

### GUI tools for Kubernetes

You might want to try one of these GUI tools for Kubernetes:

* [Lens](https://k8slens.dev/) - Graphical user interface
* [K9s](https://k9scli.io/) - console graphical user interface

Both will use your config file in the default location to get access to the cluster.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/start/getting-started.mdx)

[Previous  
Start Here](/documentation/)  [Next  
Using Nautilus](/documentation/userdocs/start/using-nautilus)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.