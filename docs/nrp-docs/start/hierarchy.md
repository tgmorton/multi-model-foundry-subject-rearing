# How National Research Platform (NRP) Manages Resources

**Source:** https://nrp.ai/documentation/userdocs/start/hierarchy

# How National Research Platform (NRP) Manages Resources

The National Research Platform (NRP) is using a novel way for managing computational resources. This new model allows managing the team’s access and resources without needing to contact the NRP core team for every change, enabling everyone to get their work done faster and more efficiently.

Info

* The portal updates the caches for groups info every minute.
* The k8s access token expires in half an hour. If you need to update the group membership sooner, use the [oidc-login clean command](/documentation/userdocs/start/getting-started#updating-namespace-membership).

### How It Works: The New Hierarchical System

The core of this upgrade is a **hierarchical management system**. This structure organizes users and resources in a clear, nested hierarchy, much like a folder structure:

* **Organizations:** The top level, such as a university or a large research consortium.
* **Labs:** Groups within an organization, typically led by a faculty member or primary researcher.
* **Projects:** Specific research initiatives within a lab.

This structure allows designated administrators to manage permissions for their own members across all their labs and projects.

### Getting Started: Your First Login

Access to the NRP is managed through **Authentik**, our single-sign-on authentication system. It connects to your existing university or institutional account via [CILogon](https://www.cilogon.org).

1. The first time you log in, you will be required to read and accept the [NRP Acceptable Use Policy (AUP)](/NRP-AUP.pdf).
2. Once you accept the AUP, you become a registered user with access to all standard [NRP services and resources](/documentation/userdocs/start/resources/).

### Key Resources and How They’re Managed

This new hierarchical model applies to all resources you use on the platform.

#### Kubernetes Compute

Our primary service is providing compute resources via Kubernetes. Under the new system, each **Project** you create in the hierarchy directly corresponds to a **namespace** in the Kubernetes cluster. This gives you a secure, dedicated space for your team’s applications and workflows.

#### LLM Proxy

Groups can have the LLM Proxy capability, allowing creating LLM tokens in those.

#### Vector database

Groups can have the Vector DB capability, which creates the database in our managed Vector DB.

### Group Administrators

Users with **faculty**, **researcher**, or **postdoc** status can request administrator permissions for their Lab or Organization. Admins can:

* Add or remove users from their groups.
* Create new Projects (i.e., Kubernetes namespaces).
* Manage resource allocations for their teams.

### Fair Share Scheduling & Resource Allocation

The resource hierarchy will be integrated with scheduling add-ons to manage how compute jobs are prioritized and allocated.

To ensure fairness and reward contributions, groups that donate hardware to the cluster will receive an increased resource allowance. This bonus is automatically passed down to all the labs and projects within that group, ensuring your team directly benefits from your contributions on top of the base allocation available to everyone.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/start/hierarchy.md)

[Previous  
Using Nautilus](/documentation/userdocs/start/using-nautilus)  [Next  
Cluster Policies](/documentation/userdocs/start/policies)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.