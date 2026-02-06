# Private Repos

**Source:** https://nrp.ai/documentation/userdocs/development/private-repos

# Private Repos

Follow these steps to provide access to container images stored in the **private** [Nautilus GitLab][<https://gitlab.nrp-nautilus.io>] repository.

1. Go to your repository **Settings->Repository->Deploy Tokens**, and [create a deploy token](https://docs.gitlab.com/ce/user/project/deploy_tokens/) with **read\_registry** flag enabled.
2. Follow the instructions for [pulling image from private registry](https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/#create-a-secret-in-the-cluster-that-holds-your-authorization-token). Your registry server **your-registry-server** will be NRP’s default docker images registry FQDN, identifies as one of

   * **gitlab-registry.nrp-nautilus.io**
   * **gitlab-registry.nrp-nautilus.io/USERNAME/REPONAME**

   where **USERNAME** is your Gitlab user name and **REPONAME** is your repository.

   Terminal window

   ```
   kubectl create -n somenamespace secret docker-registry regcred --docker-server=gitlab-registry.nrp-nautilus.io/somegroup/somerepo --docker-username=gitlab+deploy-token-XXX --docker-password=XXXXXXXXXXXX
   ```

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/development/private-repos.md)

[Previous  
Building in GitLab](/documentation/userdocs/development/gitlab)  [Next  
Integrating GitLab and k8s](/documentation/userdocs/development/k8s-integration)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.