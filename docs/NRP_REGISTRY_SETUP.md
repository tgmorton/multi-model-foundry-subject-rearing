# NRP Container Registry Setup for Corpus Analysis

## Overview

The corpus analysis pipeline runs as Kubernetes Jobs on NRP/Nautilus, pulling Docker
images from the NRP-hosted GitLab Container Registry at `gitlab-registry.nrp-nautilus.io`.

**Registry image path:**
```
gitlab-registry.nrp-nautilus.io/multi-model-foundry/corpus-analysis:latest
```

---

## Step-by-Step Setup

### 1. Create a GitLab Account

Register at https://gitlab.nrp-nautilus.io using your institutional credentials
(CILogon federated authentication).

### 2. Create or Join the GitLab Project

Either create a new project or get added to the existing `multi-model-foundry`
group at `https://gitlab.nrp-nautilus.io/multi-model-foundry`.

### 3. Log In to the Container Registry

```bash
docker login gitlab-registry.nrp-nautilus.io
```

Use your GitLab username and a **personal access token** (not your password).
To create a token: GitLab > User Settings > Access Tokens > create one with
`read_registry` and `write_registry` scopes.

### 4. Build the Docker Image

From the project root:

```bash
docker build -t gitlab-registry.nrp-nautilus.io/multi-model-foundry/corpus-analysis:latest \
  -f k8s/Dockerfile .
```

### 5. Push to the Registry

```bash
docker push gitlab-registry.nrp-nautilus.io/multi-model-foundry/corpus-analysis:latest
```

### 6. (If Private Repo) Create a Kubernetes Pull Secret

If the GitLab project is **private**, Kubernetes needs credentials to pull the image.

a. In GitLab, go to **Settings > Repository > Deploy Tokens** and create a
   deploy token with `read_registry` scope.

b. Create a Kubernetes secret:

```bash
kubectl create -n <your-namespace> secret docker-registry regcred \
  --docker-server=gitlab-registry.nrp-nautilus.io/multi-model-foundry/corpus-analysis \
  --docker-username=gitlab+deploy-token-XXX \
  --docker-password=<token-value>
```

c. Add `imagePullSecrets` to your Job specs:

```yaml
spec:
  template:
    spec:
      imagePullSecrets:
        - name: regcred
      containers:
        - name: analyzer
          image: gitlab-registry.nrp-nautilus.io/multi-model-foundry/corpus-analysis:latest
```

### 7. Deploy Jobs

See `k8s/README.md` for the full deployment workflow (PVC creation, data upload,
job submission, output retrieval).

---

## CI/CD Alternative: Build on Push with GitLab CI

Instead of building locally, you can automate builds using GitLab CI.
Add a `.gitlab-ci.yml` to the GitLab project:

```yaml
image: ghcr.io/osscontainertools/kaniko:debug

stages:
  - build-and-push

build-and-push-job:
  stage: build-and-push
  variables:
    GODEBUG: "http2client=0"
  script:
    - echo "{\"auths\":{\"$CI_REGISTRY\":{\"username\":\"$CI_REGISTRY_USER\",\"password\":\"$CI_REGISTRY_PASSWORD\"}}}" > /kaniko/.docker/config.json
    - /kaniko/executor --cache=true --push-retry=10 --context $CI_PROJECT_DIR --dockerfile $CI_PROJECT_DIR/k8s/Dockerfile --destination $CI_REGISTRY_IMAGE:$CI_COMMIT_SHORT_SHA --destination $CI_REGISTRY_IMAGE:latest
```

The `GODEBUG="http2client=0"` variable is required to work around a known
Kaniko speed issue with GitLab's registry.

---

## Troubleshooting

### Registry Login Failures

- Ensure you're using a **personal access token**, not your password
- Token must have `read_registry` and `write_registry` scopes
- Check the registry URL is exactly `gitlab-registry.nrp-nautilus.io`

### Image Pull Errors in Kubernetes

- **ErrImagePull / ImagePullBackOff**: Usually means missing `imagePullSecrets`
  for a private repo, or the image tag doesn't exist
- Verify the image exists: `docker pull <image-url>` from your local machine
- If the project is public, no pull secret is needed

### Push Permission Denied

- Ensure your GitLab account has at least **Developer** role in the project
- Re-authenticate: `docker logout gitlab-registry.nrp-nautilus.io && docker login gitlab-registry.nrp-nautilus.io`

---

## GitLab Container Registry Metadata Database Errors (Admin Reference)

The following errors apply to **GitLab administrators** managing the registry's
internal metadata database. Regular users will not encounter these, but they
are documented here for reference if coordinating with NRP support.

### Error: Cannot import all repositories while tags table has entries

```
ERRO[0000] cannot import all repositories while the tags table has entries,
you must truncate the table manually before retrying
```

**Cause:** Existing entries in the registry database `tags` table from a
previous failed or interrupted import attempt.

**Resolution (admin):**

1. Disable the metadata database in `/etc/gitlab/gitlab.rb`:
   ```ruby
   registry['database'] = {
     'enabled' => false,
   }
   ```
2. Connect to the registry PostgreSQL instance
3. Truncate the tags table:
   ```sql
   TRUNCATE TABLE tags RESTART IDENTITY CASCADE;
   ```
4. Re-run the import process

### Error: database-in-use lockfile exists

```
step two: import tags failed to import metadata: importing all repositories:
could not restore lockfiles: database-in-use lockfile exists
```

**Cause:** A previous import completed step two and left a lock file.

**Resolution (admin):** Delete the lock file at
`/path/to/rootdirectory/docker/registry/lockfiles/database-in-use`.
Only do this if you are certain you need to re-import.

### Error: AccessDenied (AWS S3 backend)

```
pre importing all repositories: AccessDenied: Access Denied
```

**Resolution (admin):** Ensure the IAM user/role executing the import has
the correct S3 permission scopes.

### Error: registry filesystem metadata in use

Registry won't start because the database is enabled in config but metadata
hasn't been imported yet.

**Resolution (admin):** Run the metadata import before enabling the database.

### Error: registry metadata database in use

Registry won't start because metadata was imported but the database hasn't
been enabled in config.

**Resolution (admin):** Set `registry['database'] = { 'enabled' => true }`.

### Error: permission denied for schema public (SQLSTATE 42501)

Occurs with PostgreSQL 15+ which removed default CREATE privileges on `public`.

**Resolution (admin):**
```sql
ALTER DATABASE <registry_database_name> OWNER TO <registry_user>;
```

### Storage usage not decreasing after tag deletion

The online garbage collector has a 48-hour delay before deleting unreferenced
layers. This prevents interference with in-progress image pushes.
