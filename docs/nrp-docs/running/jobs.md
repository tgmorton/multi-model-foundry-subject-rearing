# Running Batch Jobs

**Source:** https://nrp.ai/documentation/userdocs/running/jobs

# Running Batch Jobs

We highly recommend using [Jobs](https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/) for any kind of development and computations in our cluster. This will ensure you never lose your work, get the results in the most convenient way, and don’t waste resources, since this method does not require any babysitting of processes from you. Once your development is done, you are immediately ready to run a large-scale stuff with no changes to the code and minimal changes in the definition, plus your changes are saved in Git.

Danger

Since jobs in Nautilus can run forever, you can only run jobs with meaningful `command` field. Running in interactive mode (`sleep infinity` command and manual start of computation) or any command that doesn’t end by itself is prohibited, and user **can be banned**.

Use [our tutorial](/documentation/userdocs/tutorial/jobs/) for a simple job example.

#### Pulling code from GIT

You can put your code to our  [GitLab](https://gitlab.nrp-nautilus.io) repository and pull it from there using the following example. To modify your code between iterations you can use the [Web IDE](https://docs.gitlab.com/ce/user/project/repository/web_editor.html) - simply click the *Web IDE* button on your project’s repository page once you are logged in on Gitlab. Make sure your repo is not private or use the [private repo example](#private-repo).

```
apiVersion: batch/v1

kind: Job

metadata:

name: myapp

spec:

template:

spec:

containers:

- name: demo

image: gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp

command:

- "python"

args:

- "/opt/repo/REPONAME/my_script.py"

- "arg_job_to_run"

volumeMounts:

- name: git-repo

mountPath: /opt/repo

resources:

limits:

memory: 6Gi

cpu: "6"

nvidia.com/gpu: "1"

requests:

memory: 4Gi

cpu: "1"

nvidia.com/gpu: "1"

initContainers:

- name: init-clone-repo

image: alpine/git

args:

- clone

- --single-branch

- https://gitlab.nrp-nautilus.io/USERNAME/REPONAME

- /opt/repo/REPONAME

volumeMounts:

- name: git-repo

mountPath: /opt/repo

volumes:

- name: git-repo

emptyDir: {}

restartPolicy: Never

backoffLimit: 5
```

Two containers, *init-clone-repo* and *demo*, share the initially empty storage volume.

This pod will:

* start initContainer, pull your code from Git repository and put it in /opt/repo/REPONAME
* then will start your main container, and execute the script that was downloaded from the git repo
* when script is finished, terminate the whole pod.

#### Running several bash commands

You can group several commands, and use pipes, like this:

```
command:

- sh

- -c

- "cd /home/user/my_folder && apt-get install -y wget && wget pull some_file && do something else"
```

#### Logs

All stdout and stderr output from the script will be preserved and accessible by running

```
kubectl logs pod_name
```

Output from initContainer can be seen with

```
kubectl logs pod_name -c init-clone-repo
```

To see logs in real time do:

```
kubectl logs -f pod_name
```

The pod will remain in Completed state until you delete it or timeout is passed.

#### Retries

The backoffLimit field specifies how many times your pod will run in case the exit status of your script is not 0 or if pod was terminated for a different reason (for example a node was rebooted). It’s a good idea to have it more than 0.

#### Fair queueing

There is no fair queue implemented on Nautilus. If you submit 1000 jobs, you block **all** other users from submitting in the cluster.

To limit your submission to a fair portion of the cluster, refer to [this guide](https://kubernetes.io/docs/tasks/job/fine-parallel-processing-work-queue/). Make sure to use a deployment and persistent storage for Redis pod. Here’s [our example](https://gitlab.nrp-nautilus.io/prp/job-queue/-/blob/master/redis.yaml)

#### Private repo

If your repository is **private**, you should create a Gitlab Personal Access Token (see [Access Tokens](https://docs.gitlab.com/ee/user/profile/personal_access_tokens.html) and [Access Tokens for command line](https://help.github.com/en/articles/creating-a-personal-access-token-for-the-command-line)) of type *read\_repository*, and put it in your namespace secret to be consumed by the pod:

```
kubectl create secret generic gitlab-secret --from-literal=user=USERNAME --from-literal=password=TOKEN
```

The pod yaml file references the secret via `user` and `password` keys (must match secret literal definitions in the above command:

```
apiVersion: batch/v1

kind: Job

metadata:

name: myapp

spec:

template:

spec:

containers:

- name: demo

image: gitlab-registry.nrp-nautilus.io/prp/jupyter-stack/prp

command:

- "python"

args:

- "/opt/repo/REPONAME/my_script.py"

- "arg_job_to_run"

volumeMounts:

- name: git-repo

mountPath: /opt/repo

resources:

limits:

memory: 6Gi

cpu: "6"

nvidia.com/gpu: "1"

requests:

memory: 4Gi

cpu: "1"

nvidia.com/gpu: "1"

initContainers:

- name: init-clone-repo

image: alpine/git

env:

- name: GIT_USERNAME

valueFrom:

secretKeyRef:

name: gitlab-secret

key: user

- name: GIT_PASSWORD

valueFrom:

secretKeyRef:

name: gitlab-secret

key: password

args:

- clone

- --single-branch

- https://$(GIT_USERNAME):$(GIT_PASSWORD)@gitlab.nrp-nautilus.io/USERNAME/REPONAME

- /opt/repo/REPONAME

volumeMounts:

- name: git-repo

mountPath: /opt/repo

volumes:

- name: git-repo

emptyDir: {}

restartPolicy: Never

backoffLimit: 5
```

You can use several Work Queue Brokers, like Redis or RabbitMQ, to distribute tasks once you’re ready to scale out your computation.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/jobs.mdx)

[Previous  
Monitoring](/documentation/userdocs/running/monitoring)  [Next  
Running CPU only jobs](/documentation/userdocs/running/cpu-only)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.