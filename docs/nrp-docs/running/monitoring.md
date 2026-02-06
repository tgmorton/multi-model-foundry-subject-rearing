# Monitoring

**Source:** https://nrp.ai/documentation/userdocs/running/monitoring

# Monitoring

When you run your jobs, it’s your responsibility to make sure they are running as intended, without overrequesting the resources. Our [Grafana](https://grafana.nrp-nautilus.io/) page is a great resource to see what your jobs are doing.

To get an idea how much resources your jobs are using, go to [namespace dashboard](https://grafana.nrp-nautilus.io/d/85a562078cdf77779eaa1add43ccec1e/kubernetes-compute-resources-namespace-pods) and choose your namespace in the dropdown list on top. Your requests percentage for memory and CPU should be as close to 100% as possible. Also check the [GPU dashboard](https://grafana.nrp-nautilus.io/d/dRG9q0Ymz/k8s-compute-resources-namespace-gpus) for your namespace to make sure the utilization is above 40%, and ideally is close to 100%.

When checking the memory utilization, make sure to use the **Memory Usage (RSS)** column. The **Memory Usage** includes the disk cache, which can grow indefinitely.

You can use `time` in your yaml command to print the maximum memory utilization. The following yaml configuration will do this. Note that you will need to `apt-get install time` in your Dockerfile to use this command line tool.

```
apiVersion: batch/v1

kind: Job

metadata:

name: myname-poddescription

spec:

template:

spec:

containers:

- command: ["stdbuf", "-i0", "-o0", "-e0", "/usr/bin/time", "-v", "bash", "-c"]

args:

- >-

python -u myprocess.py

--arg1 "a"

--arg2 "b"
```

Description:

* `stdbuff -i0 -o0 -e0` Unbuffers output so that `kubectl logs --follow PODNAME` writes output immediately, without unbuffering your process may appear hung when it’s just line buffering the output.
* `/usr/bin/time` Prints the processes time and usage statistics at the end of the run, most importantly this includes maximum memory utilization. You will find this output via `kubectl logs PODNAME`.
* `bash -c` Executes the value of `args` as a single line bash command. It’s useful to do this way so that the command being executed runs the same as it does from the bash command line.
* `args: - >-` This is a directive to the yaml parser to concatenate multiple lines. This makes the command easy to read and write. The command can include pipes, and you can string together multiple commands with `;` or `&&` in the same way you do on the bash command line without any need for escape characters (thanks to passing the `args` string to `bash -c`).

#### Tensorboard

If you are training models such as neural networks, statistical models, and the like on platforms such as Python, Tensorflow, PyTorch, etc, it is common to plot real time statistics to tools such as Tensorboard. Tensorboard in particular is an excellent real time visualization tool, but requires that you launch the Tensorboard process and keep track of the log files, all of which are extra steps to deal with under cluster environments such as the NRP.

You should first activate tensorboard in the pods:

```
tensorboard --logdir=${LOG-FILE}
```

The kubectl can link you local port to the specified port of the pods:

```
kubectl port-forward ${POD_NAME} ${REMOTE-PORTNUM}:${LOCAL-PORTNUM}
```

Then the website for tensorboard can be seen in **[http://localhost:${LOCAL-PORTNUM}](http://localhost:$%7BLOCAL-PORTNUM%7D)**

#### Comet.ml

An alternative solution is to use [http://comet.ml](https://www.comet.ml), which is free for academic users, and provides a similar set of functions as Tensorboard (plus a Baysian Hyperparameter Tuning tool). Comet.ml stores everything on their website, so there are no logs to maintain or servers to run, and this makes it an easy solution to deploy on a distributed cluster like the NRP.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/monitoring.md)

[Previous  
Long idle pods](/documentation/userdocs/running/long-idle)  [Next  
Running batch jobs](/documentation/userdocs/running/jobs)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.