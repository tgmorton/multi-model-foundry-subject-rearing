# Asking Effective Questions in Support Chat

**Source:** https://nrp.ai/documentation/userdocs/start/support

# Asking Effective Questions in Support Chat

To ensure that our support team can provide you with the most accurate and efficient assistance, we’ve outlined some guidelines for asking effective questions in our support chat.

To reach out to us please use the [matrix chat registration guide](/contact).

## Before You Ask

Before reaching out to us, please take a moment to gather the necessary information about the issue you’re experiencing. This will help us quickly identify the problem and provide a more accurate solution.

## Essential Information to Provide

When asking a question in support chat, please provide the following essential information:

* **Namespace and Pod Name**: If the issue is related to a specific pod in your cluster, please specify the namespace and pod name. This information helps us pinpoint the exact resource that’s experiencing the problem.
* **Minimal Reproducible Example**: If the issue is not specific to a particular pod or namespace, please provide a minimal reproducible example (MRE) that demonstrates the problem. MREs should be a concise, self-contained code snippet that reproduces the issue.

## Gathering Information with kubectl

To help you gather more information about the issue, we recommend using the `kubectl` command-line tool. Here are some tips on how to use `kubectl` to check pod logs and status:

### Checking Pod Status

To check the status of a pod, use the following command:

Terminal window

```
kubectl describe pod <pod-name> -n <namespace>
```

This command will provide you with a detailed overview of the pod’s status, including:

* Pod events
* Container status
* Volumes and mount points

Look for any error messages or unusual conditions that may indicate the cause of the issue.

### Checking Pod Logs

To check the logs of a pod, use the following command:

Terminal window

```
kubectl logs <pod-name> -n <namespace> -c <container-name>
```

Replace `<container-name>` with the name of the container you want to inspect. If the pod has only one container, you can omit the `-c` flag.

You can also use the following flags to customize the log output:

* `-f` to follow the log output in real-time
* `--tail` to specify the number of lines to display from the end of the log
* `--since` to specify the timestamp from which to display logs

For example:

Terminal window

```
kubectl logs <pod-name> -n <namespace> -c <container-name> -f --tail 100
```

This command will display the last 100 lines of the log output from the specified container and follow the log output in real-time.

## What to Expect Without Essential Information

If you don’t provide the namespace and pod name or a minimal reproducible example, our support team can only provide a best guess based on incomplete information. While we’ll do our best to help, we may not be able to provide an accurate diagnosis or solution. This can lead to unnecessary back-and-forth, resulting in delays and inefficiencies.

## Examples of Good Questions

Here are some examples of well-crafted questions that include the necessary information:

* “I’m seeing an error in my pod `my-pod` in the `default` namespace. The logs indicate a connection issue. Can you help me troubleshoot?”
* “I’m experiencing a problem with my deployment. Here’s a minimal reproducible example that demonstrates the issue: [insert code snippet]. Can you help me identify the root cause?”
* “I’ve checked the pod logs and status using `kubectl describe pod` and `kubectl logs`, but I’m still unsure what’s causing the issue. Here’s the output from both commands: [insert output]. Can you help me interpret the results?”

## Tips for Asking Questions

To ensure that your question is answered efficiently, follow these additional tips:

* Be concise and clear in your question.
* Provide any relevant error messages or logs.
* Use proper formatting to make your question easy to read. For example, use tripple backticks ``` around your code snippets like so:

```
```

import psycopg2

conn = psycopg2.connect("dbname=mydatabase user=myapp password=mypassword")

cur = conn.cursor()

cur.execute("SELECT version();")

print(cur.fetchone())

cur.close()

conn.close()

```
```

Note

By following these guidelines, you’ll be able to ask effective questions that help our support team provide you with accurate and efficient solutions. Thank you for helping us help you!

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/start/support.md)

[Previous  
FAQ](/documentation/userdocs/start/faq)  [Next  
Introduction](/documentation/userdocs/tutorial/introduction)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.