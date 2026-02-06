# Scientific Images

**Source:** https://nrp.ai/documentation/userdocs/running/sci-img

# Scientific Images

This page is mostly related to our [Official JupyterHub](https://jupyterhub-west.nrp-nautilus.io), but all images can be also used in other pods deployed directly on the cluster.

There are two main projects providing the stack of images:

[Docker Stack](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html)

[B-Data](https://gitlab.b-data.ch/jupyterlab)

Both projects now support CUDA and ARM. Docker stack only has CUDA in TensorFlow and PyTorch variants.

**The list of NRP-provided images and registry links are available in our [GitLab registry](https://gitlab.nrp-nautilus.io/nrp/scientific-images)**:

The **NRP image with additional libraries** (gitlab-registry.nrp-nautilus.io/nrp/scientific-images/python) is based on Docker Stack TensorFlow and PyTorch, with additional packages.

NOTE: If you are using the VS Code editor with a Jupyter Notebook, use the `base` Python Environment as the kernel. Moreover, do not use the `/usr/bin/python`/`/usr/bin/python3` Python executable. Instead, use `python` or `python3` without the path.

Refer to <https://gitlab.nrp-nautilus.io/nrp/scientific-images/python> for the list of packages installed.

We can add more libraries to this image by requesting in [Matrix](/contact).

The Desktop image has the X11 Window system installed and you can launch the GUI interface in Jupyter with this image. It’s based on the Minimal stack.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/sci-img.md)

[Previous  
GUI Desktop](/documentation/userdocs/running/gui-desktop)  [Next  
Postgres cluster](/documentation/userdocs/running/postgres)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.