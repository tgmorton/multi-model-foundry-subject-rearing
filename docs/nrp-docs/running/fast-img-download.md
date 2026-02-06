# Faster Image Download

**Source:** https://nrp.ai/documentation/userdocs/running/fast-img-download

# Faster Image Download

## Spegel image cache

We’re currently using [spegel](https://github.com/XenitAB/spegel) on most nodes running containerd. It currently does not support CRI-O (several nodes in the cluster).

Spegel allows all nodes act as a caching proxy for images, which speeds up images downloads a lot.

**UBER Kraken is now deprecated! If you used “localhost:30081” URLs, please switch back to normal ones.**

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/fast-img-download.md)

[Previous  
Special use](/documentation/userdocs/running/special)  [Next  
Globus-connect](/documentation/userdocs/running/globus-connect)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.