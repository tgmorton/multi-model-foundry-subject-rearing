# GUI Desktop

**Source:** https://nrp.ai/documentation/userdocs/running/gui-desktop

# GUI Desktop

**Note that Coder (<https://coder.nrp-nautilus.io>) and JupyterHub West (<https://jupyterhub-west.nrp-nautilus.io/>) are currently the preferred methods to deploy the GUI Desktop containers. Use the below instructions if you want to deploy in your own namespace.**

The default authentication credentials for Selkies in the browser are: Username: ubuntu Password: mypasswd

With [docker-nvidia-glx-desktop](https://github.com/selkies-project/docker-nvidia-glx-desktop) or [docker-nvidia-egl-desktop](https://github.com/selkies-project/docker-nvidia-egl-desktop), users may start a GUI Desktop interface accelerated with NVIDIA GPUs. Both containers support OpenGL and Vulkan.

Out of the two containers, [docker-nvidia-glx-desktop](https://github.com/selkies-project/docker-nvidia-glx-desktop) is generally recommended to be used because it provides the best performance.

However, [docker-nvidia-egl-desktop](https://github.com/selkies-project/docker-nvidia-egl-desktop) is versatile in various environments and has less processes running, meaning less possible errors. It is also possible to be used in HPC clusters with [Apptainer](https://github.com/apptainer/apptainer)/[Singularity](https://github.com/sylabs/singularity) available, and sharing a GPU with multiple containers is also possible.

Both these containers use [Selkies-GStreamer](https://github.com/selkies-project/selkies-gstreamer) to stream the remote desktop from the NRP to your web browser.

**Support is available in [#ue4research:matrix.nrp-nautilus.io](https://matrix-to.nrp-nautilus.io/#/#ue4research:matrix.nrp-nautilus.io) (use [this link](https://matrix.to/#/#ue4research:matrix.nrp-nautilus.io) if you use a matrix.org account), just ask your questions there.**

Please give the repositories a star!

## GPU-Accelerated Selkies

For GPU-accelerated desktop performance, you can modify the configurations to request NVIDIA GPUs and use hardware encoding:

### GPU Resource Request

Add GPU resources to your deployment by modifying the `resources` section:

```
resources:

limits:

memory: 64Gi

cpu: "16"

nvidia.com/gpu: 1  # Request 1 GPU

requests:

memory: 100Mi

cpu: 100m

nvidia.com/gpu: 1  # Request 1 GPU
```

### Hardware Encoding

For NVIDIA GPUs, you can use hardware-accelerated encoding by setting the encoder environment variable:

```
env:

- name: SELKIES_ENCODER

value: "nvh264enc"  # Use NVIDIA hardware encoding
```

**Note:** Hardware encoding requires:

* NVIDIA GPU with NVENC support
* Proper NVIDIA drivers installed on the node
* The `nvidia.com/gpu` resource request

### GPU vs CPU Encoding

* **`nvh264enc`**: Hardware-accelerated encoding using NVIDIA GPU (best performance, requires GPU)
* **`x264enc`**: CPU-based encoding (default, works on all nodes, moderate performance)
* **`vp8enc`/`vp9enc`**: Alternative CPU encoders with different compression ratios

### Performance Considerations

* **GPU encoding**: Lower CPU usage, better performance for high-resolution displays
* **CPU encoding**: Higher CPU usage, works on all nodes, suitable for lower-resolution displays
* **Network bandwidth**: Hardware encoding typically provides better compression and lower bandwidth usage

#### DNS Setup

Since the right TURN server closest to you leads to the lowest latency, use the command `ping turn.nrp-nautilus.io` on your client (install `iputils-ping` when using Linux if the `ping` command does not work) to check that your DNS is correctly configured.

If the latency is considerably high (over 60-70 miliseconds) or the TURN server host node location is suspiciously far from both you and the Kubernetes node, consider changing your client DNS server to `8.8.8.8` and `8.8.4.4` or the DNS over HTTPS / DNS over TLS options [provided by Google](https://developers.google.com/speed/public-dns/docs/using). [CloudFlare’s DNS](https://1.1.1.1/dns/) server (addresses `1.1.1.1` and `1.0.0.1`) may show issues locating the nearest relay server, which is crucial for performance.

If you are using the Nautilus coTURN server for other WebRTC workloads, make sure to use the TURN REST API credential endpoint `http://turn-rest.nrp-nautilus.io` (accessible only to Nautilus pods, username and password are valid for 24 hours, renewed every time `http://turn-rest.nrp-nautilus.io` is probed), and also add the below configuration (change DNS server as adequate) in your Kubernetes `.yml` configuration to minimize TURN server latency:

```
dnsPolicy: None

dnsConfig:

nameservers:

- 8.8.8.8

- 8.8.4.4
```

#### Usage

The below is a reference configuration `xgl.yml` for [docker-nvidia-glx-desktop](https://github.com/selkies-project/docker-nvidia-glx-desktop):

```
apiVersion: apps/v1

kind: Deployment

metadata:

name: xgl

spec:

replicas: 1

selector:

matchLabels:

app: xgl

template:

metadata:

labels:

app: xgl

spec:

hostname: xgl

containers:

- name: xgl

# Change tag `latest` to Ubuntu versions such as `24.04`, use a persistent tag such as `24.04-20210101010101` to persist a certain container version

image: ghcr.io/selkies-project/nvidia-glx-desktop:latest

env:

- name: TZ

value: "UTC"

- name: DISPLAY_SIZEW

value: "1920"

- name: DISPLAY_SIZEH

value: "1080"

- name: DISPLAY_REFRESH

value: "60"

- name: DISPLAY_DPI

value: "96"

- name: DISPLAY_CDEPTH

value: "24"

# With driver versions lower than 550, change to `DP-0` or any other `DP-*` port for larger resolution support if NOT using datacenter GPUs

- name: VIDEO_PORT

value: "DFP"

# Choose either `value:` or `secretKeyRef:` but not both at the same time

- name: PASSWD

#          value: "mypasswd"

valueFrom:

secretKeyRef:

name: my-pass

key: my-pass

# Uncomment to enable KasmVNC instead of Selkies-GStreamer, `SELKIES_BASIC_AUTH_PASSWORD` is used for authentication with KasmVNC, defaulting to `PASSWD` if not provided

# Uses: `SELKIES_ENABLE_BASIC_AUTH`, `SELKIES_BASIC_AUTH_USER`, `SELKIES_BASIC_AUTH_PASSWORD`, `SELKIES_ENABLE_RESIZE`, `SELKIES_ENABLE_HTTPS`, `SELKIES_HTTPS_CERT`, `SELKIES_HTTPS_KEY`

#        - name: KASMVNC_ENABLE

#          value: "true"

# Number of threads for encoding frames with KasmVNC, default value is all threads

#        - name: KASMVNC_THREADS

#          value: "0"

###

# Selkies-GStreamer parameters, for additional configurations see `selkies-gstreamer --help`

###

# Change `SELKIES_ENCODER` to `nvh264enc` for GPU-accelerated encoding, or use `vp8enc`/`vp9enc` as alternatives

- name: SELKIES_ENCODER

value: "x264enc"

# Do NOT set to `true` if physical monitor is connected to video port

- name: SELKIES_ENABLE_RESIZE

value: "false"

# Initial video bitrate in kilobits per second, may be changed later within web interface

- name: SELKIES_VIDEO_BITRATE

value: "8000"

# Initial frames per second, may be changed later within web interface

- name: SELKIES_FRAMERATE

value: "60"

# Initial audio bitrate in bits per second, may be changed later within web interface

- name: SELKIES_AUDIO_BITRATE

value: "128000"

# Uncomment if network conditions rapidly fluctuate

#        - name: SELKIES_CONGESTION_CONTROL

#          value: "true"

# Enable Basic Authentication from the web interface

- name: SELKIES_ENABLE_BASIC_AUTH

value: "true"

# Defaults to `PASSWD` if unspecified, choose either `value:` or `secretKeyRef:` but not both at the same time

#        - name: SELKIES_BASIC_AUTH_PASSWORD

#          value: "mypasswd"

#          valueFrom:

#            secretKeyRef:

#              name: my-pass

#              key: my-pass

- name: SELKIES_TURN_REST_URI

value: "http://turn-rest.nrp-nautilus.io"

# Change to `tcp` if the UDP protocol is throttled or blocked in your client network, or when the TURN server does not support UDP

- name: SELKIES_TURN_PROTOCOL

value: "udp"

# You need a valid hostname and a certificate from authorities such as ZeroSSL or Let's Encrypt with your TURN server to enable TURN over TLS (Works for NRP TURN)

- name: SELKIES_TURN_TLS

value: "false"

stdin: true

tty: true

ports:

- name: http

containerPort: 8080

protocol: TCP

resources:

limits:

memory: 64Gi

cpu: "16"

requests:

memory: 100Mi

cpu: 100m

volumeMounts:

- mountPath: /dev/shm

name: dshm

- mountPath: /cache

name: xgl-cache-vol

- mountPath: /home/ubuntu

name: xgl-root-vol

dnsPolicy: None

dnsConfig:

nameservers:

- 8.8.8.8

- 8.8.4.4

volumes:

- name: dshm

emptyDir:

medium: Memory

- name: xgl-cache-vol

emptyDir: {}

#        persistentVolumeClaim:

#          claimName: xgl-cache-vol

- name: xgl-root-vol

emptyDir: {}

#        persistentVolumeClaim:

#          claimName: xgl-root-vol

affinity:

nodeAffinity:

requiredDuringSchedulingIgnoredDuringExecution:

nodeSelectorTerms:

- matchExpressions:

- key: topology.kubernetes.io/zone

operator: NotIn

values:

- ucsd-suncave

#              - key: topology.kubernetes.io/region

#                operator: In

#                values:

#                - us-west
```

The below is a reference configuration `egl.yml` for [docker-nvidia-egl-desktop](https://github.com/selkies-project/docker-nvidia-egl-desktop):

```
apiVersion: apps/v1

kind: Deployment

metadata:

name: egl

spec:

replicas: 1

selector:

matchLabels:

app: egl

template:

metadata:

labels:

app: egl

spec:

hostname: egl

containers:

- name: egl

# Change tag `latest` to Ubuntu versions such as `24.04`, use a persistent tag such as `24.04-20210101010101` to persist a certain container version

image: ghcr.io/selkies-project/nvidia-egl-desktop:latest

env:

env:

- name: TZ

value: "UTC"

- name: DISPLAY_SIZEW

value: "1920"

- name: DISPLAY_SIZEH

value: "1080"

- name: DISPLAY_REFRESH

value: "60"

- name: DISPLAY_DPI

value: "96"

- name: DISPLAY_CDEPTH

value: "24"

# Keep to default unless you know what you are doing with VirtualGL, `VGL_DISPLAY` should be set to either `egl[n]` or `/dev/dri/card[n]` only when the device was passed to the container

#        - name: VGL_DISPLAY

#          value: "egl"

# Choose either `value:` or `secretKeyRef:` but not both at the same time

- name: PASSWD

#          value: "mypasswd"

valueFrom:

secretKeyRef:

name: my-pass

key: my-pass

# Uncomment to enable KasmVNC instead of Selkies-GStreamer, `SELKIES_BASIC_AUTH_PASSWORD` is used for authentication with KasmVNC, defaulting to `PASSWD` if not provided

# Uses: `SELKIES_ENABLE_BASIC_AUTH`, `SELKIES_BASIC_AUTH_USER`, `SELKIES_BASIC_AUTH_PASSWORD`, `SELKIES_ENABLE_RESIZE`, `SELKIES_ENABLE_HTTPS`, `SELKIES_HTTPS_CERT`, `SELKIES_HTTPS_KEY`

#        - name: KASMVNC_ENABLE

#          value: "true"

# Number of threads for encoding frames with KasmVNC, default value is all threads

#        - name: KASMVNC_THREADS

#          value: "0"

###

# Selkies-GStreamer parameters, for additional configurations see `selkies-gstreamer --help`

###

# Change `SELKIES_ENCODER` to `nvh264enc` for GPU-accelerated encoding, or use `vp8enc`/`vp9enc` as alternatives

- name: SELKIES_ENCODER

value: "x264enc"

# Do NOT set to `true` if physical monitor is connected to video port

- name: SELKIES_ENABLE_RESIZE

value: "false"

# Initial video bitrate in kilobits per second, may be changed later within web interface

- name: SELKIES_VIDEO_BITRATE

value: "8000"

# Initial frames per second, may be changed later within web interface

- name: SELKIES_FRAMERATE

value: "60"

# Initial audio bitrate in bits per second, may be changed later within web interface

- name: SELKIES_AUDIO_BITRATE

value: "128000"

# Uncomment if network conditions rapidly fluctuate

#        - name: SELKIES_CONGESTION_CONTROL

#          value: "true"

# Enable Basic Authentication from the web interface

- name: SELKIES_ENABLE_BASIC_AUTH

value: "true"

# Defaults to `PASSWD` if unspecified, choose either `value:` or `secretKeyRef:` but not both at the same time

#        - name: SELKIES_BASIC_AUTH_PASSWORD

#          value: "mypasswd"

#          valueFrom:

#            secretKeyRef:

#              name: my-pass

#              key: my-pass

- name: SELKIES_TURN_REST_URI

value: "http://turn-rest.nrp-nautilus.io"

# Change to `tcp` if the UDP protocol is throttled or blocked in your client network, or when the TURN server does not support UDP

- name: SELKIES_TURN_PROTOCOL

value: "udp"

# You need a valid hostname and a certificate from authorities such as ZeroSSL or Let's Encrypt with your TURN server to enable TURN over TLS (Works for NRP TURN)

- name: SELKIES_TURN_TLS

value: "false"

stdin: true

tty: true

ports:

- name: http

containerPort: 8080

protocol: TCP

resources:

limits:

memory: 64Gi

cpu: "16"

requests:

memory: 100Mi

cpu: 100m

volumeMounts:

- mountPath: /dev/shm

name: dshm

- mountPath: /cache

name: egl-cache-vol

- mountPath: /home/ubuntu

name: egl-root-vol

dnsPolicy: None

dnsConfig:

nameservers:

- 8.8.8.8

- 8.8.4.4

volumes:

- name: dshm

emptyDir:

medium: Memory

- name: egl-cache-vol

emptyDir: {}

#        persistentVolumeClaim:

#          claimName: egl-cache-vol

- name: egl-root-vol

emptyDir: {}

#        persistentVolumeClaim:

#          claimName: egl-root-vol

#      affinity:

#        nodeAffinity:

#          requiredDuringSchedulingIgnoredDuringExecution:

#            nodeSelectorTerms:

#            - matchExpressions:

#              - key: topology.kubernetes.io/region

#                operator: In

#                values:

#                - us-west
```

##### Customization

In one entry, `value:` and `valueFrom:` must not exist at the same time.

**Comment out `emptyDir: {}` and uncomment `persistentVolumeClaim:`, then change `claimName:` to the name of your PersistentVolumeClaim after uncommenting.**

`rook-ceph-block-[region]` or `linstor-[region]` are the recommended StorageClasses to be mounted to `/home/ubuntu`. However, if their performances are slow for sequential read/writes, you may use `rook-cephfs-[region]` but **NEVER MOUNT TO `/home/ubuntu`**. Instead mount to a different directory such as `/home/ubuntu/persistent` or `/mnt/persistent`. Refer to the [Storage](/documentation/userdocs/storage/intro/) section for more information.

**If your client network blocks (likely your campus network) or throttles (likely your home network restricted by your Internet Service Privider) the UDP protocol, change the environment variable `SELKIES_TURN_PROTOCOL` to `tcp`.**

**If your network or country enforces [Deep Packet Inspection](https://en.wikipedia.org/wiki/Deep_packet_inspection) to the WebRTC or STUN/TURN protocols, set `SELKIES_TURN_TLS` to `true`.**

Check the `README.md` of both GitHub repositories [docker-nvidia-glx-desktop](https://github.com/selkies-project/docker-nvidia-glx-desktop) or [docker-nvidia-egl-desktop](https://github.com/selkies-project/docker-nvidia-egl-desktop), as well as [Selkies-GStreamer](https://github.com/selkies-project/selkies-gstreamer), for more details on configuration customization. Please give the repositories a star too!

##### Secret Generation

Replace `YOUR_PASSWORD` with the password for the container that you intend to use. Replace `my-name` and `my-key` with the name of the secret you want to use for your password. If you want to use a different name and key, make sure to update the above `xgl.yml` or `egl.yml` files as well as the below command.

After customizing, run the below command in edited form:

Terminal window

```
kubectl create secret generic my-name --from-literal=my-key=YOUR_PASSWORD
```

The above command should be used in conjunction to the below `xgl.yml`/`egl.yml` configuration (edit as adequate):

```
env:

- name: PASSWD

valueFrom:

secretKeyRef:

name: my-name

key: my-key
```

##### Container Start

After saving and editing the reference configurations, run either of these commands to start the container depending on the type of your container:

Terminal window

```
kubectl create -f xgl.yml
```

Terminal window

```
kubectl create -f egl.yml
```

#### Exposing the Container

The below reference configuration `xgl-ingress.yml` is to expose your [docker-nvidia-glx-desktop](https://github.com/selkies-project/docker-nvidia-glx-desktop) container to the `*.nrp-nautilus.io` endpoint. Replace `YOUR_ENDPOINT` to the subdomain you want to use.

Modify the configuration as in [Scaling and exposing](/documentation/userdocs/tutorial/basic2) to customize when there are multiple desktop deployments in a namespace. You can just use `kubectl port-forward deployment/xgl 8080:8080` and access localhost:8080, but this will likely have higher latency and subpar performance.

```
apiVersion: networking.k8s.io/v1

kind: Ingress

metadata:

name: xgl

spec:

ingressClassName: haproxy

rules:

- host: YOUR_ENDPOINT.nrp-nautilus.io

http:

paths:

- backend:

service:

name: xgl

port:

name: http

path: /

pathType: ImplementationSpecific

tls:

- hosts:

- YOUR_ENDPOINT.nrp-nautilus.io

---

apiVersion: v1

kind: Service

metadata:

name: xgl

labels:

app: xgl

spec:

selector:

app: xgl

ports:

- name: http

protocol: TCP

port: 8080
```

If you are deploying multiple instances in one namespace, you must change `backend:`, `selector:`, and `labels:` to the name of your `Deployment` and `Service`.

Run the below command after saving the changed reference configuration file:

Terminal window

```
kubectl create -f xgl-ingress.yml
```

Access `YOUR_ENDPOINT.nrp-nautilus.io` with your web browser.

**The username is `ubuntu` and the password is the `my-name`/`my-key` secret that you have set.**

The below reference configuration `egl-ingress.yml` is to expose your [docker-nvidia-egl-desktop](https://github.com/selkies-project/docker-nvidia-egl-desktop) container to the `*.nrp-nautilus.io` endpoint. Replace `YOUR_ENDPOINT` to the subdomain you want to use.

Modify the configuration as in [Scaling and exposing](/documentation/userdocs/tutorial/basic2) to customize when there are multiple desktop deployments in a namespace. You can just use `kubectl port-forward deployment/egl 8080:8080` and access localhost:8080, but this will likely have higher latency and subpar performance.

```
apiVersion: networking.k8s.io/v1

kind: Ingress

metadata:

name: egl

spec:

ingressClassName: haproxy

rules:

- host: YOUR_ENDPOINT.nrp-nautilus.io

http:

paths:

- backend:

service:

name: egl

port:

name: http

path: /

pathType: ImplementationSpecific

tls:

- hosts:

- YOUR_ENDPOINT.nrp-nautilus.io

---

apiVersion: v1

kind: Service

metadata:

name: egl

labels:

app: egl

spec:

selector:

app: egl

ports:

- name: http

protocol: TCP

port: 8080
```

If you are deploying multiple instances in one namespace, you must change `backend:`, `selector:`, and `labels:` to the name of your `Deployment` and `Service`.

Run the below command after saving the changed reference configuration file:

Terminal window

```
kubectl create -f egl-ingress.yml
```

Access `YOUR_ENDPOINT.nrp-nautilus.io` with your web browser.

**The username is `ubuntu` and the password is the `my-name`/`my-key` secret that you have set.**

#### Reproducibility in Containers

Reproducibility is important, but the `latest` and `2*.04` containers are consistently updated to fix various issues that arise and add improvements.

Because some changes are radical and change the fundamental structure of the container (normally when there is a dead-end with breaking issues or breakable upstream changes), they might break for your specific workflow.

Since you would not want to see your container break one day because there was a change, you could use persistent container tags (in the format `22.04-20240101010101`) with the time which a commit was made, if you want to base this container to build a customized container.

Because the containers have rolling releases, container versions are tagged by the time of which a commit has been made.

You may access such available persistent container tags in <https://github.com/selkies-project/docker-nvidia-glx-desktop/pkgs/container/nvidia-glx-desktop> and <https://github.com/selkies-project/docker-nvidia-egl-desktop/pkgs/container/nvidia-egl-desktop>.

It is **STRONGLY** recommended that you update the container tags frequently or use `apt-get upgrade` in every new build where this container has been based (but this may also break the container in a few edge cases), as many serious security vulnerabilities are frequently present in old containers. Conversely, completely fresh builds from the `Dockerfile` with older commits may and will break at any time and thus the latest `Dockerfile` should be used.

#### Container Customization

You can import either `https://github.com/selkies-project/docker-nvidia-glx-desktop.git` or `https://github.com/selkies-project/docker-nvidia-egl-desktop.git` by importing with **Repo by URL** from <https://gitlab.nrp-nautilus.io/projects/new#import_project>.

Refer to [Building in GitLab](/documentation/userdocs/development/gitlab) on how you can change and build your own customized container (add `--build-arg="UBUNTU_RELEASE=<Ubuntu Version>"` after `/kaniko/executor` in the example `.gitlab-ci.yml` file to change the Ubuntu version).

**If you want to change the `Dockerfile`, you are recommended to use the original container as a base container and only replace the `entrypoint.sh` and `supervisord.conf` files. This will keep you up to date with the latest updates. Use persistent container tags (such as `24.04-20210101010101`) to preserve a specific container build.**

Start with the below sample `Dockerfile` example and place your modified `entrypoint.sh` and `supervisord.conf` files within the same empty directory or Git repository (switch the `FROM` line to `ghcr.io/selkies-project/nvidia-glx-desktop:${DISTRIB_RELEASE}` or `ghcr.io/selkies-project/nvidia-egl-desktop:${DISTRIB_RELEASE}`):

```
ARG DISTRIB_RELEASE=24.04

FROM ghcr.io/selkies-project/nvidia-glx-desktop:${DISTRIB_RELEASE}

ARG DISTRIB_RELEASE

USER 0

# Restore file permissions to ubuntu user

RUN if [ -d "/usr/libexec/sudo" ]; then SUDO_LIB="/usr/libexec/sudo"; else SUDO_LIB="/usr/lib/sudo"; fi && \

chown -R -f -h --no-preserve-root ubuntu:ubuntu /usr/bin/sudo-root /etc/sudo.conf /etc/sudoers /etc/sudoers.d /etc/sudo_logsrvd.conf "${SUDO_LIB}" || echo 'Failed to provide user permissions in some paths relevant to sudo'

USER 1000

# Use BUILDAH_FORMAT=docker in buildah

SHELL ["/usr/bin/fakeroot", "--", "/bin/sh", "-c"]

# Install and configure below this line

# Replace changed files

# Copy scripts and configurations used to start the container with `--chown=1000:1000`

#COPY --chown=1000:1000 entrypoint.sh /etc/entrypoint.sh

#RUN chmod -f 755 /etc/entrypoint.sh

#COPY --chown=1000:1000 selkies-gstreamer-entrypoint.sh /etc/selkies-gstreamer-entrypoint.sh

#RUN chmod -f 755 /etc/selkies-gstreamer-entrypoint.sh

#COPY --chown=1000:1000 kasmvnc-entrypoint.sh /etc/kasmvnc-entrypoint.sh

#RUN chmod -f 755 /etc/kasmvnc-entrypoint.sh

#COPY --chown=1000:1000 supervisord.conf /etc/supervisord.conf

#RUN chmod -f 755 /etc/supervisord.conf

SHELL ["/bin/sh", "-c"]

USER 0

# Enable sudo through sudo-root with uid 0

RUN if [ -d "/usr/libexec/sudo" ]; then SUDO_LIB="/usr/libexec/sudo"; else SUDO_LIB="/usr/lib/sudo"; fi && \

chown -R -f -h --no-preserve-root root:root /usr/bin/sudo-root /etc/sudo.conf /etc/sudoers /etc/sudoers.d /etc/sudo_logsrvd.conf "${SUDO_LIB}" || echo 'Failed to provide root permissions in some paths relevant to sudo' && \

chmod -f 4755 /usr/bin/sudo-root || echo 'Failed to set chmod setuid for root'

USER 1000

ENV SHELL=/bin/bash

ENV USER=ubuntu

ENV HOME=/home/ubuntu

WORKDIR /home/ubuntu

EXPOSE 8080

ENTRYPOINT ["/usr/bin/supervisord"]
```

#### Conclusion

Again, please give [docker-nvidia-glx-desktop](https://github.com/selkies-project/docker-nvidia-glx-desktop), [docker-nvidia-egl-desktop](https://github.com/selkies-project/docker-nvidia-egl-desktop), and [Selkies-GStreamer](https://github.com/selkies-project/selkies-gstreamer) a star if the containers were useful to you.

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/running/gui-desktop.md)

[Previous  
Federation](/documentation/userdocs/running/federation)  [Next  
Scientific images](/documentation/userdocs/running/sci-img)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.