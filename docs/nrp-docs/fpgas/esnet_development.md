# ESnet SmartNIC FPGA - Development

**Source:** https://nrp.ai/documentation/userdocs/fpgas/esnet_development

# ESnet SmartNIC FPGA - Development

**For video recording, please see: [youtube.](https://youtu.be/tdxe0jsRZVk)**

**For the ipynb notebooks and slides, please see: [the tutorial respository.](https://github.com/nrp-nautilus/esnet-smartnic)**

## Development (Notebook 1/3): Writing and Testing a P4 Program

This notebook is **Part 1** of the **ESnet SmartNIC Tutorial on NRP** series. It provides an example of cloning the `esnet-smartnic-hw` repository, and writing and testing a simple P4 program.

---

### Test Environment

This notebook was tested on the **National Research Platform (NRP)** using the **AMD/Xilinx Alveo U55C FPGA** and **Vivado 2023.1**. The Kubernetes pods were provisioned by [Coder](https://coder.nrp-nautilus.io).

This tutorial is built on the following software/respositories along with versions/commits:

```
Ubuntu 22.04 with Linux 5.15.0-153.

Vivado 2023.1 with the VitisNetowrkingP4 license.

The esnet-smartnic-hw repository.

The esnet-smartnic-fw repository.

The smartnic-dpdk-docker repository.

The xilinx-labtools-docker repository.
```

If you run into any issues, please refer to the official [NRP Documentation](https://docs.nrp.ai), or reach out to us via [Matrix](https://element.nrp-nautilus.io) or [email](/cdn-cgi/l/email-protection#ccbfb9bcbca3beb88ca2adb8a5a3a2ada0bea9bfa9adbeafa4bca0adb8aaa3bea1e2a3beab).

Before using the ESnet SmartNIC tools, kindly review the official [ESnet SmartNIC Copyright Notice](https://github.com/esnet/esnet-smartnic-hw).

---

For more documentation, please refer to some other documentations that we have authored:

## 1. The tutorial on my [GitHub](https://groundsada.github.io/esnet-smartnic-tutorial/)

## 2. The FABRIC Testbed ESnet SmartNIC docs: [FABRIC ESnet SmartNIC docs](https://learn.fabric-testbed.net/knowledge-base/using-esnet-p4-workflow-on-fabric/)

## 3. The video tutorial on my [YouTube](https://www.youtube.com/watch?v=fiZMPPW_oRk&list=PL5Ght4QkHL8QK75R3ThqU7vzob5f65_Zi&ab_channel=MohammadFirasSada)

---

### Technical Information for Reproducing This Experiment in a Different Environment

1. **ESnet SmartNIC Tool Stack**

After building the ESnet SmartNIC tool stack, it runs as a **Docker Compose** stack. Therefore, you need a system capable of running `docker-compose`. We’ve tested this on multiple baremetal and KVM environments. For Kubernetes, we use **sysbox** to run **rootless Docker-in-Docker**, allowing `docker-compose` to run within the `crio` runtime without exposing a Docker daemon socket from the host.

The [FABRIC Testbed Guide](https://learn.fabric-testbed.net/knowledge-base/passing-xilinx-u280-fpga-into-a-kvm-vm/) explains how to pass Alveo FPGAs into a KVM VM.

2. **Vivado Software Requirements**

The ESnet SmartNIC tool stack requires **Vivado** software (with the correct version) for **development purposes only**. This version depends on the commits from the `esnet` repositories. The **National Research Platform** offers centralized Xilinx tools (Vitis, Vivado, Vitis\_HLS, etc.) served from a Ceph storage pool, along with a floating license server.

If you prefer not to use Coder, you can request access to the Persistent Volume Claim (PVC) for your namespace by contacting the Operations team (contact via Matrix).

3. **Licensing Information**

Proper **licensing** is required. When provisioning from Coder, the licensing server is already configured. Other namespaces or environments can access and point to the license server at:

`[email protected]`.

You can use NRP’s setup for **software-only** use cases, such as building FPGA artifacts for other environments (e.g., FABRIC, CC).

4. **Flashing FPGAs**

Reprogramming FPGAs (e.g., for P4 OpenNIC Shell, XRT, etc.) requires a **JTAG-over-USB connection** to the devices. For Alveo devices, this necessitates an external USB connection to the host server (in this case, the Kubernetes node). The pods provisioned for FPGA tasks will have the USB connection passed through using the **Smarter Device Manager**.

Flashing requires a **power cycle** of the host server (node), which must be coordinated with the Operations team. Please contact us if flashing is required for your work.

5. **FPGA Availability on NRP**

At the time of writing, there are **32 Alveo U55C FPGAs** on NRP, all available on the **Nautilus cluster** at the **San Diego Supercomputer Center**. These FPGAs are located on **PNRP nodes** following the naming convention:

`node-X-Y.sdsc.optiputer.net`.

6. **SmartNIC Configuration**

The FPGAs can be programmed as **SmartNICs**, and in some cases, users may expose them as network interfaces. Pods that handle network operations require special capabilities, such as `CAP_NET_RAW`. These capabilities are pre-configured in Coder, but if you are running outside of Coder, you will need to define these capabilities explicitly.

7. **DPDK Requirements**

Running **DPDK** requires both **hugepages** and **IOMMU passthrough**. These are provided on nodes hosting FPGAs.

8. **Privileges for ESnet SmartNIC Stack**

The ESnet SmartNIC stack performs privileged tasks (e.g., binding and unbinding from devices), which require extra privileges on the host node. These privileges are available in Coder. If you are setting up in your own namespace, please reach out for assistance.

**Important:** Misuse of these privileges will violate our Acceptable Use Policy and may result in immediate account suspension and accountability measures.

---

The ESnet SmartNIC framework provides an entire workflow to program AMD/Xilinx Alveo FPGA cards using P4. The ESnet framework is open-source and available on GitHub. ESnet is a high-performance network that supports scientific research. The ESnet team created the framework that seamlessly integrates AMD/Xilinx tools along with various tools like DPDK to provide an easy way of programming Alveo cards as SmartNICs. The framework runs in docker containers as demonstrated in this Jupyter Notebook.

### Step 1: Set up environment

Remove any pre-existing clones of the tutorial repo.

Terminal window

```
rm -rf ~/esnet-smartnic/esnet-smartnic-hw
```

Terminal window

```
echo "$BASH_VERSION"
```

If the above command doesn’t show a bash version, **you may be running with a Python kernel. Please switch to a Bash kernel.**

Terminal window

```
mkdir -p ~/esnet-smartnic

cd ~/esnet-smartnic
```

Clone the `esnet-smartnic-hw` repository from ESnet. Checkout at the latest tested commit.

### Step 2: Clone the reposirtory

Terminal window

```
git clone https://github.com/esnet/esnet-smartnic-hw.git

cd esnet-smartnic-hw

git checkout d3782445ce5f090ca955693a98ce68f96b68943c

git submodule update --init --recursive

sudo apt install python3-yaml python3-jinja2 python3-click -y

pip3 install -r esnet-fpga-library/tools/regio/requirements.txt

ls
```

You can see the contents of the repository. The examples directory has multiple examples to show.

Terminal window

```
cd examples/p4_only

ls
```

Running `make` in the p4\_only directory will build the *artifacts*, which is a **zip package** containing the compiled bitstream and all other necessary files to run on the FPGA.

The `sim` directory has the simulation-related files.

Terminal window

```
cd p4

ls
```

### Step 3: P4 Experiments

Terminal window

```
cat p4_only.p4
```

Terminal window

```
cp ../../../../assets/p4_only.p4 .

cat p4_only.p4
```

Terminal window

```
source /tools/Xilinx/Vivado/2023.1/settings64.sh

export XILINXD_LICENSE_FILE=2100@xilinxd.xilinx-dev
```

Terminal window

```
cd sim

head test-fwd-p0/packets_in.user
```

Terminal window

```
cp -r test-fwd-p0 test-fwd-p1

sed -i 's/^P4BM_DIRS = test-fwd-p0$/P4BM_DIRS = test-fwd-p0 test-fwd-p1/' Makefile

sed -i 's/^\(P4BM_DIR = test-fwd-p0\)/# \1/' Makefile
```

Terminal window

```
env | grep XILINXD_LICENSE_FILE
```

Terminal window

```
make

ls test-fwd-p1
```

Terminal window

```
head test-fwd-p1/packets_in.user

head test-fwd-p1/packets_out.user
```

Terminal window

```
python3 - <<EOF

from scapy.all import Ether, IP, wrpcap

# Generate 10 IPv4 packets with TTL values 1 to 10

packets = [Ether() / IP(dst="192.168.1.1", ttl=ttl) for ttl in range(1, 11)]

# Save the packets to the specified pcap file

wrpcap("test-fwd-p1/packets_in.pcap", packets)

print("PCAP file 'test-fwd-p1/packets_in.pcap' created with 10 Ethernet+IPv4 packets, TTLs 1-10")

EOF
```

Terminal window

```
make clean

rm -rf test-fwd-p1/packets_in.user

ls test-fwd-p1
```

Terminal window

```
tshark -r test-fwd-p1/packets_in.pcap -T tabs
```

Terminal window

```
make

tshark -r test-fwd-p1/packets_in.pcap -T tabs

tshark -r test-fwd-p1/packets_in.pcap -T fields -e ip.ttl
```

Terminal window

```
tshark -r test-fwd-p1/packets_out.pcap -T tabs

tshark -r test-fwd-p1/packets_out.pcap -T fields -e ip.ttl
```

### Step 4: Control Plane Table Entries

Terminal window

```
cat test-fwd-p1/packets_in.meta
```

Terminal window

```
cat test-fwd-p1/packets_out.meta
```

---

---

---

Now we reach the end of writing a P4 program and testing it against custom PCAP files. In the next notebook, we will be building the artifacts from the P4 logic.

---

This notebook is part 1 out of 3 in the **ESnet SmartNIC Tutorial on NRP** series.

This was last modified on March 4th, 2025.

For any inquiries, questions, feedback, please contact: [[email protected]](/cdn-cgi/l/email-protection#9ef3f8edfffaffdeebfdedfab0fbfaeb)

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/fpgas/esnet_development.md)

[Previous  
ESnet SmartNIC](/documentation/userdocs/fpgas/esnet)  [Next  
SmartNIC: Building](/documentation/userdocs/fpgas/esnet_building)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.