# ESnet SmartNIC FPGA - Running

**Source:** https://nrp.ai/documentation/userdocs/fpgas/esnet_running

# ESnet SmartNIC FPGA - Running

## Running (Notebook 3/3): Running SmartNIC Logic on the FPGA

This notebook is **Part 3** of the **ESnet SmartNIC Tutorial on NRP** series. It continues from **Notebook 2** to provide an example for building the FPGA logic from our SmartNIC P4 code.

---

### Test Environment

This notebook was tested on the **National Research Platform (NRP)** using the **AMD/Xilinx Alveo U55C FPGA** and **Vivado 2023.1**. The Kubernetes pods were provisioned by [Coder](https://coder.nrp-nautilus.io).

If you run into any issues, please refer to the official [NRP Documentation](https://docs.nrp.ai), or reach out to us via [Matrix](https://element.nrp-nautilus.io) or [email](/cdn-cgi/l/email-protection#4a393f3a3a25383e0a242b3e2325242b26382f392f2b3829223a262b3e2c2538276425382d).

Before using the ESnet SmartNIC tools, kindly review the official [ESnet SmartNIC Copyright Notice](https://github.com/esnet/esnet-smartnic-hw).

---

This notebook doesn’t require Vivado or special software.

### Step 1: Acquiring the artifcats

You can start directly from this notebook if you have different development and deployment environments and already have the artifacts

Terminal window

```
echo "$BASH_VERSION"
```

If the above command doesn’t show a bash version, **you may be running with a Python kernel. Please switch to a Bash kernel.**

Terminal window

```
cd ~/esnet-smartnic/esnet-smartnic-hw/examples/p4_only/artifacts

ls
```

Terminal window

```
cp artifacts.u55c.p4_only.0.zip to ~/artifacts.u55c.p4_only.0.zip
```

You should see your artifacts here.

For the sake of demonstration, we will be using an artifacts zip file that we have made publicly available. **Please note that the artifacts use an evaluation license and will only work for 48 hours.** As a best practice, **always build fresh.** The way this issue materializes is with a SmartNIC that drops 100% of the packets, and stuck DPDK counters.

Terminal window

```
cd ~/

wget "https://nextcloud.nrp-nautilus.io/s/5LmLADJtNKmTYSp/download" -O "artifacts.au55c.p4_only.0.zip"
```

### Step 2: Setting up Docker

Terminal window

```
docker image ls
```

The `ESnet SmartNIC` stack requires 3 Docker images:

1. **esnet-smartnic-fw**: The firmware image, requires the artifacts zip file (built for every logic).  
   Repo: [esnet-smartnic-fw](https://github.com/esnet/esnet-smartnic-fw)
2. **smartnic-dpdk-docker**: A containerized DPDK with `pktgen`, patched and built for Alveo FPGAs. Consistent for all FPGA models. Build once, pull, and run everywhere.  
   Repo: [smartnic-dpdk-docker](https://github.com/esnet/smartnic-dpdk-docker)
3. **xilinx-labtools-docker**: A Vivado Lab image that provides tools for flashing the cards. Consistent for all FPGA models. Build once, pull, and run everywhere.  
   Repo: [xilinx-labtools-docker](https://github.com/esnet/xilinx-labtools-docker)

We have provided publicly hosted versions of the `smartnic-dpdk-docker` and `xilinx-labtools-docker` images. These images are hosted on our community gitlab ([gitlab.nrp-nautilus.io](https://gitlab.nrp-nautilus.io)) in repositories. **The versions are not universal, the images with the latest tag are the ones tested with the versions of esnet-smartnic-hw and esnet-smartnic-fw that we refer to in most recent docs and notebooks.**

Terminal window

```
docker pull gitlab-registry.nrp-nautilus.io/esnet/xilinx-labtools-docker

docker tag gitlab-registry.nrp-nautilus.io/esnet/xilinx-labtools-docker xilinx-labtools-docker:${USER}-dev

docker pull gitlab-registry.nrp-nautilus.io/esnet/smartnic-dpdk-docker

docker tag gitlab-registry.nrp-nautilus.io/esnet/smartnic-dpdk-docker smartnic-dpdk-docker:${USER}-dev
```

Terminal window

```
docker image ls
```

Terminal window

```
whoami
```

### Step 3: Prep the `esnet-smartnic-fw` repo

The last step is building the `esnet-smartnic-fw` image.

Terminal window

```
pwd
```

Terminal window

```
cd ~/esnet-smartnic && \

git clone https://github.com/esnet/esnet-smartnic-fw.git && \

cd esnet-smartnic-fw && \

git checkout c064d4ac775ed1a4c50ec72dea3615f9c644433e && \

git submodule update --init --recursive && \

ls
```

**Copy the artifacts to sn-hw without renaming.**

Terminal window

```
cp ~/artifacts.au55c.p4_only.0.zip sn-hw/

ls sn-hw
```

`artifacts` is a prefix all `hw` artifacts start with.

`au55c` is the board model (U55C). For U280, it would be `au280`.

`p4_only` is the name of the example we built. Had we built our own, it would’ve had a name we specified.

`0` is the version. Specified in the Makefile..

### Step 4: Building the docker image

The root director of the repo requires a correctly filled `.env` file.

Terminal window

```
cat example.env | grep -i required -A 4
```

Terminal window

```
cat <<EOL > .env

SN_HW_APP_NAME=p4_only

SN_HW_BOARD=au55c

SN_HW_VER=0

EOL
```

Terminal window

```
cat .env
```

Now we can build the image with `./build.sh`. This might take ~30-40 minutes.

Terminal window

```
./build.sh
```

Terminal window

```
docker image ls
```

### Step 5: Configuring the environment for the stack

The stack with the docker compose profile is in the `sn-stack` directory.

Terminal window

```
cd sn-stack && \

cat example.env
```

Similar to the root directory, the `sn-stack` directory requires a `.env` file correctly populated. There are three main environment variables that are needed for the stack to work:

1- The PCI bus address for the FPGA (this is always needed).

2- The USB iSerial for the correct FPGA (this is needed in a multi-FPGA scenario).

3- The profile to run in. In this experiment, we are running in `smartnic-mgr-vfio-unlock`, which allows us to use the smartnic directly without it being locked by DPDK and pktgen.

The command below gets us the list of all Xilinx pci devices that the pod sees. We pick the FPGA we want. Normally, users would reserve the FPGAs per node.

Terminal window

```
lspci -d 10ee:
```

The command below lists all USB devices with an iSerial, and looks for the iSerial starting with XF. You might get multiple results, for the mapping, please contact us.

**This works on our systems because the only USB devices with a serial number are the JTAG connections, but for different systems, there may be more USB connections, please carefully go through lsusb -vvv to get the correct device.**

Terminal window

```
sudo lsusb -vvv | grep -i iserial | grep -i XF
```

Terminal window

```
cat <<EOL > .env

FPGA_PCIE_DEV=0000:c1:00

HW_TARGET_SERIAL=XFL1QYVIDV45A

COMPOSE_PROFILES=smartnic-mgr-vfio-unlock

EOL
```

Terminal window

```
docker compose up -d
```

Now, our stack is up.

This will show information about the device such as the build version, build date/time and temperature:

Terminal window

```
docker compose exec smartnic-fw sn-cli dev version

docker compose exec smartnic-fw sn-cli dev temp
```

The USR\_ACCESS value is typically the unique build pipeline number that produced the embedded FPGA bitfile. The BUILD\_STATUS value holds an encoded date/time (Aug 30 at 05:32am) which is when the embedded FPGA bitfile build was started. The DNA value holds the factory-programmed unique ID of the FPGA

The script below resets the names of the ports, sets up app0 and app0 port and routes all egress packets to the propoer egress.

For more details about the commands and options, please check out the appendix (Appendix A) below.

Terminal window

```
docker compose exec smartnic-fw bash -c '

sn-cli dev version

sn-cli sw in-port-rename cmac0:cmac0 cmac1:cmac1 host0:host0 host1:host1

sn-cli sw app0-port-redirect cmac0:cmac0 cmac1:cmac1 host0:host0 host1:host1

sn-cli sw app1-port-redirect cmac0:cmac0 cmac1:cmac1 host0:host0 host1:host1

sn-cli sw bypass-connect cmac0:cmac0 cmac1:cmac1 host0:host0 host1:host1

sn-cli sw in-port-connect cmac0:app0 cmac1:app0 host0:app0 host1:app0

sn-cli sw status

sn-cli qdma setqs 1 1

sn-cli cmac enable

sn-cli cmac status

'
```

Terminal window

```
docker compose exec smartnic-fw sn-cli dev temp
```

Terminal window

```
docker compose exec smartnic-fw sn-cli probe stats
```

Terminal window

```
docker compose exec smartnic-fw sn-p4-cli info
```

Terminal window

```
docker compose exec smartnic-fw sn-p4-cli table-insert --help
```

### Step 6: Running DPDK

**Always make sure the stack is down before changing profiles.**

Terminal window

```
docker compose down
```

Terminal window

```
cat <<EOL > .env

FPGA_PCIE_DEV=0000:c1:00

HW_TARGET_SERIAL=XFL1QYVIDV45A

COMPOSE_PROFILES=smartnic-mgr-dpdk-manual

EOL
```

We bring the stack up with the correct profile for DPDK.

Please note that running in the DPDK profile requries **re-running the setup script we used previously, and whenever pktgen is close, the card is locked and all registers return 0s and Fs.**

Terminal window

```
docker compose up -d
```

**Please run pktgen in a terminal, otherwise it will hang.**

For more details about the commands and options, please check out the appendix (Appendix A) below.

Terminal window

```
##docker compose exec smartnic-dpdk pktgen -a $SN_PCIE_DEV.0 -a $SN_PCIE_DEV.1 -l 3-7 -n 3 -d librte_net_qdma.so --file-prefix $SN_PCIE_DEV- -- -v -m [4:5].0 -m [6:7].1
```

**Always make sure the stack is down before finishing your work.**

Terminal window

```
docker compose down -v --remove-orphans
```

For more details about the commands and options, please check out the appendix (Appendix A) below.

### Appendix A: `sn-cli` and `pktgen` options and commands:

#### Using the sn-cli tool

The sn-cli tool provides subcommands to help you accomplish many common tasks for inspecting and configuring the smartnic platform components.

All commands described below are expected to be executed within the `smartnic-fw` container environment. Use this command to enter the appropriate environment.

```
docker compose exec smartnic-fw bash
```

The `sn-cli` tool will automatically look for an environment variable called `SN_CLI_SLOTADDR` which can be set to the PCIe BDF address of the device that you would like to interact with. In the `smartnic-fw` container, this value will already be set for you.

##### Displaying device information with the “dev” subcommand

This will show information about the device such as the build version, build date/time, and temperature.

```
root@smartnic-fw:/# sn-cli dev version

Device Version Info

DNA:           0x40020000012306a21c10c285

USR_ACCESS:    0x000086d3 (34515)

BUILD_STATUS:  0x04130920

root@smartnic-fw:/# sn-cli dev temp

Temperature Monitors

FPGA SLR0:    45.551 (deg C)
```

The `USR_ACCESS` value is typically the unique build pipeline number that produced the embedded FPGA bitfile.  
The `BUILD_STATUS` value holds an encoded date/time (Aug 30 at 05:32am) which is when the embedded FPGA bitfile build was started.  
The `DNA` value holds the factory-programmed unique ID of the FPGA.

##### Inspecting and Configuring the CMAC (100G) Interfaces with the “cmac” subcommand

Enable/Disable one or more (or all by default) 100G MAC interfaces using these commands:

```
sn-cli cmac enable

sn-cli cmac disable

sn-cli cmac -p 0 enable

sn-cli cmac -p 1 disable
```

Enabling a CMAC interface allows frames to pass (Rx/Tx) at the MAC layer. These commands **do not affect** whether the underlying physical layer (PHY) is operational.

Display the current MAC and PHY status of one or more (or all by default) 100G MAC interfaces using these commands:

```
root@smartnic-fw:/# sn-cli cmac status

CMAC0

Tx (MAC ENABLED/PHY UP)

Rx (MAC ENABLED/PHY UP)

CMAC1

Tx (MAC ENABLED/PHY UP)

Rx (MAC ENABLED/PHY DOWN)
```

In the example output above, CMAC0 PHY layer is **UP** in both the Tx and Rx directions. The MAC is fully enabled. This link is operational and should be passing packets normally.

In the example output above, CMAC1 PHY layer is **DOWN** in the Rx (receive) direction. Possible causes for this are:

* No QSFP28 plugged into 100G port 0 the U280 card
* Wrong type of QSFP28 module plugged into 100G port 0
  + 100G QSFP28 SR4 or LR4 modules are supported
  + Some 100G AOC or DACs are known to work
  + QSFP+ 40G modules **are not supported**
  + QSFP 5G modules **are not supported**
* QSFP28 card improperly seated in the U280 card
  + Check if the QSFP28 module is inserted upside down and physically blocked from being fully inserted
  + Unplug/replug the module, ensuring that it is properly oriented and firmly seated
* Fiber not properly inserted
  + Unplug/replug the fiber connection at each end
* Far end is operating in 4x25G or 2x50G split mode
  + The smartnic platform **does not support** 4x25G or 2x50G mode
  + Only 100G mode is supported on each of the U280 100G interfaces
  + Configure far end in 100G mode
* Far end has RS-FEC (Reed-Solomon Forward Error Correction) enabled
  + The smartnic platform **does not support** RS-FEC
  + Disable RS-FEC on the far end equipment

A more detailed status can also be displayed using the `--verbose` option. Note that the `--verbose` option is a global option and thus must be positioned **before** the `cmac` subcommand.

```
root@smartnic-fw:/# sn-cli --verbose cmac -p 1 status

CMAC1

Tx (MAC ENABLED/PHY UP)

tx_local_fault 0

Rx (MAC ENABLED/PHY DOWN)

rx_got_signal_os 0

rx_bad_sfd 0

rx_bad_preamble 0

rx_test_pattern_mismatch 0

rx_received_local_fault 0

rx_internal_local_fault 1

rx_local_fault 1

rx_remote_fault 0

rx_hi_ber 0

rx_aligned_err 0

rx_misaligned 0

rx_aligned 0

rx_status 0
```

Display summary statistics for packets Rx’d and Tx’d from CMAC ports:

```
root@smartnic-fw:/# sn-cli cmac stats

CMAC0: TX      0 RX      0 RX-DISC      0 RX-ERR      0

CMAC1: TX      0 RX      0 RX-DISC      0 RX-ERR      0
```

Note: The CMAC counters are only cleared/reset when the FPGA is reprogrammed.

##### Inspecting and Configuring the PCIe Queue DMA (QDMA) block with the “qdma” subcommand

The QDMA block is responsible for managing all DMA queues used for transferring packets and/or events bidirectionally between the U280 card and the Host CPU over the PCIe bus. In order for any DMA transfers to be allowed on either of the PCIe Physical Functions (PF), an appropriate number of DMA Queue IDs must be provisioned. This can be done using the `qdma` subcommand.

Configure the number of queues allocated to each of the PCIe Physical Functions:

```
sn-cli qdma setqs 1 1
```

This assigns 1 QID to PF0 and 1 QID to PF1. The `setqs` subcommand also takes care of configuring the RSS entropy -> QID map with an equal weighted distribution of all allocated queues. If you’re unsure of how many QIDs to allocate, using `1 1` here is your best choice.

Inspect the configuration of the QDMA block:

```
sn-cli qdma status
```

Packet, byte, and error counters are tracked for packets heading between the QDMA engine and the user application. You can display them with this command:

```
sn-cli qdma stats
```

Refer to the `open-nic-shell` documentation for an explanation of exactly where in the FPGA design these statistics are measured.

##### Inspecting packet counters in the smartnic platform with the “probe” subcommand

The smartnic platform implements monitoring points in the datapath at various locations. You can inspect these counters using this command:

```
sn-cli probe stats
```

Refer to the `esnet-smartnic-hw` documentation for an explanation of exactly where in the FPGA design these statistics are measured.

##### Configuring the smartnic platform ingress/egress/bypass switch port remapping functions with the “sw” subcommand

The smartnic platform implements reconfigurable ingress and egress port remapping, connections, and redirecting. You can inspect and modify these configuration points using the “sw” subcommand.

Most of the `sw` subcommands take one or more port bindings as parameters. The port bindings are of the form:

```
<port>:<port-connector>
```

Where:

* `<port>` is one of:
  + cmac0 — 100G port 0
  + cmac1 — 100G port 1
  + host0 — DMA over PCIe Physical Function 0 (PF0)
  + host1 — DMA over PCIe Physical Function 1 (PF1)
* `<port-connector>` is context dependent and is one of:
  + cmac0
  + cmac1
  + host0
  + host1
  + bypass — a high bandwidth channel through the smartnic which does **NOT** pass through the user’s application
  + app0 — user application port 0 (typically a p4 program ingress)
  + app1 — user application port 1 (only available when user implements it in verilog)
  + drop — infinite blackhole that discards all packets sent to it

##### Display the current configuration status

```
sn-cli sw status
```

##### Remap/rename physical input ports to logical input ports

The `in-port-rename` subcommand allows you to remap the identity of a smartnic platform physical ingress port to any logical port as seen by the user logic. Once remapped (eg. from `a`->`b`), all following logic in the smartnic will perceive that the packet arrived on ingress port `b` even though it physically arrived on port `a`. This can be useful for test injection scenarios but would typically be set to a straight-through mapping in production.

```
sn-cli sw in-port-rename a:b
```

To reset this mapping so each port maps to its usual identity:

```
sn-cli sw in-port-rename cmac0:cmac0 cmac1:cmac1 host0:host0 host1:host1
```

##### Attach logical input ports to pipelines

The `in-port-connect` subcommand allows you to connect a logical input port to different processing pipelines within the smartnic. This can be used to connect to a p4 program or to custom logic within the user application. It can also be used to shunt all packets to a blackhole or to bypass packets around the user application entirely.

```
sn-cli sw in-port-connect cmac0:app0 cmac1:app0 host0:bypass host1:bypass
```

##### Connect input ports to output ports in the bypass path

The `bypass-connect` subcommand allows you to connect input ports directly to output ports as they pass through the bypass path (ie. not through the user application). This is useful for providing direct connectivity from host PCIe PFs to 100G CMAC interfaces for network testing.

```
sn-cli sw bypass-connect host0:cmac0 host1:cmac1 cmac0:host0 cmac1:host1
```

**NOTE** any packets that follow the bypass path will not be processed by the user’s p4 program

##### Override user application output port decisions and redirect to an alternate port

The `app0-port-redirect` and `app1-port-redirect` subcommands allow the user to override the forwarding decisions made by the user application and/or p4 program and redirect any given output port to a different output port. This can be useful during development/debugging and in test fixtures.

**NOTE** there are separate overrides for the app0 outputs and the app1 outputs.

```
sn-cli sw app0-port-redirect cmac0:host0 cmac1:host1

sn-cli sw app1-port-redirect cmac0:host0 cmac1:host1
```

To reset this mapping so each output ports maps to its usual destination:

```
sn-cli sw app0-port-redirect cmac0:cmac0 cmac1:cmac1 host0:host0 host1:host1

sn-cli sw app1-port-redirect cmac0:cmac0 cmac1:cmac1 host0:host0 host1:host1
```

#### Using the sn-p4-cli tool

The user’s p4 application embedded within the smartnic design may have configurable lookup tables which are used during the wire-speed execution of the packet processing pipeline. The sn-p4-cli tool provides subcommands to help you to manage the rules in all of the lookup tables defined in your p4 program.

All commands described below are expected to be executed within the `smartnic-fw` container environment. Use this command to enter the appropriate environment.

```
docker compose exec smartnic-fw bash
```

The `sn-p4-cli` tool will automatically look for an environment variable called `SN_P4_CLI_SERVER` which can be set to the hostname of the `sn-p4-agent` that will perform all of the requested actions on the real hardware. In the `smartnic-fw` container, this value will already be set for you.

##### Inspecting the pipeline structure with the “info” subcommand

The `info` subcommand is used to display the pipeline structure, including table names, match fields (and their types), action names and the list of parameters for each action. This information can be used to formulate new rule definitions for the other subcommands.

```
sn-p4-cli info
```

##### Inserting a new rule into a table

The `table-insert` subcommand allows you to insert a new rule into a specified table.

```
sn-p4-cli table-insert <table-name> <action-name> --match <match-expr> [--param <param-expr>] [--priority <prio-val>]
```

Where:

* `<table-name>` is the name of the table to be operated on
* `<action-name>` is the action that you would like to activate when this rule matches
* `<match-expr>` is one or more match expressions which collectively define when this rule should match a given packet
  + The number and type of the match fields depends on the p4 definition of the table
  + The `--match` option may be specified multiple times and all `match-expr`s will be concatenated
* `<param-expr>` is one or more parameter values which will be returned as a result when this rule matches a given packet
  + The number and type of the action parameters depends on the p4 definition of the action within the table
  + Some actions require zero parameters. In this case, omit the optional `--param` option entirely.
* `<prio-val>` is the priority to be used to resolve scenarios where multiple matches could occur
  + The `--priority` option is *required* for tables with CAM/TCAM type matches (prefix/range/ternary)
  + The `--priority` option is *prohibited* for tables without CAM/TCAM type mathes

**NOTE**: You can find details about your pipeline structure and valid names by running the `info` subcommand.

##### Updating an existing rule within a table

The `table-update` subcommand allows you to update the action and parameters for an existing rule within a table

```
sn-p4-cli table-update <table-name> <new-action-name> --match <match-expr> [--param <new-param-expr>]
```

Where:

* `<table-name>` is the table containing the rule to be updated
* `<new-action-name>` is the new action that should be applied when this rule matches
* `<match-expr>` is the exact original `<match-expr>` used when the original rule was inserted
* `<new-param-expr>` is the set of new parameters to be returned when this rule matches
  + **NOTE**: the new parameters must be consistent with the new action

##### Removing previously inserted rules

The `clear-all` and `table-clear` and `table-delete` subcommands allow you to remove rules from tables with varying precision.

Clear all rules from *all tables* in the pipeline.

```
sn-p4-cli clear-all`
```

Clear all rules from a *single* specified table.

```
sn-p4-cli table-clear <table-name>
```

Remove a specific rule from a specific table.

```
table-delete <table-name> --match <match-expr>
```

##### Bulk changes of rules using a p4bm simulator rules file

Using the the `p4bm-apply` subcommand, a list of pipeline modifications can be applied from a file. A subset of the full p4bm simulator file format is supported by the `sn-p4-cli` command.

```
sn-p4-cli p4bm-apply <filename>
```

Supported actions within the p4bm file are:

* `table_insert <table-name> <action-name> <match-expr> => <param-expr> [priority]`
  + Insert a rule
* `clear_all`
  + Clear all rules from all tables
* `table_clear <table-name>`
  + Clear all rules from a specified table

All comment characters `#` and text following them up to the end of the line are ignored.

#### Stopping the runtime environment

When we’re finished using the smartnic runtime environment, we can stop and remove our docker containers.

```
docker compose down -v
```

### Using the smartnic-dpdk container

The `sn-stack` environment can be started in a mode where the FPGA can be controlled by a DPDK application. Running in this mode requires a few carefully ordered steps.

Broadly speaking, the steps required to bring up a DPDK application are as follows:

* Bind the `vfio-pci` kernel driver to each FPGA PCIe physical function (PF)
  + This is handled automatically by the sn-stack.
* Run a DPDK application with appropriate DPDK Environment Abstraction Layer (EAL) settings
  + Use `-a $SN_PCIE_DEV.0` to allow control of one or more specific FPGA PCIe PFs
  + Use `-d librte_net_qdma.so` to dynamically link the correct Userspace Polled-Mode Driver (PMD) for the smartnic QDMA engine
  + The EAL will
    - Open the PCIe PFs using the kernel’s `vfio-pci` driver
    - Take the FPGA device out of reset
    - Open and map large memory regions for DMA using the kernel’s `hugepages` driver
  + The application is responsible for assigning buffers to one or more of the FPGA’s DMA queues
* Use the `sn-cli` tool to configure some of the low-level hardware components in the FPGA
  + Configure the set of valid DMA queues in the FPGA (must match what is set in the DPDK application)
  + Bring up the physical ethernet ports

In the examples below, we will be running the `pktgen-dpdk` application to control packet tx/rx via the FPGA’s PCIe physical functions. This can be very useful for injecting packets into a design for testing behaviour on real hardware.

For more information about DPDK in general, see:

* <http://core.dpdk.org/doc/>

For more information about the `pktgen-dpdk` application, see:

* <https://pktgen-dpdk.readthedocs.io/en/latest/index.html>

Before you bring up the `sn-stack`, please ensure that you have uncommented this line in your `.env` file

```
COMPOSE_PROFILES=smartnic-dpdk
```

If you changed this while the stack was already running, you’ll need to restart the stack with down/up.

First, you’ll need to start up the `pktgen` application to open the vfio-pci device for PF0 and PF1 and take the FPGA out of reset.

```
$ docker compose exec smartnic-dpdk bash

root@smartnic-dpdk:/# pktgen -a $SN_PCIE_DEV.0 -a $SN_PCIE_DEV.1 -l 4-8 -n 4 -d librte_net_qdma.so --file-prefix $SN_PCIE_DEV- -- -v -m [5:6].0 -m [7:8].1

Pktgen:/> help
```

NOTE: Leave this application running while doing the remaining setup steps. The setup steps below must be re-run after each time you restart the pktgen application since the FPGA gets reset between runs.

Open a **separate** shell window which you will use for doing the low-level smartnic platform configuration.

Configure the Queue mappings for host PF0 and PF1 interfaces and bring up the physical ethernet ports using the `smartnic-fw` container.

```
$ docker compose exec smartnic-fw bash

root@smartnic-fw:/# sn-cli qdma setqs 1 1

root@smartnic-fw:/# sn-cli qdma status

root@smartnic-fw:/# sn-cli cmac enable

root@smartnic-fw:/# sn-cli cmac status
```

Setting up the queue mappings tells the smartnic platform which QDMA queues to use for h2c and c2h packets. Enabling the CMACs allows Rx and Tx packets to flow (look for `MAC ENABLED/PHY UP`).

### Advanced usage of the pktgen-dpdk application

Example of streaming packets out of an interface from a pcap file rather than generating the packets within the UI. Note the `-s <P>:file.pcap` option where `P` refers to the port number to bind the pcap file to.

```
root@smartnic-dpdk:/# pktgen -a $SN_PCIE_DEV.0 -a $SN_PCIE_DEV.1 -l 4-8 -n 4 -d librte_net_qdma.so --file-prefix $SN_PCIE_DEV- -- -v -m [5:6].0 -m [7:8].1 -s 1:your_custom.pcap

Pktgen:/> port 1

Pktgen:/> page pcap

Pktgen:/> page main

Pktgen:/> start 1

Pktgen:/> stop 1

Pktgen:/> clr
```

Example of running a particular test case via a script rather than typing at the UI

```
cat <<_EOF > /tmp/test.pkt

clr

set 1 size 1400

set 1 count 1000000

enable 0 capture

start 1

disable 0 capture

_EOF
```

```
root@smartnic-dpdk:/# pktgen -a $SN_PCIE_DEV.0 -a SN_PCIE_DEV.1 -l 4-8 -n 4 -d librte_net_qdma.so --file-prefix $SN_PCIE_DEV- -- -v -m [5:6].0 -m [7:8].1 -f /tmp/test.pkt
```

### Troubleshooting the pktgen-dpdk Application

If pktgen isn’t starting, please consider the following troubleshooting steps:

Ensure you are using the correct profile in your `sn-stack/.env` file and that you are starting pktgen with the right command. For a more detailed understanding of the command, please refer to the pktgen documentation provided earlier.

If pktgen is starting, but packets aren’t flowing as expected, you can check the packet path using the following command inside the `smartnic-fw` container:

```
sn-cli probe stats
```

If packets sent to/from the host aren’t achieving line rate (100Gbps per port), it could be due to QDMA queue allocation. You can attempt to allocate more QDMA queues per port by setting `sn-cli qdma setqs` to values higher than `1 1`.

If packets are egressing to the wrong port (whether CMAC or PF), it might be due to the `sn-cli` configuration. For example, here’s a script that routes all egress packets to CMAC1:

```
#!/bin/bash

sn-cli dev version

sn-cli sw in-port-rename cmac0:cmac0 cmac1:cmac1 host0:host0 host1:host1

sn-cli sw app0-port-redirect cmac0:cmac0 cmac1:cmac1 host0:host0 host1:host1

sn-cli sw app1-port-redirect cmac0:cmac0 cmac1:cmac1 host0:host0 host1:host1

sn-cli sw bypass-connect cmac0:cmac0 cmac1:cmac1 host0:host0 host1:host1

sn-cli sw in-port-connect cmac0:app0 cmac1:app0 host0:app0 host1:app0

sn-cli sw status

sn-cli qdma setqs 1 1

sn-cli cmac enable

sn-cli cmac status
```

These steps should help you troubleshoot issues related to the pktgen-dpdk application effectively.

---

---

---

Now we reach the end of our tutorial.

For more information, please refer to some other documentations that we have authored:

## 1. The tutorial on my [GitHub](https://groundsada.github.io/esnet-smartnic-tutorial/)

## 2. The FABRIC Testbed ESnet SmartNIC docs: [FABRIC ESnet SmartNIC docs](https://learn.fabric-testbed.net/knowledge-base/using-esnet-p4-workflow-on-fabric/)

## 3. The video tutorial on my [YouTube](https://www.youtube.com/watch?v=fiZMPPW_oRk&list=PL5Ght4QkHL8QK75R3ThqU7vzob5f65_Zi&ab_channel=MohammadFirasSada)

---

This notebook is part 3 out of 3 in the **ESnet SmartNIC Tutorial on NRP** series.

This was last modified on March 4th, 2025.

For any inquiries, questions, feedback, please contact: [[email protected]](/cdn-cgi/l/email-protection#7419120715101534011707105a111001)

[Edit page](https://gitlab.nrp-nautilus.io/prp/nrp-site/-/tree/main/src/content/docs/Documentation/userdocs/fpgas/esnet_running.md)

[Previous  
SmartNIC: Building](/documentation/userdocs/fpgas/esnet_building)  [Next  
Intro](/documentation/userdocs/storage/intro)

![NSF Logo](/nsf-logo.png)

This work was supported in part by National Science Foundation (NSF) awards CNS-1730158, ACI-1540112, ACI-1541349, OAC-1826967, OAC-2112167, CNS-2100237, CNS-2120019.