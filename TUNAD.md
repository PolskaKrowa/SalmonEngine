# Setting up a Tunad cluster

A guide for deploying Tunad (Tuna Distributed) across multiple machines.

Tunad handles two of the trickiest parts of heterogeneous MPI clusters automatically:

- **Architecture mismatches** — the bundled `configure` system generates machine-specific Makefiles, so each node compiles a native binary.
- **Non-shared filesystems** — Tunad's communication layer is designed for nodes that don't share a filesystem. No NFS required.

What you still need to set up yourself: a reliable network, OpenMPI on each machine, and passwordless SSH from your head node to every worker.

---

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Network setup](#network-setup)
3. [Install OpenMPI on every machine](#install-openmpi-on-every-machine)
4. [Configure SSH with different usernames](#configure-ssh-with-different-usernames)
5. [Build Tunad on each machine](#build-tunad-on-each-machine)
6. [Create an appfile](#create-an-appfile)
7. [Run Tunad](#run-tunad)
8. [macOS-specific notes](#macos-specific-notes)
9. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- All machines are on the same network (wired LAN recommended for performance)
- SSH is enabled on every machine
- You have admin/sudo access on each machine
- Each machine has a user account you control (usernames can differ)

This guide assumes a head node that launches jobs and one or more worker nodes that run them. Example setup used throughout:

| Role       | Hostname | IP address   | Username |
|------------|----------|--------------|----------|
| Head node  | head     | 192.168.1.10 | youruser |
| Worker 1   | node1    | 192.168.1.11 | alice    |
| Worker 2   | node2    | 192.168.1.12 | bob      |
| Worker 3   | node3    | 192.168.1.13 | carol    |

> **Note:** The head node will also contribute computational resources to model tuning.

---

## Network setup

A reliable network is the foundation of a working cluster. Flaky networking causes mysterious MPI failures that are hard to debug.

### 1. Assign static IP addresses

Avoid DHCP for cluster nodes — if an IP changes, your SSH config and appfile break. Set static IPs on each machine.

**Ubuntu / Debian** (using Netplan) — edit `/etc/netplan/01-netcfg.yaml`:

```yaml
network:
  version: 2
  ethernets:
    eth0:
      dhcp4: no
      addresses:
        - 192.168.1.11/24
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 1.1.1.1]
```

Apply with:

```bash
sudo netplan apply
```

**macOS:** System Settings → Network → your interface → Details → TCP/IP → set "Configure IPv4" to "Manually" and fill in the address.

### 2. Edit /etc/hosts on the head node

Add each worker so you can use hostnames instead of IPs everywhere. Edit `/etc/hosts`:

```
192.168.1.11  node1
192.168.1.12  node2
192.168.1.13  node3
```

This file is in the same location on macOS.

### 3. Test connectivity

From the head node, ping each worker:

```bash
ping -c 3 node1
ping -c 3 node2
ping -c 3 node3
```

All pings should succeed with consistent latency. Packet loss or timeouts will cause MPI to fail or hang.

### 4. Open firewall rules

MPI opens ports dynamically between nodes during a run. On a private LAN, the simplest approach is to allow all traffic within your cluster subnet.

**Ubuntu / Debian (ufw):**

```bash
sudo ufw allow from 192.168.1.0/24
sudo ufw reload
```

**macOS:** System Settings → Network → Firewall → Options — ensure `mpirun` is allowed, or disable the firewall temporarily for initial testing.

---

## Install OpenMPI on every machine

Run the appropriate command on **each machine**.

**Ubuntu / Debian:**

```bash
sudo apt update && sudo apt install -y openmpi-bin libopenmpi-dev
```

**Fedora / RHEL / CentOS:**

```bash
sudo dnf install -y openmpi openmpi-devel
# Also add this to ~/.bashrc so MPI is on your PATH at login
module load mpi
```

**Arch Linux:**

```bash
sudo pacman -S openmpi
```

**macOS (via Homebrew):**

```bash
brew install open-mpi
```

If `mpirun` is not found after installing on macOS, add Homebrew to your PATH:

```bash
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Verify on each machine

```bash
mpirun --version
mpicc --version
```

> **Important:** All nodes should run the same **major** version of OpenMPI (e.g. all 4.x or all 5.x). Mixing major versions can cause silent communication errors. Check with `mpirun --version` on each machine before continuing.

---

## Configure SSH with different usernames

MPI uses SSH to launch processes on remote nodes. Since each worker has a different username, you need to tell SSH which user to connect as on each host.

### 1. Generate an SSH key on the head node

Skip this if you already have a key at `~/.ssh/id_rsa`.

```bash
ssh-keygen -t rsa -b 4096
# Press Enter through all prompts to accept defaults
```

### 2. Copy your public key to each worker

```bash
ssh-copy-id alice@192.168.1.11
ssh-copy-id bob@192.168.1.12
ssh-copy-id carol@192.168.1.13
```

You'll be prompted for each user's password once. After this, login will be passwordless.

If `ssh-copy-id` is not available, do it manually:

```bash
cat ~/.ssh/id_rsa.pub | ssh alice@192.168.1.11 \
  "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

### 3. Create an SSH config on the head node

Edit `~/.ssh/config` (create it if it doesn't exist):

```
Host node1
    HostName 192.168.1.11
    User alice

Host node2
    HostName 192.168.1.12
    User bob

Host node3
    HostName 192.168.1.13
    User carol
```

Set the correct permissions:

```bash
chmod 600 ~/.ssh/config
```

### 4. Test passwordless login

```bash
ssh node1   # should log in as alice with no password prompt
ssh node2   # should log in as bob
ssh node3   # should log in as carol
```

If any of these still prompt for a password, check that:

- The public key is present in `~/.ssh/authorized_keys` on the worker
- `~/.ssh/authorized_keys` has permissions `600`
- `~/.ssh/` on the worker has permissions `700`
- Run `ssh -v node1` for verbose output showing exactly where auth fails

---

## Build Tunad on each machine

Because each machine compiles its own binary, architecture differences (x86_64 vs ARM, Linux vs macOS) are handled automatically by Tunad's `configure` script.

### 1. Copy the SalmonEngine source to every machine

From the head node:

```bash
for node in node1 node2 node3; do
    echo "Copying source to $node..."
    scp -r /path/to/SalmonEngine ${node}:~/SalmonEngine
done
```

### 2. Run configure and make on each machine

```bash
for node in node1 node2 node3; do
    echo "Building on $node..."
    ssh ${node} "cd ~/SalmonEngine && ./configure --enable-dist-tuner && make -j$(nproc)"
done
```

Also build on the head node itself:

```bash
cd /path/to/SalmonEngine && ./configure --enable-dist-tuner && make -j$(nproc)
```

Each machine now has a natively compiled binary at `~/SalmonEngine/src/tunad` (or wherever `make` places the output).

---

## Create an appfile

A plain hostfile assumes the binary lives at the same path on every node. An **appfile** lets you specify the binary path and process count independently per node, which is what you need when each user has a different home directory.

Create `appfile.txt` on the head node:

```
# Head node
-host 192.168.1.10 -np 1 /home/youruser/SalmonEngine/src/tunad --threads 4

# Worker nodes — SSH config aliases (node1, node2, node3) are used here,
# so MPI connects as the right user automatically
-host node1 -np 1 /home/alice/SalmonEngine/src/tunad --threads 4
-host node2 -np 1 /home/bob/SalmonEngine/src/tunad --threads 4
-host node3 -np 1 /home/carol/SalmonEngine/src/tunad --threads 4
```

Each line specifies:

- `-host` — which machine to use (hostname, IP, or SSH config alias)
- `-np` — number of MPI processes on that machine (tunad already handles threading via POSIX threads.)
- the path — where the Tunad binary lives **on that specific machine**
- `--threads` — number of CPU threads tunad will occupy. (defaults to 4 threads. Ideally this should be set to the maximum amount of threads available on your machine.)

---

## Run Tunad

```bash
mpirun --app appfile.txt
```

MPI SSHes into each node using the aliases defined in `~/.ssh/config`, launches the specified number of Tunad processes, and coordinates them. Output is streamed back to your terminal.

To see all errors if something goes wrong (disables error aggregation):

```bash
mpirun --mca orte_base_help_aggregate 0 --app appfile.txt
```

---

## macOS-specific notes

- **Firewall popups:** The first time `mpirun` tries to accept network connections, macOS may prompt you. Click "Allow", or pre-authorize it under System Settings → Network → Firewall → Options.
- **SIP (System Integrity Protection)** can block inter-process communication in some configurations. If you see permission errors that don't make obvious sense, this is worth looking into.
- **macOS uses clang under the hood**, but `mpicc` wraps it transparently — the `./configure --enable-dist-tuner && make` workflow is identical to Linux.

---

## Troubleshooting

### SSH still asks for a password

- Check the key is in `~/.ssh/authorized_keys` on the worker: `cat ~/.ssh/authorized_keys`
- Check permissions: `~/.ssh/` must be `700`, `authorized_keys` must be `600`
- Run `ssh -v node1` to see exactly where authentication fails

### "No such file or directory" when launching

The binary path in your appfile doesn't exist on that node. Check with:

```bash
ssh node1 "ls ~/SalmonEngine/src/tunad"
```

If it's missing, re-run the build step for that node.

### Processes launch but hang or crash immediately

Usually a version mismatch between OpenMPI installations. Check all nodes at once:

```bash
for node in node1 node2 node3; do
    echo -n "$node: "
    ssh $node "mpirun --version | head -1"
done
```

All nodes should report the same major version.

### MPI errors mentioning "BTL" or "OOB"

Transport-layer errors, usually caused by firewall rules blocking MPI's dynamic ports. Make sure all traffic between cluster IPs is allowed — see [Network setup](#network-setup).

### Process count doesn't match expected total

The total processes launched is the sum of all `-np` values in your appfile. If fewer than expected appear, one node likely failed silently. Run with error aggregation disabled to surface the real error:

```bash
mpirun --mca orte_base_help_aggregate 0 --app appfile.txt
```