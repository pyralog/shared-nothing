# Low-Level Networking Options for Shared-Nothing Architecture

This document evaluates ultra-high-performance networking options for extending the shared-nothing library to distributed systems.

## Performance Overview

| Technology | Latency | Throughput | Complexity | Platform |
|-----------|---------|------------|------------|----------|
| **Kernel Modules** | <100ns | 100+ Gbps | Extreme | Linux |
| **DPDK** | 200-500ns | 100+ Gbps | Very High | Linux/BSD |
| **RDMA** | 500ns-1μs | 100+ Gbps | High | Linux + HW |
| **AF_XDP** | 1-2μs | 40+ Gbps | High | Linux 4.18+ |
| **io_uring** | 2-5μs | 20+ Gbps | Medium | Linux 5.1+ |
| **Raw Sockets** | 5-10μs | 10+ Gbps | Medium | All |
| **eBPF/XDP** | 3-8μs | 40+ Gbps | Medium-High | Linux 4.8+ |
| **epoll/kqueue** | 10-50μs | 10+ Gbps | Low-Medium | All |

## 1. io_uring (Recommended for Most Cases)

### Overview
Modern Linux async I/O interface using shared ring buffers between kernel and userspace.

### Key Features
- Zero-copy I/O operations
- Batch submission/completion
- No context switches for I/O
- 30-50% faster than epoll
- Works with files, sockets, everything

### Rust Crates
```toml
tokio-uring = "0.4"        # Tokio integration
io-uring = "0.6"           # Direct bindings
```

### Architecture
```text
Userspace                  Kernel
┌─────────────┐           ┌─────────────┐
│ Submission  │──────────>│   Process   │
│   Queue     │           │   Requests  │
└─────────────┘           └─────────────┘
                                 │
┌─────────────┐           ┌─────────────┐
│ Completion  │<──────────│   Results   │
│   Queue     │           │             │
└─────────────┘           └─────────────┘
```

### Implementation Approach
```rust
use tokio_uring::net::{TcpListener, TcpStream};

pub struct IoUringTransport {
    runtime: tokio_uring::Runtime,
}

impl IoUringTransport {
    pub async fn send_message(&self, addr: SocketAddr, data: &[u8]) -> Result<()> {
        let stream = TcpStream::connect(addr).await?;
        stream.write_all(data).await?;
        Ok(())
    }
    
    pub async fn recv_message(&self, stream: &TcpStream) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; 8192];
        let n = stream.read(&mut buf).await?;
        buf.truncate(n);
        Ok(buf)
    }
}
```

### Pros
- ✅ Best general-purpose performance
- ✅ Works with standard network stack
- ✅ No special hardware required
- ✅ Growing ecosystem support
- ✅ Unified interface for all I/O

### Cons
- ❌ Linux 5.1+ only
- ❌ Tokio-uring still maturing
- ❌ More complex than epoll
- ❌ Requires newer kernels

### Use Cases
- High-performance distributed systems
- Low-latency microservices
- Database replication
- Real-time data streaming

---

## 2. DPDK (Data Plane Development Kit)

### Overview
Kernel bypass framework that gives direct access to NICs from userspace.

### Key Features
- Complete kernel bypass
- Poll-mode drivers (PMD)
- Huge page memory
- Lock-free queues
- 100+ Gbps throughput possible

### Rust Crates
```toml
dpdk-rs = "0.1"           # DPDK bindings
capsule = "0.1"           # High-level DPDK framework
```

### Architecture
```text
Traditional Stack          DPDK Stack
┌─────────────┐           ┌─────────────┐
│ Application │           │ Application │
├─────────────┤           │   (DPDK)    │
│   Socket    │           ├─────────────┤
├─────────────┤           │   PMD       │
│   TCP/IP    │           │  Drivers    │
├─────────────┤           ├─────────────┤
│  Network    │           │     NIC     │
│  Driver     │           │  (Direct)   │
├─────────────┤           └─────────────┘
│     NIC     │
└─────────────┘

Kernel Space               Userspace Only
```

### Implementation Approach
```rust
use dpdk_rs::*;

pub struct DpdkTransport {
    port_id: u16,
    tx_queue: TxQueue,
    rx_queue: RxQueue,
    mempool: MemPool,
}

impl DpdkTransport {
    pub fn send_burst(&mut self, packets: &[Packet]) -> Result<usize> {
        // Zero-copy packet transmission
        let mut mbufs: Vec<_> = packets.iter()
            .map(|p| self.mempool.alloc_mbuf(p.data()))
            .collect();
        
        self.tx_queue.tx_burst(&mut mbufs)
    }
    
    pub fn recv_burst(&mut self, max_packets: usize) -> Result<Vec<Packet>> {
        let mut mbufs = vec![std::ptr::null_mut(); max_packets];
        let n = self.rx_queue.rx_burst(&mut mbufs)?;
        
        // Process received packets
        mbufs[..n].iter()
            .map(|mbuf| Packet::from_mbuf(*mbuf))
            .collect()
    }
}
```

### Configuration
```rust
// EAL (Environment Abstraction Layer) initialization
fn init_dpdk() -> Result<()> {
    let args = vec![
        "myapp".to_string(),
        "-l".to_string(), "0-3".to_string(),      // CPU cores
        "-n".to_string(), "4".to_string(),         // Memory channels
        "--huge-dir".to_string(), "/mnt/huge".to_string(),
    ];
    
    dpdk_rs::eal_init(&args)?;
    Ok(())
}
```

### Pros
- ✅ Absolute maximum throughput
- ✅ Microsecond-level latency
- ✅ Full control over packet processing
- ✅ Mature, battle-tested (Intel)
- ✅ Rich ecosystem of libraries

### Cons
- ❌ Requires dedicated NICs
- ❌ Complex setup and configuration
- ❌ No standard network stack (custom TCP/IP needed)
- ❌ Large memory footprint (huge pages)
- ❌ Steep learning curve
- ❌ Application becomes network driver

### Use Cases
- High-frequency trading
- Network appliances (firewalls, routers)
- 5G/telecom infrastructure
- Video streaming at scale
- When you need >10M packets/sec

---

## 3. AF_XDP (Address Family XDP)

### Overview
Linux socket type that leverages XDP (eXpress Data Path) for fast packet processing in userspace.

### Key Features
- Kernel bypass with eBPF filtering
- Zero-copy RX/TX
- Faster than standard sockets
- Simpler than DPDK
- Standard network driver

### Rust Crates
```toml
xsk-rs = "0.2"            # AF_XDP socket bindings
libbpf-rs = "0.21"        # eBPF support
```

### Architecture
```text
┌──────────────────────────────────────┐
│          Application                 │
├──────────────────────────────────────┤
│        AF_XDP Socket                 │
│    ┌────────────┐  ┌────────────┐   │
│    │  RX Ring   │  │  TX Ring   │   │ Userspace
│    └─────┬──────┘  └──────┬─────┘   │
════════════╪═════════════════╪═════════════════
│           │                 │         │ Kernel
│    ┌──────▼──────┐  ┌──────▼─────┐  │
│    │  RX Queue   │  │  TX Queue  │  │
│    └─────┬───────┘  └──────┬─────┘  │
├──────────┼──────────────────┼────────┤
│       XDP Program (eBPF)    │        │
├─────────────────────────────┼────────┤
│        Network Driver        │        │
├──────────────────────────────────────┤
│             NIC                       │
└──────────────────────────────────────┘
```

### Implementation Approach
```rust
use xsk_rs::{Socket, FrameDesc, Umem, Config};

pub struct AfXdpTransport {
    socket: Socket<'static>,
    umem: Umem<'static>,
}

impl AfXdpTransport {
    pub fn new(interface: &str) -> Result<Self> {
        let config = Config {
            rx_queue_size: 4096,
            tx_queue_size: 4096,
            ..Default::default()
        };
        
        let (umem, socket) = Socket::new(config, interface, 0)?;
        
        Ok(Self { socket, umem })
    }
    
    pub fn send_packet(&mut self, data: &[u8]) -> Result<()> {
        // Get frame from UMEM
        let mut frame = self.umem.frame()?;
        frame.copy_from_slice(data);
        
        // Submit to TX ring
        self.socket.tx_submit(frame)?;
        Ok(())
    }
    
    pub fn recv_packet(&mut self) -> Result<Vec<u8>> {
        // Poll RX ring
        let frame = self.socket.rx_recv()?;
        let data = frame.as_slice().to_vec();
        
        // Return frame to UMEM
        self.umem.release(frame);
        Ok(data)
    }
}
```

### eBPF Filter Example
```c
// XDP program to filter packets to AF_XDP socket
SEC("xdp")
int xdp_sock_prog(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;
    
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;
    
    // Redirect to AF_XDP socket
    return bpf_redirect_map(&xsks_map, ctx->rx_queue_index, 0);
}
```

### Pros
- ✅ Near-DPDK performance
- ✅ Simpler than DPDK
- ✅ Works with standard drivers
- ✅ Flexible eBPF filtering
- ✅ Zero-copy capable

### Cons
- ❌ Linux 4.18+ only
- ❌ Requires eBPF knowledge
- ❌ Less mature than DPDK
- ❌ Some drivers don't support zero-copy
- ❌ Complex setup

### Use Cases
- Network monitoring/packet capture
- Load balancers
- DDoS protection
- Custom protocols at line rate
- When DPDK is overkill but epoll is too slow

---

## 4. RDMA (Remote Direct Memory Access)

### Overview
Hardware-offloaded network protocol that allows direct memory access between machines without CPU involvement.

### Key Features
- Zero-copy transfers
- Kernel bypass
- <1μs latency possible
- CPU offload
- Hardware-reliable transport

### Rust Crates
```toml
rdma-sys = "0.1"          # Low-level RDMA bindings
ibverbs = "0.3"           # InfiniBand verbs
```

### Architecture
```text
Machine A                  Machine B
┌─────────────┐           ┌─────────────┐
│ Application │           │ Application │
│   Memory    │           │   Memory    │
│  ┌───────┐  │           │  ┌───────┐  │
│  │Buffer │  │           │  │Buffer │  │
│  └───┬───┘  │           │  └───▲───┘  │
└──────┼──────┘           └──────┼──────┘
       │                         │
┌──────▼──────┐           ┌──────┴──────┐
│ RDMA NIC    │═══════════│  RDMA NIC   │
│  (HCA/RNIC) │  Network  │  (HCA/RNIC) │
└─────────────┘           └─────────────┘

Direct memory-to-memory transfer
No CPU involvement!
```

### Implementation Approach
```rust
use rdma_sys::*;

pub struct RdmaTransport {
    context: *mut ibv_context,
    pd: *mut ibv_pd,
    qp: *mut ibv_qp,
    mr: *mut ibv_mr,
}

impl RdmaTransport {
    pub fn rdma_write(&self, local_buf: &[u8], remote_addr: u64, rkey: u32) -> Result<()> {
        let mut sge = ibv_sge {
            addr: local_buf.as_ptr() as u64,
            length: local_buf.len() as u32,
            lkey: unsafe { (*self.mr).lkey },
        };
        
        let mut wr = ibv_send_wr {
            wr_id: 1,
            sg_list: &mut sge,
            num_sge: 1,
            opcode: ibv_wr_opcode::IBV_WR_RDMA_WRITE,
            wr: ibv_send_wr__bindgen_ty_1 {
                rdma: ibv_send_wr__bindgen_ty_1__bindgen_ty_1 {
                    remote_addr,
                    rkey,
                },
            },
            ..Default::default()
        };
        
        unsafe {
            let mut bad_wr = std::ptr::null_mut();
            ibv_post_send(self.qp, &mut wr, &mut bad_wr);
        }
        
        Ok(())
    }
    
    pub fn rdma_read(&self, local_buf: &mut [u8], remote_addr: u64, rkey: u32) -> Result<()> {
        // Similar to write but with IBV_WR_RDMA_READ
        // ...
        Ok(())
    }
}
```

### Protocols
- **InfiniBand**: Native RDMA, highest performance
- **RoCE** (RDMA over Converged Ethernet): RDMA over Ethernet
- **iWARP**: RDMA over TCP/IP

### Pros
- ✅ Absolute lowest latency
- ✅ Zero CPU overhead for transfers
- ✅ Hardware-reliable delivery
- ✅ One-sided operations (read/write without remote CPU)
- ✅ Perfect for distributed shared memory

### Cons
- ❌ Requires specialized hardware ($$$$)
- ❌ Complex programming model
- ❌ Not available in cloud (mostly)
- ❌ Limited ecosystem
- ❌ Network configuration complexity

### Use Cases
- Distributed databases (ScyllaDB, Cassandra)
- HPC clusters
- Distributed machine learning
- Storage systems (Ceph with RoCE)
- Financial trading platforms

---

## 5. Raw Sockets

### Overview
Direct access to network layer, bypassing transport protocols.

### Key Features
- Custom protocol implementation
- Access to IP layer
- Full packet control
- Stateless communication

### Rust Crates
```toml
pnet = "0.34"             # Packet manipulation
socket2 = "0.5"           # Low-level socket control
libc = "0.2"              # Direct syscalls
```

### Implementation Approach
```rust
use socket2::{Socket, Domain, Type, Protocol};
use pnet::packet::ip::IpNextHeaderProtocols;

pub struct RawSocketTransport {
    socket: Socket,
}

impl RawSocketTransport {
    pub fn new() -> Result<Self> {
        // Requires CAP_NET_RAW capability
        let socket = Socket::new(
            Domain::IPV4,
            Type::RAW,
            Some(Protocol::from(IpNextHeaderProtocols::Tcp))
        )?;
        
        Ok(Self { socket })
    }
    
    pub fn send_raw_packet(&self, packet: &[u8], dest: SocketAddr) -> Result<()> {
        let addr = dest.into();
        self.socket.send_to(packet, &addr)?;
        Ok(())
    }
    
    pub fn recv_raw_packet(&self) -> Result<(Vec<u8>, SocketAddr)> {
        let mut buf = vec![0u8; 65536];
        let (n, addr) = self.socket.recv_from(&mut buf)?;
        buf.truncate(n);
        Ok((buf, addr.as_socket().unwrap()))
    }
}

// Build custom TCP packet
use pnet::packet::tcp::MutableTcpPacket;
use pnet::packet::Packet;

fn build_custom_tcp(data: &[u8]) -> Vec<u8> {
    let mut packet = vec![0u8; 20 + data.len()]; // 20 = TCP header
    let mut tcp = MutableTcpPacket::new(&mut packet).unwrap();
    
    tcp.set_source(8080);
    tcp.set_destination(80);
    tcp.set_sequence(12345);
    tcp.set_flags(0b00010010); // SYN + ACK
    tcp.set_payload(data);
    
    packet
}
```

### Pros
- ✅ Full protocol control
- ✅ Custom network protocols
- ✅ Stateless operation
- ✅ Educational value
- ✅ Works on all platforms

### Cons
- ❌ Requires root/capabilities
- ❌ Must implement TCP/IP yourself
- ❌ Error-prone
- ❌ No automatic retransmission
- ❌ Security implications

### Use Cases
- Custom network protocols
- Network diagnostics tools (ping, traceroute)
- Protocol research
- Packet injection/manipulation
- Network security tools

---

## 6. eBPF/XDP

### Overview
Extended Berkeley Packet Filter - programmable packet processing in the kernel.

### Key Features
- Process packets in kernel
- Filter before userspace
- Programmable with C/Rust
- Safe sandboxed execution
- Zero-copy to userspace

### Rust Crates
```toml
aya = "0.11"              # eBPF framework (Rust)
redbpf = "2.3"            # Alternative eBPF framework
libbpf-rs = "0.21"        # libbpf bindings
```

### Architecture
```text
┌─────────────────────────────────┐
│       Userspace Program         │
│  ┌──────────────────────────┐   │
│  │   Control & Statistics   │   │
│  └───────────┬──────────────┘   │
═══════════════╪═══════════════════════
│              │                  │
│  ┌───────────▼──────────────┐   │
│  │    eBPF Maps (shared)    │   │ Kernel
│  └───────────┬──────────────┘   │
│              │                  │
│  ┌───────────▼──────────────┐   │
│  │    XDP Program (eBPF)    │   │
│  │  ┌────────────────────┐  │   │
│  │  │ Filter / Forward   │  │   │
│  │  │ Modify / Drop      │  │   │
│  │  └────────────────────┘  │   │
│  └───────────┬──────────────┘   │
├──────────────┼──────────────────┤
│       Network Driver             │
└──────────────────────────────────┘
```

### Implementation Approach (Aya Framework)

**eBPF Program (runs in kernel):**
```rust
// my-ebpf/src/main.rs
#![no_std]
#![no_main]

use aya_bpf::{
    bindings::xdp_action,
    macros::xdp,
    programs::XdpContext,
};

#[xdp]
pub fn packet_filter(ctx: XdpContext) -> u32 {
    match process_packet(&ctx) {
        Ok(action) => action,
        Err(_) => xdp_action::XDP_PASS,
    }
}

fn process_packet(ctx: &XdpContext) -> Result<u32, ()> {
    // Parse ethernet header
    let eth_hdr = ctx.data()?;
    
    // Filter logic
    if should_drop_packet(eth_hdr) {
        return Ok(xdp_action::XDP_DROP);
    }
    
    if should_redirect(eth_hdr) {
        return Ok(xdp_action::XDP_REDIRECT);
    }
    
    Ok(xdp_action::XDP_PASS)
}
```

**Userspace Control Program:**
```rust
use aya::{Bpf, programs::Xdp};
use aya::maps::HashMap;

pub struct EbpfTransport {
    bpf: Bpf,
}

impl EbpfTransport {
    pub fn load(interface: &str) -> Result<Self> {
        // Load eBPF program
        let mut bpf = Bpf::load_file("my-ebpf.o")?;
        
        // Attach XDP program to interface
        let program: &mut Xdp = bpf.program_mut("packet_filter")?.try_into()?;
        program.load()?;
        program.attach(interface, aya::programs::XdpFlags::default())?;
        
        Ok(Self { bpf })
    }
    
    pub fn get_stats(&self) -> Result<PacketStats> {
        // Read statistics from eBPF map
        let stats_map: HashMap<_, u32, u64> = 
            HashMap::try_from(self.bpf.map("stats")?)?;
        
        let dropped = stats_map.get(&0, 0)?;
        let passed = stats_map.get(&1, 0)?;
        
        Ok(PacketStats { dropped, passed })
    }
}
```

### Pros
- ✅ Kernel-level performance
- ✅ Programmable packet processing
- ✅ Safe (verified by kernel)
- ✅ Rich observability
- ✅ Growing ecosystem

### Cons
- ❌ Linux-only
- ❌ Limited instruction set
- ❌ Debugging difficulty
- ❌ Kernel version dependencies
- ❌ Learning curve

### Use Cases
- Network monitoring (Cilium, Falco)
- Load balancing
- DDoS mitigation
- Custom packet filtering
- Observability/tracing

---

## 7. epoll/kqueue Direct

### Overview
Direct use of kernel event notification mechanisms without abstraction layers.

### Key Features
- Platform's native async I/O
- Minimal overhead
- Full control
- No framework dependencies

### Implementation Approach

**Linux (epoll):**
```rust
use libc::{epoll_create1, epoll_ctl, epoll_wait, epoll_event};
use std::os::unix::io::RawFd;

pub struct EpollTransport {
    epoll_fd: RawFd,
    sockets: Vec<RawFd>,
}

impl EpollTransport {
    pub fn new() -> Result<Self> {
        let epoll_fd = unsafe { epoll_create1(0) };
        if epoll_fd < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        
        Ok(Self {
            epoll_fd,
            sockets: Vec::new(),
        })
    }
    
    pub fn add_socket(&mut self, fd: RawFd) -> Result<()> {
        let mut event = epoll_event {
            events: (libc::EPOLLIN | libc::EPOLLET) as u32,
            u64: fd as u64,
        };
        
        unsafe {
            if epoll_ctl(self.epoll_fd, libc::EPOLL_CTL_ADD, fd, &mut event) < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
        }
        
        self.sockets.push(fd);
        Ok(())
    }
    
    pub fn poll(&self, timeout_ms: i32) -> Result<Vec<Event>> {
        let mut events = vec![epoll_event { events: 0, u64: 0 }; 128];
        
        let n = unsafe {
            epoll_wait(
                self.epoll_fd,
                events.as_mut_ptr(),
                events.len() as i32,
                timeout_ms,
            )
        };
        
        if n < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        
        Ok(events[..n as usize]
            .iter()
            .map(|e| Event {
                fd: e.u64 as RawFd,
                readable: (e.events & libc::EPOLLIN as u32) != 0,
                writable: (e.events & libc::EPOLLOUT as u32) != 0,
            })
            .collect())
    }
}
```

**macOS/BSD (kqueue):**
```rust
use libc::{kqueue, kevent, timespec};
use std::os::unix::io::RawFd;

pub struct KqueueTransport {
    kq: RawFd,
}

impl KqueueTransport {
    pub fn new() -> Result<Self> {
        let kq = unsafe { kqueue() };
        if kq < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        
        Ok(Self { kq })
    }
    
    pub fn add_socket(&self, fd: RawFd) -> Result<()> {
        let mut event = kevent {
            ident: fd as usize,
            filter: libc::EVFILT_READ,
            flags: libc::EV_ADD | libc::EV_ENABLE,
            fflags: 0,
            data: 0,
            udata: std::ptr::null_mut(),
        };
        
        unsafe {
            if kevent(self.kq, &event, 1, std::ptr::null_mut(), 0, std::ptr::null()) < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
        }
        
        Ok(())
    }
    
    pub fn poll(&self, timeout_ms: i32) -> Result<Vec<Event>> {
        let mut events = vec![kevent::default(); 128];
        let timeout = timespec {
            tv_sec: (timeout_ms / 1000) as _,
            tv_nsec: ((timeout_ms % 1000) * 1_000_000) as _,
        };
        
        let n = unsafe {
            kevent(
                self.kq,
                std::ptr::null(),
                0,
                events.as_mut_ptr(),
                events.len() as i32,
                &timeout,
            )
        };
        
        if n < 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        
        Ok(events[..n as usize]
            .iter()
            .map(|e| Event {
                fd: e.ident as RawFd,
                readable: e.filter == libc::EVFILT_READ,
                writable: e.filter == libc::EVFILT_WRITE,
            })
            .collect())
    }
}
```

### Pros
- ✅ Maximum control
- ✅ Minimal abstraction overhead
- ✅ Platform-specific optimization
- ✅ No dependencies
- ✅ Educational value

### Cons
- ❌ Platform-specific code
- ❌ More code to maintain
- ❌ Easy to make mistakes
- ❌ Must handle edge cases
- ❌ No high-level features

### Use Cases
- When you need absolute control
- Educational/learning
- Performance-critical custom protocols
- Building your own async runtime
- Embedded systems

---

## 8. Kernel Modules (Extreme)

### Overview
Write network processing code that runs in kernel space.

### Key Features
- Lowest possible latency
- Direct hardware access
- No userspace overhead
- Complete system control

### Rust Crates
```toml
# Out-of-tree kernel modules
kernel = { git = "https://github.com/Rust-for-Linux/linux" }
```

### Implementation Approach
```rust
// kernel_module/src/lib.rs
#![no_std]

use kernel::prelude::*;
use kernel::net::{NetDevice, SkBuff};

module! {
    type: SharedNothingNet,
    name: "shared_nothing_net",
    license: "GPL",
}

struct SharedNothingNet;

impl kernel::Module for SharedNothingNet {
    fn init(_module: &'static ThisModule) -> Result<Self> {
        pr_info!("Shared-Nothing Network Module Loading\n");
        
        // Register network hooks
        register_netfilter_hooks()?;
        
        Ok(SharedNothingNet)
    }
}

// Netfilter hook for packet interception
fn packet_hook(skb: &mut SkBuff) -> kernel::net::NfAction {
    // Process packet in kernel space
    let data = skb.data();
    
    // Custom protocol logic
    if is_shared_nothing_packet(data) {
        process_packet_kernel(skb);
        return kernel::net::NfAction::Stolen; // Packet consumed
    }
    
    kernel::net::NfAction::Accept
}

fn process_packet_kernel(skb: &mut SkBuff) {
    // Zero-copy processing
    // Direct memory access
    // No context switches
}
```

### Pros
- ✅ Absolute minimum latency (<100ns)
- ✅ Zero context switches
- ✅ Direct hardware access
- ✅ Maximum performance possible

### Cons
- ❌ Extreme complexity
- ❌ Kernel panics crash system
- ❌ Difficult debugging
- ❌ Limited Rust support (improving)
- ❌ Security/stability risks
- ❌ Requires deep kernel knowledge

### Use Cases
- Specialized hardware appliances
- Research projects
- When microseconds matter
- Custom network stacks
- **Generally not recommended**

---

## Recommendation Matrix

### For Production Systems

| Requirement | Recommended | Alternative |
|------------|-------------|-------------|
| **Best General Performance** | io_uring | Tokio (async/await) |
| **Maximum Throughput** | DPDK | AF_XDP |
| **Lowest Latency** | RDMA | DPDK |
| **Cross-Platform** | epoll/kqueue | mio/Tokio |
| **Ease of Use** | Tokio | mio |
| **Cloud Environments** | io_uring | Standard TCP/UDP |
| **Custom Protocols** | Raw Sockets | eBPF/XDP |
| **Packet Filtering** | eBPF/XDP | AF_XDP |

### Decision Tree

```text
Start Here
    │
    ├─ Need absolute maximum performance?
    │   ├─ Yes, have RDMA hardware? ──> RDMA
    │   ├─ Yes, can dedicate NICs? ──> DPDK
    │   └─ Yes, Linux only? ──> io_uring / AF_XDP
    │
    ├─ Need custom packet processing?
    │   ├─ In kernel? ──> eBPF/XDP
    │   └─ In userspace? ──> Raw Sockets / AF_XDP
    │
    ├─ Need cross-platform?
    │   ├─ High-level? ──> Tokio
    │   └─ Low-level? ──> epoll/kqueue direct
    │
    └─ Standard distributed system?
        └─ io_uring (Linux) or Tokio (all platforms)
```

## Integration with Shared-Nothing Library

### Recommended Approach: io_uring

```rust
// Add to lib.rs
pub mod network {
    pub mod io_uring;
    pub mod transport;
}

// src/network/transport.rs
pub trait NetworkTransport: Send + Sync {
    async fn send(&self, addr: SocketAddr, data: &[u8]) -> Result<()>;
    async fn recv(&self) -> Result<(Vec<u8>, SocketAddr)>;
}

// src/network/io_uring.rs
pub struct IoUringNetworkTransport {
    runtime: tokio_uring::Runtime,
    listeners: HashMap<SocketAddr, TcpListener>,
}

impl NetworkTransport for IoUringNetworkTransport {
    async fn send(&self, addr: SocketAddr, data: &[u8]) -> Result<()> {
        let stream = TcpStream::connect(addr).await?;
        stream.write_all(data).await?;
        Ok(())
    }
    
    async fn recv(&self) -> Result<(Vec<u8>, SocketAddr)> {
        // Implementation
        Ok((vec![], "0.0.0.0:0".parse()?))
    }
}
```

## Performance Testing Plan

1. **Microbenchmarks**: Measure raw send/recv latency
2. **Throughput Tests**: Messages per second at scale
3. **Scalability Tests**: Performance vs. number of nodes
4. **Latency Distribution**: p50, p99, p999, p9999
5. **Comparison**: Baseline vs. each technology

## Rust Libraries & Crates Detailed Reference

### io_uring Libraries

#### tokio-uring
- **Crate**: `tokio-uring = "0.4"`
- **GitHub**: https://github.com/tokio-rs/tokio-uring
- **Maturity**: Beta / Production Ready
- **Maintained By**: Tokio team
- **Documentation**: ⭐⭐⭐⭐ (Good)
- **Last Updated**: Active development
- **Features**:
  - Drop-in replacement for tokio
  - Async/await support
  - File and network I/O
  - Zero-copy operations
- **Pros**: Best integration with Tokio ecosystem
- **Cons**: Still stabilizing API
- **Example**:
  ```rust
  use tokio_uring::net::TcpListener;
  
  tokio_uring::start(async {
      let listener = TcpListener::bind("127.0.0.1:8080".parse()?).await?;
      loop {
          let (stream, addr) = listener.accept().await?;
          tokio_uring::spawn(async move {
              // Handle connection
          });
      }
  });
  ```

#### io-uring
- **Crate**: `io-uring = "0.6"`
- **GitHub**: https://github.com/tokio-rs/io-uring
- **Maturity**: Stable
- **Documentation**: ⭐⭐⭐⭐⭐ (Excellent)
- **Features**:
  - Direct liburing bindings
  - Low-level control
  - Synchronous and async APIs
  - Full feature parity with C liburing
- **Pros**: Maximum control, all features
- **Cons**: Lower-level, more complex
- **Example**:
  ```rust
  use io_uring::{IoUring, opcode, types};
  
  let mut ring = IoUring::new(256)?;
  let sockfd = /* socket fd */;
  
  let read_e = opcode::Read::new(types::Fd(sockfd), buf.as_mut_ptr(), buf.len() as _)
      .build()
      .user_data(0x42);
  
  unsafe { ring.submission().push(&read_e)?; }
  ring.submit_and_wait(1)?;
  ```

#### glommio
- **Crate**: `glommio = "0.8"`
- **GitHub**: https://github.com/DataDog/glommio
- **Maturity**: Production Ready
- **Maintained By**: DataDog
- **Documentation**: ⭐⭐⭐⭐ (Good)
- **Features**:
  - Thread-per-core architecture
  - io_uring based
  - Task scheduling
  - NUMA-aware
- **Pros**: Designed for high-performance servers
- **Cons**: Different programming model
- **Best For**: Thread-per-core architectures

---

### DPDK Libraries

#### dpdk-rs
- **Crate**: `dpdk-rs = "0.1"`
- **GitHub**: https://github.com/mesalock-linux/rust-dpdk
- **Maturity**: Alpha / Experimental
- **Documentation**: ⭐⭐ (Limited)
- **Status**: Not actively maintained
- **Features**:
  - Basic DPDK bindings
  - EAL initialization
  - Memory management
- **Pros**: Direct DPDK access
- **Cons**: Incomplete, outdated

#### capsule
- **Crate**: `capsule = "0.1"`
- **GitHub**: https://github.com/capsule-rs/capsule
- **Maturity**: Beta
- **Maintained By**: Capsule team
- **Documentation**: ⭐⭐⭐ (Fair)
- **Features**:
  - High-level DPDK framework
  - Safe abstractions
  - Packet processing pipelines
  - Multiple port support
- **Pros**: Safer than raw DPDK
- **Cons**: Still maturing
- **Example**:
  ```rust
  use capsule::prelude::*;
  
  fn install(pipeline: &mut Pipeline) -> Result<()> {
      pipeline.port("eth0")?.add_group(group! {
          my_packet_processor
      })?;
      Ok(())
  }
  ```

#### dpdk-sys
- **Crate**: `dpdk-sys` (via git)
- **Type**: Low-level bindings
- **Status**: Community maintained
- **Use**: Building your own abstractions

#### Alternative: Use C DPDK directly
- **Approach**: FFI bindings
- **Maturity**: ⭐⭐⭐⭐⭐ (DPDK itself is mature)
- **Docs**: Official DPDK docs apply
- **Best For**: When Rust crates are insufficient

---

### AF_XDP Libraries

#### xsk-rs
- **Crate**: `xsk-rs = "0.2"`
- **GitHub**: https://github.com/DouglasGray/xsk-rs
- **Maturity**: Beta
- **Documentation**: ⭐⭐⭐⭐ (Good)
- **Features**:
  - Safe AF_XDP socket wrapper
  - Zero-copy and copy modes
  - UMEM management
  - Frame handling
- **Pros**: Well-documented, safe API
- **Cons**: Linux-specific, requires eBPF
- **Example**:
  ```rust
  use xsk_rs::{Socket, Config, Umem};
  
  let config = Config::default();
  let (mut umem, mut socket) = Socket::new(config, "eth0", 0)?;
  
  // Send packet
  let frame = umem.frame()?;
  socket.tx_submit(frame)?;
  
  // Receive packet
  let frame = socket.rx_recv()?;
  ```

#### libbpf-rs
- **Crate**: `libbpf-rs = "0.21"`
- **GitHub**: https://github.com/libbpf/libbpf-rs
- **Maturity**: Stable
- **Maintained By**: libbpf team
- **Documentation**: ⭐⭐⭐⭐⭐ (Excellent)
- **Features**:
  - eBPF program loading
  - Map manipulation
  - XDP attachment
  - CO-RE support
- **Pros**: Official bindings, complete
- **Cons**: Requires libbpf installed
- **Use With**: AF_XDP for eBPF programs

---

### RDMA Libraries

#### rdma-sys
- **Crate**: `rdma-sys = "0.1"`
- **GitHub**: https://github.com/jonhoo/rust-rdma
- **Maturity**: Alpha
- **Documentation**: ⭐⭐ (Minimal)
- **Status**: Experimental
- **Features**:
  - Raw ibverbs bindings
  - Unsafe API
- **Cons**: Very low-level, incomplete

#### ibverbs
- **Crate**: `ibverbs = "0.3"`
- **Type**: Bindings to libibverbs
- **Maturity**: Beta
- **Features**:
  - InfiniBand verbs
  - RDMA operations
  - Queue pair management
- **Pros**: More complete than rdma-sys
- **Cons**: Still low-level, unsafe

#### async-rdma
- **Crate**: `async-rdma = "0.4"`
- **GitHub**: https://github.com/datenlord/async-rdma
- **Maturity**: Beta
- **Maintained By**: DatenLord
- **Documentation**: ⭐⭐⭐ (Fair)
- **Features**:
  - Async RDMA operations
  - High-level API
  - Memory region management
  - Connection management
- **Pros**: Async/await, safer abstractions
- **Cons**: Relatively new
- **Example**:
  ```rust
  use async_rdma::{Rdma, RdmaBuilder};
  
  let rdma = RdmaBuilder::default()
      .connect("192.168.1.100:7471")
      .await?;
  
  rdma.send(b"Hello RDMA").await?;
  let data = rdma.receive().await?;
  ```

#### Alternative: Use C libraries (ibverbs, rdma-core)
- **Approach**: Direct FFI
- **Maturity**: ⭐⭐⭐⭐⭐ (C libraries are mature)
- **Best For**: Production systems until Rust crates mature

---

### Raw Sockets Libraries

#### pnet
- **Crate**: `pnet = "0.34"`
- **GitHub**: https://github.com/libpnet/libpnet
- **Maturity**: Stable
- **Documentation**: ⭐⭐⭐⭐⭐ (Excellent)
- **Features**:
  - Packet construction/parsing
  - Multiple protocol support
  - Raw socket handling
  - Cross-platform
- **Pros**: Comprehensive, well-maintained
- **Cons**: Sync only
- **Example**:
  ```rust
  use pnet::packet::tcp::MutableTcpPacket;
  use pnet::transport::{transport_channel, TransportChannelType};
  
  let (mut tx, _) = transport_channel(4096, TransportChannelType::Layer4(...))?;
  
  let mut packet = vec![0u8; 20];
  let mut tcp = MutableTcpPacket::new(&mut packet).unwrap();
  tcp.set_source(8080);
  tcp.set_destination(80);
  
  tx.send_to(tcp, dst_addr)?;
  ```

#### socket2
- **Crate**: `socket2 = "0.5"`
- **GitHub**: https://github.com/rust-lang/socket2
- **Maturity**: Stable / Production Ready
- **Documentation**: ⭐⭐⭐⭐⭐ (Excellent)
- **Features**:
  - Cross-platform socket control
  - Raw sockets
  - Socket options
  - IPv4/IPv6 support
- **Pros**: Official Rust project, reliable
- **Cons**: Lower-level than pnet
- **Best For**: Building custom socket abstractions

#### smoltcp
- **Crate**: `smoltcp = "0.11"`
- **GitHub**: https://github.com/smoltcp-rs/smoltcp
- **Maturity**: Stable
- **Documentation**: ⭐⭐⭐⭐ (Good)
- **Features**:
  - Pure Rust TCP/IP stack
  - No std support
  - Embedded friendly
  - Custom protocols
- **Pros**: No kernel dependency
- **Cons**: Must implement everything
- **Best For**: Embedded systems, custom stacks

---

### eBPF/XDP Libraries

#### aya
- **Crate**: `aya = "0.11"`
- **GitHub**: https://github.com/aya-rs/aya
- **Maturity**: Stable
- **Documentation**: ⭐⭐⭐⭐⭐ (Excellent)
- **Features**:
  - Write eBPF in Rust
  - XDP, TC, LSM programs
  - CO-RE support
  - No C toolchain needed
- **Pros**: Pure Rust workflow, modern
- **Cons**: Requires nightly for eBPF code
- **Example**:
  ```rust
  // Userspace
  use aya::{Bpf, programs::Xdp};
  
  let mut bpf = Bpf::load_file("bpf.o")?;
  let program: &mut Xdp = bpf.program_mut("xdp_prog")?.try_into()?;
  program.attach("eth0", Default::default())?;
  
  // eBPF program (separate crate)
  #[xdp]
  pub fn xdp_prog(ctx: XdpContext) -> u32 {
      // Process packet
      xdp_action::XDP_PASS
  }
  ```

#### redbpf
- **Crate**: `redbpf = "2.3"`
- **GitHub**: https://github.com/foniod/redbpf
- **Maturity**: Stable
- **Documentation**: ⭐⭐⭐⭐ (Good)
- **Features**:
  - Rust eBPF programs
  - Multiple program types
  - Async support
- **Pros**: Mature, async support
- **Cons**: More complex than Aya
- **Best For**: Production eBPF applications

#### libbpf-rs (see AF_XDP section)
- Can be used standalone for eBPF
- Most compatible with kernel

#### bpf-linker
- **Crate**: `bpf-linker`
- **Type**: Development tool
- **Purpose**: Link Rust eBPF programs
- **Required For**: Aya and redbpf

---

### epoll/kqueue Direct

#### mio
- **Crate**: `mio = "0.8"`
- **GitHub**: https://github.com/tokio-rs/mio
- **Maturity**: Stable / Production Ready
- **Documentation**: ⭐⭐⭐⭐⭐ (Excellent)
- **Features**:
  - Cross-platform event loop
  - epoll/kqueue/IOCP abstraction
  - Non-blocking I/O
  - Used by Tokio
- **Pros**: Battle-tested, cross-platform
- **Cons**: Low-level API
- **Example**:
  ```rust
  use mio::{Events, Interest, Poll, Token};
  use mio::net::TcpListener;
  
  let mut poll = Poll::new()?;
  let mut events = Events::with_capacity(128);
  
  let mut listener = TcpListener::bind("127.0.0.1:8080".parse()?)?;
  poll.registry().register(&mut listener, Token(0), Interest::READABLE)?;
  
  loop {
      poll.poll(&mut events, None)?;
      for event in events.iter() {
          // Handle event
      }
  }
  ```

#### polling
- **Crate**: `polling = "3.3"`
- **GitHub**: https://github.com/smol-rs/polling
- **Maturity**: Stable
- **Documentation**: ⭐⭐⭐⭐ (Good)
- **Features**:
  - Minimal epoll/kqueue wrapper
  - Simpler than mio
  - Cross-platform
- **Pros**: Lightweight, simple API
- **Cons**: Less features than mio
- **Best For**: Simple event loops

#### Direct libc
- **Crate**: `libc = "0.2"`
- **Approach**: Direct syscalls
- **Documentation**: ⭐⭐⭐⭐⭐ (libc docs)
- **Features**: Everything the kernel supports
- **Pros**: Maximum control
- **Cons**: Platform-specific, unsafe
- **Best For**: When you need specific features

---

### Kernel Modules

#### Rust-for-Linux
- **Project**: https://github.com/Rust-for-Linux
- **Status**: In-tree (Linux 6.1+)
- **Maturity**: Experimental
- **Documentation**: ⭐⭐⭐ (Growing)
- **Features**:
  - Write kernel modules in Rust
  - Safe abstractions for kernel APIs
  - Netfilter hooks
  - Network device drivers
- **Requirements**: Linux kernel source, specific toolchain
- **Example**:
  ```rust
  use kernel::prelude::*;
  
  module! {
      type: MyNetModule,
      name: "my_net_module",
      license: "GPL",
  }
  
  struct MyNetModule;
  
  impl kernel::Module for MyNetModule {
      fn init(_module: &'static ThisModule) -> Result<Self> {
          pr_info!("Module loaded\n");
          Ok(MyNetModule)
      }
  }
  ```

#### netfilter_rs
- **Type**: Bindings for netfilter
- **Status**: Community project
- **Maturity**: Alpha
- **Use**: Packet filtering in kernel

---

## Library Comparison Matrix

### Maturity & Production Readiness

| Crate | Maturity | Production Ready | Active Maintenance | Community |
|-------|----------|------------------|-------------------|-----------|
| **tokio-uring** | ⭐⭐⭐⭐ | ✅ Yes | ✅ Active | Large |
| **io-uring** | ⭐⭐⭐⭐⭐ | ✅ Yes | ✅ Active | Large |
| **glommio** | ⭐⭐⭐⭐ | ✅ Yes | ✅ Active | Medium |
| **capsule** | ⭐⭐⭐ | ⚠️ Beta | ✅ Active | Small |
| **dpdk-rs** | ⭐⭐ | ❌ No | ❌ Stale | Small |
| **xsk-rs** | ⭐⭐⭐⭐ | ⚠️ Beta | ✅ Active | Small |
| **libbpf-rs** | ⭐⭐⭐⭐⭐ | ✅ Yes | ✅ Active | Large |
| **async-rdma** | ⭐⭐⭐ | ⚠️ Beta | ✅ Active | Small |
| **rdma-sys** | ⭐⭐ | ❌ No | ❌ Stale | Small |
| **pnet** | ⭐⭐⭐⭐⭐ | ✅ Yes | ✅ Active | Large |
| **socket2** | ⭐⭐⭐⭐⭐ | ✅ Yes | ✅ Active | Large |
| **smoltcp** | ⭐⭐⭐⭐ | ✅ Yes | ✅ Active | Medium |
| **aya** | ⭐⭐⭐⭐⭐ | ✅ Yes | ✅ Active | Large |
| **redbpf** | ⭐⭐⭐⭐ | ✅ Yes | ✅ Active | Medium |
| **mio** | ⭐⭐⭐⭐⭐ | ✅ Yes | ✅ Active | Large |
| **polling** | ⭐⭐⭐⭐ | ✅ Yes | ✅ Active | Medium |

### Feature Coverage

| Technology | Best Crate | Alternative | Async Support | Zero-Copy | Cross-Platform |
|-----------|-----------|-------------|---------------|-----------|----------------|
| **io_uring** | tokio-uring | glommio | ✅ | ✅ | ❌ Linux only |
| **DPDK** | capsule | dpdk-sys | ❌ | ✅ | ⚠️ Limited |
| **AF_XDP** | xsk-rs | - | ❌ | ✅ | ❌ Linux only |
| **RDMA** | async-rdma | rdma-sys | ✅ | ✅ | ⚠️ Limited |
| **Raw Sockets** | pnet | socket2 | ❌ | ⚠️ | ✅ |
| **eBPF/XDP** | aya | redbpf | ✅ | ✅ | ❌ Linux only |
| **epoll/kqueue** | mio | polling | ❌ | ❌ | ✅ |

### Installation & Setup Difficulty

| Crate | Difficulty | System Requirements | Additional Setup |
|-------|-----------|---------------------|------------------|
| **tokio-uring** | ⭐ Easy | Linux 5.1+ | None |
| **io-uring** | ⭐⭐ Medium | Linux 5.1+ | liburing |
| **glommio** | ⭐⭐ Medium | Linux 5.8+ | None |
| **capsule** | ⭐⭐⭐⭐ Hard | DPDK setup | DPDK, huge pages, dedicated NICs |
| **xsk-rs** | ⭐⭐⭐ Medium | Linux 4.18+ | eBPF program, NIC driver support |
| **libbpf-rs** | ⭐⭐ Medium | Linux 4.8+ | libbpf, kernel headers |
| **async-rdma** | ⭐⭐⭐⭐ Hard | RDMA hardware | RDMA drivers, ibverbs |
| **pnet** | ⭐ Easy | Any | Root/CAP_NET_RAW |
| **aya** | ⭐⭐ Medium | Linux 5.x+ | bpf-linker |
| **mio** | ⭐ Easy | Any | None |

## Recommended Crate Selection

### For Shared-Nothing Library

**Phase 1: MVP (Cross-Platform)**
```toml
[dependencies]
tokio = { version = "1.40", features = ["full"] }
socket2 = "0.5"
```

**Phase 2: High-Performance (Linux)**
```toml
[dependencies]
tokio-uring = "0.4"
# OR
glommio = "0.8"
```

**Phase 3: Specialized (Optional)**
```toml
[dependencies]
# For packet filtering
aya = "0.11"

# For RDMA support  
async-rdma = "0.4"

# For custom protocols
pnet = "0.34"
```

### Version Compatibility

```toml
# Tested combinations
tokio = "1.40"
tokio-uring = "0.4"
mio = "0.8"
socket2 = "0.5"
pnet = "0.34"
aya = "0.11"
libbpf-rs = "0.21"
```

## Learning Resources

### Documentation & Tutorials

**io_uring**
- Official: https://kernel.dk/io_uring.pdf
- Rust: https://github.com/tokio-rs/tokio-uring/tree/master/examples

**DPDK**
- Official: https://doc.dpdk.org/
- Rust: https://github.com/capsule-rs/capsule/tree/main/examples

**AF_XDP**
- Kernel docs: https://www.kernel.org/doc/html/latest/networking/af_xdp.html
- Rust: https://github.com/DouglasGray/xsk-rs/tree/main/examples

**RDMA**
- Programming guide: https://www.rdmamojo.com/
- Rust: https://github.com/datenlord/async-rdma/tree/master/examples

**eBPF/XDP**
- Learn eBPF: https://ebpf.io/
- Aya book: https://aya-rs.dev/book/

**Raw Sockets**
- pnet guide: https://github.com/libpnet/libpnet/tree/master/examples

### Community & Support

| Technology | Discord/Chat | Forum | Stack Overflow | GitHub Issues |
|-----------|-------------|-------|----------------|---------------|
| **Tokio/io_uring** | ✅ Discord | ✅ Forum | ✅ Active | ✅ Responsive |
| **DPDK** | ❌ | ✅ Mailing list | ⚠️ Limited | ⚠️ Slow |
| **eBPF (Aya)** | ✅ Discord | ❌ | ⚠️ Growing | ✅ Responsive |
| **RDMA** | ❌ | ⚠️ Limited | ⚠️ Limited | ⚠️ Slow |
| **pnet** | ❌ | ❌ | ⚠️ Limited | ✅ Active |

## Conclusion

**For the shared-nothing library:**

1. **Phase 1 (MVP)**: Start with **io_uring** (Linux) or **Tokio** (cross-platform)
2. **Phase 2 (Optimization)**: Add **AF_XDP** for specialized workloads
3. **Phase 3 (Enterprise)**: Add **RDMA** support for datacenter deployments

Avoid DPDK and kernel modules unless you have specific requirements and expertise.

---

**Document Status**: Ready for review
**Last Updated**: October 31, 2025
**Author**: Shared-Nothing Architecture Team

