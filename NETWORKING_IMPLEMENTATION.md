# Networking Implementation Plan

## Selected Technologies

Based on requirements for **maximum performance and control**, the following libraries have been selected:

| Layer | Technology | Library | Rationale |
|-------|-----------|---------|-----------|
| **Primary** | io_uring | `io-uring` | Direct liburing bindings, full control |
| **Kernel Bypass** | DPDK | C DPDK via FFI | Maximum throughput, mature ecosystem |
| **Fast Path** | AF_XDP | `xsk-rs` | Zero-copy userspace, best Rust API |
| **Packet Filter** | eBPF/XDP | `aya` | Pure Rust eBPF, modern tooling |
| **RDMA** | InfiniBand | rdma-core via FFI | Production-proven, <1μs latency |
| **Raw Sockets** | Layer 2/3 | `pnet` | Comprehensive protocol support |
| **Custom Stack** | TCP/IP | `smoltcp` | Embedded-friendly, no kernel |
| **Event Loop** | Fallback | `mio` | Cross-platform, battle-tested |
| **Kernel Modules** | Advanced | Rust-for-Linux | Minimum latency option |

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                    Shared-Nothing Library                       │
│                      (Core Abstraction)                         │
├─────────────────────────────────────────────────────────────────┤
│                    Network Transport Layer                       │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────────┐  │
│  │ io_uring │   DPDK   │  AF_XDP  │   RDMA   │   Raw/Custom │  │
│  │ (Linux)  │ (Bypass) │ (eBPF)   │ (HW)     │   (Fallback) │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                      Rust FFI Layer                             │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────────┐  │
│  │io-uring  │  DPDK    │  xsk-rs  │ rdma-core│  pnet/mio    │  │
│  │  crate   │    C     │   safe   │    C     │    safe      │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                     Kernel/Hardware                             │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────────┐  │
│  │ liburing │   PMD    │XDP+eBPF  │  ibverbs │  epoll/kqueue│  │
│  └──────────┴──────────┴──────────┴──────────┴──────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Dependencies

### Cargo.toml

```toml
[package]
name = "shared-nothing"
version = "0.2.0"
edition = "2021"

[dependencies]
# Existing dependencies
tokio = { version = "1.40", features = ["full"] }
crossbeam = "0.8"
flume = "0.11"
ahash = "0.8"
parking_lot = "0.12"
num_cpus = "1.16"

# Networking: io_uring
io-uring = "0.6"

# Networking: AF_XDP
xsk-rs = { version = "0.2", optional = true }

# Networking: eBPF/XDP
aya = { version = "0.11", optional = true }

# Networking: Raw sockets
pnet = { version = "0.34", optional = true }

# Networking: Custom TCP/IP stack
smoltcp = { version = "0.11", optional = true }

# Networking: Cross-platform event loop
mio = { version = "0.8", optional = true }

# FFI for C libraries
libc = "0.2"

# Serialization
serde = { version = "1.0", features = ["derive"] }
bincode = "1.3"

[build-dependencies]
cc = "1.0"
bindgen = "0.69"
pkg-config = "0.3"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }

[features]
default = ["io-uring", "mio"]
io-uring = ["dep:io-uring"]
af-xdp = ["dep:xsk-rs", "dep:aya"]
dpdk = []  # Requires external DPDK installation
rdma = []  # Requires rdma-core libraries
raw-sockets = ["dep:pnet"]
custom-stack = ["dep:smoltcp"]
kernel-modules = []  # Requires Rust-for-Linux setup
all-transports = ["io-uring", "af-xdp", "dpdk", "rdma", "raw-sockets", "custom-stack"]

[[example]]
name = "network_benchmark"
path = "examples/network_benchmark.rs"
```

### System Dependencies

```bash
# io_uring
sudo apt-get install liburing-dev  # Ubuntu/Debian
brew install liburing               # macOS (limited support)

# DPDK
sudo apt-get install dpdk dpdk-dev
export RTE_SDK=/usr/share/dpdk

# AF_XDP (requires kernel 4.18+)
sudo apt-get install libbpf-dev bpftool

# RDMA
sudo apt-get install rdma-core libibverbs-dev

# eBPF toolchain
cargo install bpf-linker
rustup install nightly
rustup component add rust-src --toolchain nightly

# Raw sockets (requires CAP_NET_RAW)
sudo setcap cap_net_raw=eip ./target/release/shared-nothing
```

## Implementation Structure

```
src/
├── network/
│   ├── mod.rs                 # Network module root
│   ├── transport.rs           # Transport trait
│   ├── io_uring/
│   │   ├── mod.rs
│   │   ├── tcp.rs
│   │   ├── udp.rs
│   │   └── ring.rs
│   ├── dpdk/
│   │   ├── mod.rs
│   │   ├── ffi.rs            # C FFI bindings
│   │   ├── port.rs
│   │   ├── mempool.rs
│   │   └── packet.rs
│   ├── af_xdp/
│   │   ├── mod.rs
│   │   ├── socket.rs
│   │   └── umem.rs
│   ├── rdma/
│   │   ├── mod.rs
│   │   ├── ffi.rs            # rdma-core FFI
│   │   ├── qp.rs
│   │   └── mr.rs
│   ├── raw/
│   │   ├── mod.rs
│   │   ├── ethernet.rs
│   │   └── ip.rs
│   ├── custom/
│   │   ├── mod.rs
│   │   └── smoltcp_impl.rs
│   ├── mio_fallback/
│   │   └── mod.rs
│   └── kernel/
│       └── netfilter.rs       # Rust-for-Linux
├── ffi/
│   ├── dpdk.rs                # DPDK FFI bindings
│   └── rdma.rs                # RDMA FFI bindings
└── lib.rs

build.rs                        # Build script for FFI

ebpf/                          # eBPF programs (separate workspace)
├── Cargo.toml
└── src/
    └── xdp_filter.rs
```

## Core Transport Trait

### src/network/transport.rs

```rust
use std::net::SocketAddr;
use crate::error::Result;

/// Network transport abstraction
#[async_trait::async_trait]
pub trait Transport: Send + Sync {
    /// Send data to a destination
    async fn send(&self, dest: SocketAddr, data: &[u8]) -> Result<usize>;
    
    /// Receive data from any source
    async fn recv(&self, buf: &mut [u8]) -> Result<(usize, SocketAddr)>;
    
    /// Get transport name
    fn name(&self) -> &str;
    
    /// Get transport statistics
    fn stats(&self) -> TransportStats;
    
    /// Close the transport
    async fn close(&self) -> Result<()>;
}

/// Transport statistics
#[derive(Debug, Clone, Default)]
pub struct TransportStats {
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub packets_sent: u64,
    pub packets_received: u64,
    pub errors: u64,
    pub latency_us: u64,
}

/// Transport type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportType {
    IoUring,
    Dpdk,
    AfXdp,
    Rdma,
    RawSocket,
    CustomStack,
    Mio,
}

/// Create a transport of the specified type
pub async fn create_transport(
    transport_type: TransportType,
    config: TransportConfig,
) -> Result<Box<dyn Transport>> {
    match transport_type {
        TransportType::IoUring => {
            #[cfg(feature = "io-uring")]
            { Ok(Box::new(crate::network::io_uring::IoUringTransport::new(config).await?)) }
            #[cfg(not(feature = "io-uring"))]
            { Err(crate::error::Error::Other("io_uring not enabled".into())) }
        }
        TransportType::Dpdk => {
            #[cfg(feature = "dpdk")]
            { Ok(Box::new(crate::network::dpdk::DpdkTransport::new(config)?)) }
            #[cfg(not(feature = "dpdk"))]
            { Err(crate::error::Error::Other("DPDK not enabled".into())) }
        }
        TransportType::AfXdp => {
            #[cfg(feature = "af-xdp")]
            { Ok(Box::new(crate::network::af_xdp::AfXdpTransport::new(config).await?)) }
            #[cfg(not(feature = "af-xdp"))]
            { Err(crate::error::Error::Other("AF_XDP not enabled".into())) }
        }
        TransportType::Rdma => {
            #[cfg(feature = "rdma")]
            { Ok(Box::new(crate::network::rdma::RdmaTransport::new(config)?)) }
            #[cfg(not(feature = "rdma"))]
            { Err(crate::error::Error::Other("RDMA not enabled".into())) }
        }
        TransportType::RawSocket => {
            #[cfg(feature = "raw-sockets")]
            { Ok(Box::new(crate::network::raw::RawSocketTransport::new(config)?)) }
            #[cfg(not(feature = "raw-sockets"))]
            { Err(crate::error::Error::Other("Raw sockets not enabled".into())) }
        }
        TransportType::CustomStack => {
            #[cfg(feature = "custom-stack")]
            { Ok(Box::new(crate::network::custom::SmoltcpTransport::new(config)?)) }
            #[cfg(not(feature = "custom-stack"))]
            { Err(crate::error::Error::Other("Custom stack not enabled".into())) }
        }
        TransportType::Mio => {
            Ok(Box::new(crate::network::mio_fallback::MioTransport::new(config)?))
        }
    }
}

/// Transport configuration
#[derive(Debug, Clone)]
pub struct TransportConfig {
    pub bind_addr: SocketAddr,
    pub buffer_size: usize,
    pub queue_depth: usize,
    pub num_workers: usize,
    pub interface: Option<String>,
    pub extra: std::collections::HashMap<String, String>,
}

impl Default for TransportConfig {
    fn default() -> Self {
        Self {
            bind_addr: "0.0.0.0:0".parse().unwrap(),
            buffer_size: 65536,
            queue_depth: 4096,
            num_workers: num_cpus::get(),
            interface: None,
            extra: std::collections::HashMap::new(),
        }
    }
}
```

## 1. io_uring Implementation

### src/network/io_uring/mod.rs

```rust
use io_uring::{IoUring, opcode, types};
use std::net::SocketAddr;
use std::os::unix::io::AsRawFd;
use crate::network::transport::{Transport, TransportStats, TransportConfig};
use crate::error::Result;

pub struct IoUringTransport {
    ring: IoUring,
    socket: std::net::UdpSocket,
    stats: parking_lot::RwLock<TransportStats>,
}

impl IoUringTransport {
    pub async fn new(config: TransportConfig) -> Result<Self> {
        // Create io_uring instance
        let ring = IoUring::new(config.queue_depth as u32)?;
        
        // Create UDP socket
        let socket = std::net::UdpSocket::bind(config.bind_addr)?;
        socket.set_nonblocking(true)?;
        
        Ok(Self {
            ring,
            socket,
            stats: parking_lot::RwLock::new(TransportStats::default()),
        })
    }
}

#[async_trait::async_trait]
impl Transport for IoUringTransport {
    async fn send(&self, dest: SocketAddr, data: &[u8]) -> Result<usize> {
        let sockfd = types::Fd(self.socket.as_raw_fd());
        
        // Prepare send operation
        let send_e = opcode::Send::new(sockfd, data.as_ptr(), data.len() as _)
            .build()
            .user_data(0x01);
        
        // Submit to io_uring
        unsafe {
            let mut sq = self.ring.submission();
            sq.push(&send_e)?;
        }
        self.ring.submit_and_wait(1)?;
        
        // Wait for completion
        let cqe = self.ring.completion().next().expect("cqe is None");
        let n = cqe.result() as usize;
        
        // Update stats
        let mut stats = self.stats.write();
        stats.bytes_sent += n as u64;
        stats.packets_sent += 1;
        
        Ok(n)
    }
    
    async fn recv(&self, buf: &mut [u8]) -> Result<(usize, SocketAddr)> {
        let sockfd = types::Fd(self.socket.as_raw_fd());
        
        // Prepare recv operation
        let recv_e = opcode::Recv::new(sockfd, buf.as_mut_ptr(), buf.len() as _)
            .build()
            .user_data(0x02);
        
        // Submit to io_uring
        unsafe {
            let mut sq = self.ring.submission();
            sq.push(&recv_e)?;
        }
        self.ring.submit_and_wait(1)?;
        
        // Wait for completion
        let cqe = self.ring.completion().next().expect("cqe is None");
        let n = cqe.result() as usize;
        
        // Get peer address (simplified - would need recvfrom for real addr)
        let peer = "0.0.0.0:0".parse().unwrap();
        
        // Update stats
        let mut stats = self.stats.write();
        stats.bytes_received += n as u64;
        stats.packets_received += 1;
        
        Ok((n, peer))
    }
    
    fn name(&self) -> &str {
        "io_uring"
    }
    
    fn stats(&self) -> TransportStats {
        self.stats.read().clone()
    }
    
    async fn close(&self) -> Result<()> {
        Ok(())
    }
}
```

## 2. DPDK Implementation (C FFI)

### build.rs

```rust
use std::env;
use std::path::PathBuf;

fn main() {
    #[cfg(feature = "dpdk")]
    build_dpdk_bindings();
    
    #[cfg(feature = "rdma")]
    build_rdma_bindings();
}

#[cfg(feature = "dpdk")]
fn build_dpdk_bindings() {
    // Find DPDK installation
    let dpdk_dir = env::var("RTE_SDK").unwrap_or_else(|_| "/usr/share/dpdk".to_string());
    
    println!("cargo:rerun-if-env-changed=RTE_SDK");
    println!("cargo:rustc-link-search=native={}/lib", dpdk_dir);
    println!("cargo:rustc-link-lib=dpdk");
    
    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", dpdk_dir))
        .allowlist_function("rte_.*")
        .allowlist_type("rte_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate DPDK bindings");
    
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("dpdk_bindings.rs"))
        .expect("Couldn't write bindings!");
}

#[cfg(feature = "rdma")]
fn build_rdma_bindings() {
    println!("cargo:rustc-link-lib=ibverbs");
    println!("cargo:rustc-link-lib=rdmacm");
    
    let bindings = bindgen::Builder::default()
        .header("rdma_wrapper.h")
        .allowlist_function("ibv_.*")
        .allowlist_function("rdma_.*")
        .allowlist_type("ibv_.*")
        .allowlist_type("rdma_.*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate RDMA bindings");
    
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("rdma_bindings.rs"))
        .expect("Couldn't write bindings!");
}
```

### wrapper.h

```c
#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>
#include <rte_ring.h>
```

### src/ffi/dpdk.rs

```rust
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/dpdk_bindings.rs"));
```

### src/network/dpdk/mod.rs

```rust
use crate::ffi::dpdk::*;
use crate::network::transport::{Transport, TransportStats, TransportConfig};
use crate::error::Result;
use std::net::SocketAddr;
use std::ffi::CString;

pub struct DpdkTransport {
    port_id: u16,
    mempool: *mut rte_mempool,
    stats: parking_lot::RwLock<TransportStats>,
}

impl DpdkTransport {
    pub fn new(config: TransportConfig) -> Result<Self> {
        unsafe {
            // Initialize EAL
            let args = vec![
                CString::new("shared-nothing").unwrap(),
                CString::new("-l").unwrap(),
                CString::new("0-3").unwrap(),
                CString::new("-n").unwrap(),
                CString::new("4").unwrap(),
            ];
            
            let mut c_args: Vec<*mut i8> = args
                .iter()
                .map(|s| s.as_ptr() as *mut i8)
                .collect();
            
            let ret = rte_eal_init(c_args.len() as i32, c_args.as_mut_ptr());
            if ret < 0 {
                return Err(crate::error::Error::Other("EAL init failed".into()));
            }
            
            // Create mempool
            let mempool = rte_pktmbuf_pool_create(
                b"mbuf_pool\0".as_ptr() as *const i8,
                8192,
                250,
                0,
                2048,
                rte_socket_id() as i32,
            );
            
            if mempool.is_null() {
                return Err(crate::error::Error::Other("Mempool creation failed".into()));
            }
            
            // Configure port
            let port_id = 0;
            // ... port configuration ...
            
            Ok(Self {
                port_id,
                mempool,
                stats: parking_lot::RwLock::new(TransportStats::default()),
            })
        }
    }
    
    pub fn tx_burst(&self, packets: &[&[u8]]) -> Result<usize> {
        unsafe {
            // Allocate mbufs
            let mut mbufs: Vec<*mut rte_mbuf> = Vec::with_capacity(packets.len());
            
            for packet in packets {
                let mbuf = rte_pktmbuf_alloc(self.mempool);
                if mbuf.is_null() {
                    return Err(crate::error::Error::Other("Mbuf alloc failed".into()));
                }
                
                // Copy packet data
                let data_ptr = rte_pktmbuf_mtod(mbuf, *mut u8);
                std::ptr::copy_nonoverlapping(
                    packet.as_ptr(),
                    data_ptr,
                    packet.len(),
                );
                
                (*mbuf).data_len = packet.len() as u16;
                (*mbuf).pkt_len = packet.len() as u32;
                
                mbufs.push(mbuf);
            }
            
            // Transmit burst
            let sent = rte_eth_tx_burst(
                self.port_id,
                0,
                mbufs.as_mut_ptr(),
                mbufs.len() as u16,
            );
            
            // Update stats
            let mut stats = self.stats.write();
            stats.packets_sent += sent as u64;
            
            Ok(sent as usize)
        }
    }
}

#[async_trait::async_trait]
impl Transport for DpdkTransport {
    async fn send(&self, _dest: SocketAddr, data: &[u8]) -> Result<usize> {
        self.tx_burst(&[data])
    }
    
    async fn recv(&self, buf: &mut [u8]) -> Result<(usize, SocketAddr)> {
        // RX burst implementation
        // ...
        Ok((0, "0.0.0.0:0".parse().unwrap()))
    }
    
    fn name(&self) -> &str {
        "dpdk"
    }
    
    fn stats(&self) -> TransportStats {
        self.stats.read().clone()
    }
    
    async fn close(&self) -> Result<()> {
        Ok(())
    }
}

unsafe impl Send for DpdkTransport {}
unsafe impl Sync for DpdkTransport {}
```

## 3. AF_XDP Implementation

### src/network/af_xdp/mod.rs

```rust
use xsk_rs::{Socket, Config, Umem, FrameDesc};
use crate::network::transport::{Transport, TransportStats, TransportConfig};
use crate::error::Result;
use std::net::SocketAddr;

pub struct AfXdpTransport {
    socket: Socket<'static>,
    umem: Umem<'static>,
    interface: String,
    stats: parking_lot::RwLock<TransportStats>,
}

impl AfXdpTransport {
    pub async fn new(config: TransportConfig) -> Result<Self> {
        let interface = config.interface
            .clone()
            .ok_or_else(|| crate::error::Error::Other("Interface required for AF_XDP".into()))?;
        
        let xsk_config = Config {
            rx_queue_size: config.queue_depth as u32,
            tx_queue_size: config.queue_depth as u32,
            ..Default::default()
        };
        
        let (umem, socket) = Socket::new(xsk_config, &interface, 0)
            .map_err(|e| crate::error::Error::Other(e.to_string()))?;
        
        Ok(Self {
            socket,
            umem,
            interface,
            stats: parking_lot::RwLock::new(TransportStats::default()),
        })
    }
}

#[async_trait::async_trait]
impl Transport for AfXdpTransport {
    async fn send(&self, _dest: SocketAddr, data: &[u8]) -> Result<usize> {
        // Get frame from UMEM
        let mut frame = self.umem.frame()
            .map_err(|e| crate::error::Error::Other(e.to_string()))?;
        
        // Copy data to frame
        frame[..data.len()].copy_from_slice(data);
        
        // Submit to TX ring
        self.socket.tx_submit(frame)
            .map_err(|e| crate::error::Error::Other(e.to_string()))?;
        
        // Update stats
        let mut stats = self.stats.write();
        stats.bytes_sent += data.len() as u64;
        stats.packets_sent += 1;
        
        Ok(data.len())
    }
    
    async fn recv(&self, buf: &mut [u8]) -> Result<(usize, SocketAddr)> {
        // Poll RX ring
        let frame = self.socket.rx_recv()
            .map_err(|e| crate::error::Error::Other(e.to_string()))?;
        
        // Copy data from frame
        let len = frame.len().min(buf.len());
        buf[..len].copy_from_slice(&frame[..len]);
        
        // Return frame to UMEM
        self.umem.release(frame);
        
        // Update stats
        let mut stats = self.stats.write();
        stats.bytes_received += len as u64;
        stats.packets_received += 1;
        
        Ok((len, "0.0.0.0:0".parse().unwrap()))
    }
    
    fn name(&self) -> &str {
        "af_xdp"
    }
    
    fn stats(&self) -> TransportStats {
        self.stats.read().clone()
    }
    
    async fn close(&self) -> Result<()> {
        Ok(())
    }
}
```

## 4. eBPF/XDP with Aya

### ebpf/Cargo.toml

```toml
[workspace]
members = ["ebpf-program", "ebpf-loader"]

[package]
name = "ebpf-program"
version = "0.1.0"
edition = "2021"

[dependencies]
aya-bpf = "0.1"

[[bin]]
name = "xdp_filter"
path = "src/xdp_filter.rs"

[profile.release]
lto = true
opt-level = 3
```

### ebpf/src/xdp_filter.rs

```rust
#![no_std]
#![no_main]

use aya_bpf::{
    bindings::xdp_action,
    macros::{map, xdp},
    maps::HashMap,
    programs::XdpContext,
};

#[map(name = "PACKET_COUNT")]
static mut PACKET_COUNT: HashMap<u32, u64> = HashMap::with_max_entries(1024, 0);

#[xdp(name = "shared_nothing_xdp")]
pub fn xdp_filter(ctx: XdpContext) -> u32 {
    match process_packet(&ctx) {
        Ok(action) => action,
        Err(_) => xdp_action::XDP_PASS,
    }
}

fn process_packet(ctx: &XdpContext) -> Result<u32, ()> {
    // Parse ethernet header
    let eth_hdr = ctx.data()?;
    
    // Update packet counter
    unsafe {
        let key = 0u32;
        let count = PACKET_COUNT.get(&key).copied().unwrap_or(0);
        PACKET_COUNT.insert(&key, &(count + 1), 0)?;
    }
    
    // Accept packet (redirect to AF_XDP socket in production)
    Ok(xdp_action::XDP_PASS)
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    unsafe { core::hint::unreachable_unchecked() }
}
```

### src/network/af_xdp/ebpf_loader.rs

```rust
use aya::{Bpf, programs::Xdp};
use aya::maps::HashMap;
use crate::error::Result;

pub struct XdpLoader {
    bpf: Bpf,
}

impl XdpLoader {
    pub fn load(interface: &str) -> Result<Self> {
        // Load eBPF program
        let mut bpf = Bpf::load_file("ebpf/target/bpfel-unknown-none/release/xdp_filter")?;
        
        // Attach XDP program
        let program: &mut Xdp = bpf.program_mut("shared_nothing_xdp")?.try_into()?;
        program.load()?;
        program.attach(interface, aya::programs::XdpFlags::default())?;
        
        Ok(Self { bpf })
    }
    
    pub fn get_packet_count(&self) -> Result<u64> {
        let map: HashMap<_, u32, u64> = 
            HashMap::try_from(self.bpf.map("PACKET_COUNT")?)?;
        
        Ok(map.get(&0, 0)?)
    }
}
```

## 5. RDMA Implementation (rdma-core FFI)

### rdma_wrapper.h

```c
#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>
```

### src/ffi/rdma.rs

```rust
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

include!(concat!(env!("OUT_DIR"), "/rdma_bindings.rs"));
```

### src/network/rdma/mod.rs

```rust
use crate::ffi::rdma::*;
use crate::network::transport::{Transport, TransportStats, TransportConfig};
use crate::error::Result;
use std::net::SocketAddr;
use std::ptr;

pub struct RdmaTransport {
    context: *mut ibv_context,
    pd: *mut ibv_pd,
    qp: *mut ibv_qp,
    mr: *mut ibv_mr,
    buffer: Vec<u8>,
    stats: parking_lot::RwLock<TransportStats>,
}

impl RdmaTransport {
    pub fn new(config: TransportConfig) -> Result<Self> {
        unsafe {
            // Get device list
            let mut num_devices = 0i32;
            let dev_list = ibv_get_device_list(&mut num_devices);
            if dev_list.is_null() || num_devices == 0 {
                return Err(crate::error::Error::Other("No RDMA devices found".into()));
            }
            
            // Open first device
            let context = ibv_open_device(*dev_list);
            if context.is_null() {
                return Err(crate::error::Error::Other("Failed to open RDMA device".into()));
            }
            
            // Allocate protection domain
            let pd = ibv_alloc_pd(context);
            if pd.is_null() {
                return Err(crate::error::Error::Other("Failed to allocate PD".into()));
            }
            
            // Allocate buffer
            let mut buffer = vec![0u8; config.buffer_size];
            
            // Register memory region
            let mr = ibv_reg_mr(
                pd,
                buffer.as_mut_ptr() as *mut _,
                buffer.len(),
                (ibv_access_flags::IBV_ACCESS_LOCAL_WRITE |
                 ibv_access_flags::IBV_ACCESS_REMOTE_WRITE |
                 ibv_access_flags::IBV_ACCESS_REMOTE_READ).0 as i32,
            );
            
            if mr.is_null() {
                return Err(crate::error::Error::Other("Failed to register MR".into()));
            }
            
            // Create queue pair
            // ... QP creation ...
            let qp = ptr::null_mut(); // Simplified
            
            Ok(Self {
                context,
                pd,
                qp,
                mr,
                buffer,
                stats: parking_lot::RwLock::new(TransportStats::default()),
            })
        }
    }
    
    pub fn rdma_write(&self, data: &[u8], remote_addr: u64, rkey: u32) -> Result<()> {
        unsafe {
            self.buffer[..data.len()].copy_from_slice(data);
            
            let mut sge = ibv_sge {
                addr: self.buffer.as_ptr() as u64,
                length: data.len() as u32,
                lkey: (*self.mr).lkey,
            };
            
            let mut wr = std::mem::zeroed::<ibv_send_wr>();
            wr.wr_id = 1;
            wr.sg_list = &mut sge;
            wr.num_sge = 1;
            wr.opcode = ibv_wr_opcode::IBV_WR_RDMA_WRITE;
            wr.wr.rdma.remote_addr = remote_addr;
            wr.wr.rdma.rkey = rkey;
            
            let mut bad_wr = ptr::null_mut();
            ibv_post_send(self.qp, &mut wr, &mut bad_wr);
            
            // Update stats
            let mut stats = self.stats.write();
            stats.bytes_sent += data.len() as u64;
            
            Ok(())
        }
    }
}

#[async_trait::async_trait]
impl Transport for RdmaTransport {
    async fn send(&self, _dest: SocketAddr, data: &[u8]) -> Result<usize> {
        // RDMA operations need remote address/key
        // This is a simplified version
        Ok(data.len())
    }
    
    async fn recv(&self, buf: &mut [u8]) -> Result<(usize, SocketAddr)> {
        // RDMA recv implementation
        Ok((0, "0.0.0.0:0".parse().unwrap()))
    }
    
    fn name(&self) -> &str {
        "rdma"
    }
    
    fn stats(&self) -> TransportStats {
        self.stats.read().clone()
    }
    
    async fn close(&self) -> Result<()> {
        unsafe {
            if !self.qp.is_null() {
                ibv_destroy_qp(self.qp);
            }
            if !self.mr.is_null() {
                ibv_dereg_mr(self.mr);
            }
            if !self.pd.is_null() {
                ibv_dealloc_pd(self.pd);
            }
            if !self.context.is_null() {
                ibv_close_device(self.context);
            }
        }
        Ok(())
    }
}

unsafe impl Send for RdmaTransport {}
unsafe impl Sync for RdmaTransport {}
```

## 6. Raw Sockets with pnet

### src/network/raw/mod.rs

```rust
use pnet::packet::ip::IpNextHeaderProtocols;
use pnet::packet::udp::{MutableUdpPacket, UdpPacket};
use pnet::transport::{transport_channel, TransportChannelType};
use crate::network::transport::{Transport, TransportStats, TransportConfig};
use crate::error::Result;
use std::net::SocketAddr;

pub struct RawSocketTransport {
    tx: pnet::transport::TransportSender,
    rx: pnet::transport::TransportReceiver,
    stats: parking_lot::RwLock<TransportStats>,
}

impl RawSocketTransport {
    pub fn new(config: TransportConfig) -> Result<Self> {
        let protocol = TransportChannelType::Layer4(
            pnet::transport::TransportProtocol::Ipv4(IpNextHeaderProtocols::Udp)
        );
        
        let (tx, rx) = transport_channel(config.buffer_size, protocol)
            .map_err(|e| crate::error::Error::Other(e.to_string()))?;
        
        Ok(Self {
            tx,
            rx,
            stats: parking_lot::RwLock::new(TransportStats::default()),
        })
    }
}

#[async_trait::async_trait]
impl Transport for RawSocketTransport {
    async fn send(&self, dest: SocketAddr, data: &[u8]) -> Result<usize> {
        // Build UDP packet
        let mut packet_buf = vec![0u8; 8 + data.len()];
        let mut udp_packet = MutableUdpPacket::new(&mut packet_buf).unwrap();
        
        udp_packet.set_source(8080);
        udp_packet.set_destination(dest.port());
        udp_packet.set_length((8 + data.len()) as u16);
        udp_packet.set_payload(data);
        
        self.tx.send_to(udp_packet, dest.ip())
            .map_err(|e| crate::error::Error::Other(e.to_string()))?;
        
        // Update stats
        let mut stats = self.stats.write();
        stats.bytes_sent += data.len() as u64;
        stats.packets_sent += 1;
        
        Ok(data.len())
    }
    
    async fn recv(&self, buf: &mut [u8]) -> Result<(usize, SocketAddr)> {
        // Receive packet
        let mut iter = self.rx.iter();
        if let Some((packet, addr)) = iter.next() {
            let udp = UdpPacket::new(packet).unwrap();
            let payload = udp.payload();
            let len = payload.len().min(buf.len());
            buf[..len].copy_from_slice(&payload[..len]);
            
            // Update stats
            let mut stats = self.stats.write();
            stats.bytes_received += len as u64;
            stats.packets_received += 1;
            
            let socket_addr = SocketAddr::new(addr, udp.get_source());
            return Ok((len, socket_addr));
        }
        
        Err(crate::error::Error::Timeout)
    }
    
    fn name(&self) -> &str {
        "raw_socket"
    }
    
    fn stats(&self) -> TransportStats {
        self.stats.read().clone()
    }
    
    async fn close(&self) -> Result<()> {
        Ok(())
    }
}
```

## 7. Custom Stack with smoltcp

### src/network/custom/mod.rs

```rust
use smoltcp::wire::{IpAddress, IpCidr};
use smoltcp::iface::{Config, Interface, SocketSet};
use smoltcp::socket::udp;
use smoltcp::time::Instant;
use crate::network::transport::{Transport, TransportStats, TransportConfig};
use crate::error::Result;
use std::net::SocketAddr;

pub struct SmoltcpTransport {
    iface: Interface,
    sockets: SocketSet<'static>,
    stats: parking_lot::RwLock<TransportStats>,
}

impl SmoltcpTransport {
    pub fn new(config: TransportConfig) -> Result<Self> {
        // Create interface
        let mut iface_config = Config::new();
        iface_config.random_seed = rand::random();
        
        let mut iface = Interface::new(iface_config, &mut device);
        
        iface.update_ip_addrs(|ip_addrs| {
            ip_addrs.push(IpCidr::new(IpAddress::v4(192, 168, 1, 1), 24)).unwrap();
        });
        
        let mut sockets = SocketSet::new(vec![]);
        
        Ok(Self {
            iface,
            sockets,
            stats: parking_lot::RwLock::new(TransportStats::default()),
        })
    }
}

#[async_trait::async_trait]
impl Transport for SmoltcpTransport {
    async fn send(&self, dest: SocketAddr, data: &[u8]) -> Result<usize> {
        // smoltcp UDP send
        // ... implementation ...
        Ok(data.len())
    }
    
    async fn recv(&self, buf: &mut [u8]) -> Result<(usize, SocketAddr)> {
        // smoltcp UDP recv
        // ... implementation ...
        Ok((0, "0.0.0.0:0".parse().unwrap()))
    }
    
    fn name(&self) -> &str {
        "smoltcp"
    }
    
    fn stats(&self) -> TransportStats {
        self.stats.read().clone()
    }
    
    async fn close(&self) -> Result<()> {
        Ok(())
    }
}
```

## 8. mio Fallback

### src/network/mio_fallback/mod.rs

```rust
use mio::{Events, Interest, Poll, Token};
use mio::net::UdpSocket;
use crate::network::transport::{Transport, TransportStats, TransportConfig};
use crate::error::Result;
use std::net::SocketAddr;

pub struct MioTransport {
    socket: UdpSocket,
    poll: Poll,
    stats: parking_lot::RwLock<TransportStats>,
}

impl MioTransport {
    pub fn new(config: TransportConfig) -> Result<Self> {
        let mut socket = UdpSocket::bind(config.bind_addr)?;
        let poll = Poll::new()?;
        
        poll.registry().register(
            &mut socket,
            Token(0),
            Interest::READABLE | Interest::WRITABLE,
        )?;
        
        Ok(Self {
            socket,
            poll,
            stats: parking_lot::RwLock::new(TransportStats::default()),
        })
    }
}

#[async_trait::async_trait]
impl Transport for MioTransport {
    async fn send(&self, dest: SocketAddr, data: &[u8]) -> Result<usize> {
        let n = self.socket.send_to(data, dest)?;
        
        let mut stats = self.stats.write();
        stats.bytes_sent += n as u64;
        stats.packets_sent += 1;
        
        Ok(n)
    }
    
    async fn recv(&self, buf: &mut [u8]) -> Result<(usize, SocketAddr)> {
        let mut events = Events::with_capacity(1);
        self.poll.poll(&mut events, Some(std::time::Duration::from_millis(100)))?;
        
        if events.iter().next().is_some() {
            let (n, addr) = self.socket.recv_from(buf)?;
            
            let mut stats = self.stats.write();
            stats.bytes_received += n as u64;
            stats.packets_received += 1;
            
            return Ok((n, addr));
        }
        
        Err(crate::error::Error::Timeout)
    }
    
    fn name(&self) -> &str {
        "mio"
    }
    
    fn stats(&self) -> TransportStats {
        self.stats.read().clone()
    }
    
    async fn close(&self) -> Result<()> {
        Ok(())
    }
}
```

## Integration with Worker Pool

### src/network/worker.rs

```rust
use crate::prelude::*;
use crate::network::transport::{Transport, TransportConfig, TransportType, create_transport};
use std::sync::Arc;

/// Network-enabled worker
pub struct NetworkWorker {
    transport: Arc<Box<dyn Transport>>,
}

impl NetworkWorker {
    pub async fn new(transport_type: TransportType, config: TransportConfig) -> Result<Self> {
        let transport = create_transport(transport_type, config).await?;
        Ok(Self {
            transport: Arc::new(transport),
        })
    }
}

impl Worker for NetworkWorker {
    type State = Vec<u8>;
    type Message = (SocketAddr, Vec<u8>);
    
    fn init(&mut self) -> Result<Self::State> {
        Ok(Vec::new())
    }
    
    fn handle_message(&mut self, state: &mut Self::State, msg: Envelope<Self::Message>) -> Result<()> {
        let (dest, data) = msg.payload;
        
        // Send over network
        let transport = Arc::clone(&self.transport);
        tokio::spawn(async move {
            let _ = transport.send(dest, &data).await;
        });
        
        state.extend_from_slice(&data);
        Ok(())
    }
}
```

## Testing

### tests/network_test.rs

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_io_uring_transport() {
        let config = TransportConfig::default();
        let transport = create_transport(TransportType::IoUring, config).await.unwrap();
        
        let data = b"Hello, io_uring!";
        let dest = "127.0.0.1:8080".parse().unwrap();
        
        let n = transport.send(dest, data).await.unwrap();
        assert_eq!(n, data.len());
        
        let stats = transport.stats();
        assert_eq!(stats.packets_sent, 1);
    }
    
    // More tests for each transport...
}
```

## Performance Benchmarks

### benches/network_benchmark.rs

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use shared_nothing::network::*;

fn benchmark_transports(c: &mut Criterion) {
    let mut group = c.benchmark_group("network_transports");
    
    let transports = vec![
        TransportType::IoUring,
        TransportType::Mio,
    ];
    
    for transport_type in transports {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", transport_type)),
            &transport_type,
            |b, &transport_type| {
                b.iter(|| {
                    // Benchmark send/recv
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, benchmark_transports);
criterion_main!(benches);
```

## Next Steps

1. **Phase 1**: Implement io_uring and mio (cross-platform baseline)
2. **Phase 2**: Add AF_XDP with aya eBPF
3. **Phase 3**: Integrate C DPDK via FFI
4. **Phase 4**: Add RDMA support via rdma-core
5. **Phase 5**: Implement pnet raw sockets
6. **Phase 6**: Add smoltcp custom stack
7. **Phase 7**: (Advanced) Rust-for-Linux kernel modules

## Build Instructions

```bash
# Clone repository
git clone https://github.com/yourusername/shared-nothing.git
cd shared-nothing

# Install system dependencies
./scripts/install_deps.sh

# Build with all features
cargo build --all-features --release

# Build with specific features
cargo build --features="io-uring,af-xdp" --release

# Run tests
cargo test --all-features

# Run benchmarks
cargo bench --all-features

# Build eBPF programs
cd ebpf && cargo build --release
```

---

**Status**: Implementation plan complete
**Estimated Effort**: 4-6 weeks for full implementation
**Priority Order**: io_uring → mio → AF_XDP → DPDK → RDMA → others

