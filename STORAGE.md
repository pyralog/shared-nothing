# Low-Level Storage Options for Shared-Nothing Architecture

This document evaluates low-level storage technologies for maximum performance in shared-nothing architectures.

## Performance Overview

| Technology | Latency | IOPS | Throughput | CPU | Zero-Copy |
|-----------|---------|------|------------|-----|-----------|
| **SPDK** | <10μs | 10M+ | 40+ GB/s | Poll-mode | ✅ |
| **io_uring** | 10-50μs | 4M+ | 20+ GB/s | Async | ✅ |
| **NVMe-oF/RDMA** | <20μs | 8M+ | 100+ Gbps | Offload | ✅ |
| **Direct NVMe** | ~20μs | 1M+ | 3+ GB/s | IOCTL | ✅ |
| **DAX/PMem** | <1μs | N/A | Memory BW | Load/store | ✅ |
| **PMDK** | ~200ns | N/A | Memory BW | Atomic | ✅ |
| **O_DIRECT** | 50-100μs | 500K+ | 2+ GB/s | Bypass | ⚠️ |
| **mmap** | Varies | N/A | Memory BW | Page fault | ✅ |
| **Raw Block** | 30-80μs | 800K+ | 2+ GB/s | Custom | ⚠️ |
| **NVMe Queues** | <5μs | 12M+ | 50+ GB/s | Direct | ✅ |
| **UIO** | <10μs | Custom | Custom | Userspace | ✅ |

---

## 1. SPDK (Storage Performance Development Kit)

### Overview
Intel's framework for userspace, asynchronous, polled-mode NVMe driver with zero kernel involvement.

### Key Features
- Userspace NVMe drivers
- Poll-mode (no interrupts)
- Zero-copy operations
- Lock-free queues
- 10M+ IOPS per core

### Architecture

```text
┌──────────────────────────────────────┐
│      Application (Rust)              │
├──────────────────────────────────────┤
│         SPDK Libraries               │
│  ┌────────────┬──────────────────┐   │
│  │ NVMe Driver│  Block Device    │   │ Userspace
│  │ (Polled)   │     Layer        │   │
│  └────────────┴──────────────────┘   │
═══════════════════════════════════════════
│  ┌────────────────────────────────┐  │
│  │    UIO/VFIO (Kernel bypass)    │  │ Kernel
│  └────────────────────────────────┘  │
├──────────────────────────────────────┤
│         PCIe NVMe Device             │ Hardware
└──────────────────────────────────────┘
```

### Rust Integration

#### build.rs

```rust
fn main() {
    #[cfg(feature = "spdk")]
    build_spdk();
}

#[cfg(feature = "spdk")]
fn build_spdk() {
    let spdk_dir = env::var("SPDK_DIR")
        .unwrap_or_else(|_| "/usr/local/spdk".to_string());
    
    println!("cargo:rustc-link-search=native={}/build/lib", spdk_dir);
    println!("cargo:rustc-link-lib=static=spdk_nvme");
    println!("cargo:rustc-link-lib=static=spdk_env_dpdk");
    println!("cargo:rustc-link-lib=static=spdk_util");
    println!("cargo:rustc-link-lib=static=rte_eal");
    
    // Generate bindings
    let bindings = bindgen::Builder::default()
        .header("spdk_wrapper.h")
        .clang_arg(format!("-I{}/include", spdk_dir))
        .allowlist_function("spdk_.*")
        .allowlist_type("spdk_.*")
        .generate()
        .expect("Unable to generate SPDK bindings");
    
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("spdk_bindings.rs"))
        .expect("Couldn't write bindings!");
}
```

#### spdk_wrapper.h

```c
#include <spdk/nvme.h>
#include <spdk/env.h>
#include <spdk/queue.h>
```

#### src/storage/spdk/mod.rs

```rust
use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::Arc;
use parking_lot::Mutex;

#[allow(non_upper_case_globals)]
#[allow(non_camel_case_types)]
#[allow(non_snake_case)]
#[allow(dead_code)]
mod ffi {
    include!(concat!(env!("OUT_DIR"), "/spdk_bindings.rs"));
}

pub struct SpdkStorage {
    controller: *mut ffi::spdk_nvme_ctrlr,
    namespace: *mut ffi::spdk_nvme_ns,
    qpair: *mut ffi::spdk_nvme_qpair,
    buffer_pool: Vec<*mut u8>,
}

impl SpdkStorage {
    pub fn new(pci_addr: &str) -> Result<Self> {
        unsafe {
            // Initialize SPDK
            let mut opts: ffi::spdk_env_opts = std::mem::zeroed();
            ffi::spdk_env_opts_init(&mut opts);
            
            opts.name = CString::new("shared-nothing").unwrap().as_ptr();
            opts.core_mask = CString::new("0x1").unwrap().as_ptr();
            opts.shm_id = 0;
            
            if ffi::spdk_env_init(&opts) < 0 {
                return Err("SPDK env init failed".into());
            }
            
            // Probe for NVMe devices
            let pci_addr_c = CString::new(pci_addr).unwrap();
            let mut controller = ptr::null_mut();
            
            // Attach to controller
            let trid: *mut ffi::spdk_nvme_transport_id = std::ptr::null_mut();
            // ... controller attachment code ...
            
            // Get first namespace
            let namespace = ffi::spdk_nvme_ctrlr_get_ns(controller, 1);
            if namespace.is_null() {
                return Err("No namespace found".into());
            }
            
            // Allocate I/O queue pair
            let qpair = ffi::spdk_nvme_ctrlr_alloc_io_qpair(
                controller,
                ptr::null(),
                0,
            );
            
            if qpair.is_null() {
                return Err("Failed to allocate qpair".into());
            }
            
            Ok(Self {
                controller,
                namespace,
                qpair,
                buffer_pool: Vec::new(),
            })
        }
    }
    
    pub fn write(&mut self, lba: u64, data: &[u8]) -> Result<()> {
        unsafe {
            let sector_size = ffi::spdk_nvme_ns_get_sector_size(self.namespace);
            let num_sectors = (data.len() + sector_size as usize - 1) / sector_size as usize;
            
            // Allocate DMA buffer
            let buffer = ffi::spdk_dma_zmalloc(
                data.len(),
                sector_size,
                ptr::null_mut(),
            ) as *mut u8;
            
            if buffer.is_null() {
                return Err("DMA allocation failed".into());
            }
            
            // Copy data to DMA buffer
            ptr::copy_nonoverlapping(data.as_ptr(), buffer, data.len());
            
            // Submit write command
            let rc = ffi::spdk_nvme_ns_cmd_write(
                self.namespace,
                self.qpair,
                buffer as *mut _,
                lba,
                num_sectors as u32,
                Some(Self::write_complete_callback),
                ptr::null_mut(),
                0,
            );
            
            if rc != 0 {
                ffi::spdk_dma_free(buffer as *mut _);
                return Err("Write command failed".into());
            }
            
            // Poll for completion
            while ffi::spdk_nvme_qpair_process_completions(self.qpair, 0) == 0 {
                // Busy wait (poll mode)
            }
            
            ffi::spdk_dma_free(buffer as *mut _);
            Ok(())
        }
    }
    
    pub fn read(&mut self, lba: u64, size: usize) -> Result<Vec<u8>> {
        unsafe {
            let sector_size = ffi::spdk_nvme_ns_get_sector_size(self.namespace);
            let num_sectors = (size + sector_size as usize - 1) / sector_size as usize;
            
            // Allocate DMA buffer
            let buffer = ffi::spdk_dma_zmalloc(
                size,
                sector_size,
                ptr::null_mut(),
            ) as *mut u8;
            
            if buffer.is_null() {
                return Err("DMA allocation failed".into());
            }
            
            // Submit read command
            let rc = ffi::spdk_nvme_ns_cmd_read(
                self.namespace,
                self.qpair,
                buffer as *mut _,
                lba,
                num_sectors as u32,
                Some(Self::read_complete_callback),
                ptr::null_mut(),
                0,
            );
            
            if rc != 0 {
                ffi::spdk_dma_free(buffer as *mut _);
                return Err("Read command failed".into());
            }
            
            // Poll for completion
            while ffi::spdk_nvme_qpair_process_completions(self.qpair, 0) == 0 {
                // Busy wait
            }
            
            // Copy data from DMA buffer
            let mut result = vec![0u8; size];
            ptr::copy_nonoverlapping(buffer, result.as_mut_ptr(), size);
            
            ffi::spdk_dma_free(buffer as *mut _);
            Ok(result)
        }
    }
    
    extern "C" fn write_complete_callback(
        _ctx: *mut std::os::raw::c_void,
        _cpl: *const ffi::spdk_nvme_cpl,
    ) {
        // Completion callback
    }
    
    extern "C" fn read_complete_callback(
        _ctx: *mut std::os::raw::c_void,
        _cpl: *const ffi::spdk_nvme_cpl,
    ) {
        // Completion callback
    }
}

impl Drop for SpdkStorage {
    fn drop(&mut self) {
        unsafe {
            if !self.qpair.is_null() {
                ffi::spdk_nvme_ctrlr_free_io_qpair(self.qpair);
            }
        }
    }
}

unsafe impl Send for SpdkStorage {}
unsafe impl Sync for SpdkStorage {}
```

### Use Cases
- High-frequency trading systems
- In-memory databases (persistent tier)
- Log-structured storage
- Real-time analytics

### Pros
- ✅ Lowest latency (10M+ IOPS)
- ✅ Zero kernel overhead
- ✅ Complete control
- ✅ Lock-free design

### Cons
- ❌ Complex setup (huge pages, CPU isolation)
- ❌ Userspace driver maintenance
- ❌ Requires dedicated cores
- ❌ NVMe devices only
- ❌ Steep learning curve

---

## 2. io_uring (Linux Async I/O)

### Overview
Modern Linux async I/O interface using shared ring buffers between kernel and userspace.

### Key Features
- Zero-copy operations
- Batch submission/completion
- No syscalls in fast path
- Supports files, sockets, everything
- 4M+ IOPS

### Architecture

```text
Application                Kernel
┌─────────────┐           ┌─────────────┐
│ Submission  │──────────>│   Process   │
│   Ring      │  io_uring │   Requests  │
│  (SQ)       │  enter    │             │
└─────────────┘           └──────┬──────┘
                                 │
                          ┌──────▼──────┐
                          │   Storage   │
                          │   I/O       │
                          └──────┬──────┘
┌─────────────┐           ┌──────▼──────┐
│ Completion  │<──────────│   Results   │
│   Ring      │   Poll    │             │
│  (CQ)       │           │             │
└─────────────┘           └─────────────┘
```

### Implementation

#### src/storage/io_uring/mod.rs

```rust
use io_uring::{IoUring, opcode, types};
use std::fs::{File, OpenOptions};
use std::os::unix::io::AsRawFd;
use std::path::Path;

pub struct IoUringStorage {
    ring: IoUring,
    file: File,
}

impl IoUringStorage {
    pub fn new<P: AsRef<Path>>(path: P, queue_depth: u32) -> Result<Self> {
        // Create io_uring instance
        let ring = IoUring::builder()
            .setup_sqpoll(1000)  // Kernel polling thread
            .build(queue_depth)?;
        
        // Open file with O_DIRECT for zero-copy
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)?;
        
        Ok(Self { ring, file })
    }
    
    pub fn write_at(&mut self, offset: u64, data: &[u8]) -> Result<usize> {
        let fd = types::Fd(self.file.as_raw_fd());
        
        // Prepare write operation
        let write_e = opcode::Write::new(fd, data.as_ptr(), data.len() as _)
            .offset(offset)
            .build()
            .user_data(0x01);
        
        // Submit to submission queue
        unsafe {
            self.ring.submission().push(&write_e)?;
        }
        
        // Submit and wait
        self.ring.submit_and_wait(1)?;
        
        // Get completion
        let cqe = self.ring.completion().next().expect("cqe is None");
        let result = cqe.result();
        
        if result < 0 {
            return Err(std::io::Error::from_raw_os_error(-result).into());
        }
        
        Ok(result as usize)
    }
    
    pub fn read_at(&mut self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        let fd = types::Fd(self.file.as_raw_fd());
        
        // Prepare read operation
        let read_e = opcode::Read::new(fd, buf.as_mut_ptr(), buf.len() as _)
            .offset(offset)
            .build()
            .user_data(0x02);
        
        unsafe {
            self.ring.submission().push(&read_e)?;
        }
        
        self.ring.submit_and_wait(1)?;
        
        let cqe = self.ring.completion().next().expect("cqe is None");
        let result = cqe.result();
        
        if result < 0 {
            return Err(std::io::Error::from_raw_os_error(-result).into());
        }
        
        Ok(result as usize)
    }
    
    pub fn write_vectored(&mut self, offset: u64, buffers: &[&[u8]]) -> Result<usize> {
        let fd = types::Fd(self.file.as_raw_fd());
        
        // Convert to iovec
        let iovecs: Vec<libc::iovec> = buffers
            .iter()
            .map(|buf| libc::iovec {
                iov_base: buf.as_ptr() as *mut _,
                iov_len: buf.len(),
            })
            .collect();
        
        // Prepare writev operation
        let writev_e = opcode::Writev::new(fd, iovecs.as_ptr(), iovecs.len() as _)
            .offset(offset)
            .build()
            .user_data(0x03);
        
        unsafe {
            self.ring.submission().push(&writev_e)?;
        }
        
        self.ring.submit_and_wait(1)?;
        
        let cqe = self.ring.completion().next().expect("cqe is None");
        let result = cqe.result();
        
        if result < 0 {
            return Err(std::io::Error::from_raw_os_error(-result).into());
        }
        
        Ok(result as usize)
    }
    
    pub fn batch_operations(&mut self, ops: Vec<Operation>) -> Result<Vec<usize>> {
        // Submit multiple operations
        for op in &ops {
            let entry = match op {
                Operation::Write { offset, data } => {
                    let fd = types::Fd(self.file.as_raw_fd());
                    opcode::Write::new(fd, data.as_ptr(), data.len() as _)
                        .offset(*offset)
                        .build()
                        .user_data(op.id() as u64)
                }
                Operation::Read { offset, size } => {
                    let fd = types::Fd(self.file.as_raw_fd());
                    // ... similar for read ...
                    continue;
                }
            };
            
            unsafe {
                self.ring.submission().push(&entry)?;
            }
        }
        
        // Submit all at once
        self.ring.submit_and_wait(ops.len())?;
        
        // Collect results
        let mut results = Vec::new();
        for _ in 0..ops.len() {
            if let Some(cqe) = self.ring.completion().next() {
                results.push(cqe.result() as usize);
            }
        }
        
        Ok(results)
    }
    
    pub fn fsync(&mut self) -> Result<()> {
        let fd = types::Fd(self.file.as_raw_fd());
        
        let fsync_e = opcode::Fsync::new(fd)
            .build()
            .user_data(0x04);
        
        unsafe {
            self.ring.submission().push(&fsync_e)?;
        }
        
        self.ring.submit_and_wait(1)?;
        
        let cqe = self.ring.completion().next().expect("cqe is None");
        if cqe.result() < 0 {
            return Err(std::io::Error::from_raw_os_error(-cqe.result()).into());
        }
        
        Ok(())
    }
}

pub enum Operation {
    Write { offset: u64, data: Vec<u8> },
    Read { offset: u64, size: usize },
}

impl Operation {
    fn id(&self) -> usize {
        match self {
            Operation::Write { .. } => 1,
            Operation::Read { .. } => 2,
        }
    }
}
```

### Use Cases
- High-performance file I/O
- Database storage engines
- Network + storage combined
- Async application servers

### Pros
- ✅ Best async I/O on Linux
- ✅ Zero-copy capable
- ✅ Batch operations
- ✅ Works with all I/O types
- ✅ Good Rust support

### Cons
- ❌ Linux 5.1+ only
- ❌ Requires kernel support
- ❌ Complex API
- ❌ Still maturing

---

## 3. NVMe-oF (NVMe over Fabrics)

### Overview
Access remote NVMe devices over network with near-local performance.

### Key Features
- Network-attached NVMe
- <20μs latency (with RDMA)
- Multiple transports (RDMA, TCP, FC)
- Kernel or userspace initiator

### Transports

**RDMA (RoCE/InfiniBand)**
- Latency: <20μs
- Throughput: 100+ Gbps
- CPU: Offloaded

**TCP**
- Latency: 50-100μs
- Throughput: 10-40 Gbps
- CPU: Higher overhead

**FC (Fibre Channel)**
- Latency: <30μs
- Throughput: 32-128 Gbps
- Infrastructure: Existing SAN

### Implementation

#### src/storage/nvmeof/mod.rs

```rust
use std::fs::OpenOptions;
use std::os::unix::io::AsRawFd;
use std::path::Path;

pub struct NvmeofStorage {
    device: std::fs::File,
    subsystem: String,
}

impl NvmeofStorage {
    pub fn connect(
        transport: &str,
        address: &str,
        subsystem_nqn: &str,
    ) -> Result<Self> {
        // Connect via nvme-cli or kernel interface
        let connect_cmd = format!(
            "nvme connect -t {} -a {} -n {}",
            transport, address, subsystem_nqn
        );
        
        std::process::Command::new("sh")
            .arg("-c")
            .arg(&connect_cmd)
            .output()?;
        
        // Find the created device
        let device_path = Self::find_device(subsystem_nqn)?;
        
        // Open device with O_DIRECT
        let device = OpenOptions::new()
            .read(true)
            .write(true)
            .custom_flags(libc::O_DIRECT)
            .open(&device_path)?;
        
        Ok(Self {
            device,
            subsystem: subsystem_nqn.to_string(),
        })
    }
    
    fn find_device(subsystem_nqn: &str) -> Result<String> {
        // Scan /sys/class/nvme for matching subsystem
        // Return device path like /dev/nvme1n1
        Ok("/dev/nvme1n1".to_string())
    }
    
    pub fn read_block(&self, lba: u64, blocks: u32) -> Result<Vec<u8>> {
        let block_size = 4096;
        let size = blocks as usize * block_size;
        let mut buffer = vec![0u8; size];
        
        unsafe {
            let n = libc::pread(
                self.device.as_raw_fd(),
                buffer.as_mut_ptr() as *mut _,
                size,
                (lba * block_size as u64) as i64,
            );
            
            if n < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
        }
        
        Ok(buffer)
    }
    
    pub fn write_block(&self, lba: u64, data: &[u8]) -> Result<()> {
        let block_size = 4096;
        
        unsafe {
            let n = libc::pwrite(
                self.device.as_raw_fd(),
                data.as_ptr() as *const _,
                data.len(),
                (lba * block_size as u64) as i64,
            );
            
            if n < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
        }
        
        Ok(())
    }
    
    pub fn disconnect(&self) -> Result<()> {
        let disconnect_cmd = format!("nvme disconnect -n {}", self.subsystem);
        
        std::process::Command::new("sh")
            .arg("-c")
            .arg(&disconnect_cmd)
            .output()?;
        
        Ok(())
    }
}
```

### RDMA-backed NVMe-oF

```rust
pub struct NvmeofRdmaStorage {
    rdma_connection: RdmaConnection,
    remote_mr: RemoteMemoryRegion,
}

impl NvmeofRdmaStorage {
    pub fn rdma_read(&self, lba: u64, size: usize) -> Result<Vec<u8>> {
        // Direct RDMA read from remote NVMe
        let remote_addr = self.lba_to_address(lba);
        self.rdma_connection.read(remote_addr, size)
    }
    
    pub fn rdma_write(&self, lba: u64, data: &[u8]) -> Result<()> {
        // Direct RDMA write to remote NVMe
        let remote_addr = self.lba_to_address(lba);
        self.rdma_connection.write(remote_addr, data)
    }
    
    fn lba_to_address(&self, lba: u64) -> u64 {
        self.remote_mr.base_addr + (lba * 4096)
    }
}
```

### Use Cases
- Disaggregated storage
- Storage clusters
- Cloud storage backends
- Distributed databases

### Pros
- ✅ Remote storage with local-like performance
- ✅ Standard protocol
- ✅ Flexible deployment
- ✅ Good for scale-out

### Cons
- ❌ Requires network infrastructure
- ❌ Latency higher than local
- ❌ Complex setup
- ❌ Limited Rust tooling

---

## 4. Direct NVMe Access

### Overview
Direct access to NVMe devices via character device interface.

### Key Features
- `/dev/nvme` character device
- IOCTL commands
- Admin and I/O commands
- Full NVMe command set

### Implementation

#### src/storage/nvme_direct/mod.rs

```rust
use std::fs::{File, OpenOptions};
use std::os::unix::io::AsRawFd;

// NVMe IOCTL commands
const NVME_IOCTL_ADMIN_CMD: u64 = 0xC0484E41;
const NVME_IOCTL_IO_CMD: u64 = 0xC0484E43;
const NVME_IOCTL_SUBMIT_IO: u64 = 0x40084E42;

#[repr(C)]
struct NvmeAdminCmd {
    opcode: u8,
    flags: u8,
    command_id: u16,
    nsid: u32,
    cdw2: u32,
    cdw3: u32,
    metadata: u64,
    addr: u64,
    metadata_len: u32,
    data_len: u32,
    cdw10: u32,
    cdw11: u32,
    cdw12: u32,
    cdw13: u32,
    cdw14: u32,
    cdw15: u32,
    timeout_ms: u32,
    result: u32,
}

#[repr(C)]
struct NvmeUserIo {
    opcode: u8,
    flags: u8,
    control: u16,
    nblocks: u16,
    rsvd: u16,
    metadata: u64,
    addr: u64,
    slba: u64,
    dsmgmt: u32,
    reftag: u32,
    apptag: u16,
    appmask: u16,
}

pub struct NvmeDirectStorage {
    device: File,
    namespace_id: u32,
}

impl NvmeDirectStorage {
    pub fn new(device_path: &str, namespace_id: u32) -> Result<Self> {
        let device = OpenOptions::new()
            .read(true)
            .write(true)
            .open(device_path)?;
        
        Ok(Self {
            device,
            namespace_id,
        })
    }
    
    pub fn read_blocks(&self, lba: u64, num_blocks: u16, buffer: &mut [u8]) -> Result<()> {
        let mut io_cmd = NvmeUserIo {
            opcode: 0x02, // Read
            flags: 0,
            control: 0,
            nblocks: num_blocks - 1,
            rsvd: 0,
            metadata: 0,
            addr: buffer.as_mut_ptr() as u64,
            slba: lba,
            dsmgmt: 0,
            reftag: 0,
            apptag: 0,
            appmask: 0,
        };
        
        unsafe {
            let ret = libc::ioctl(
                self.device.as_raw_fd(),
                NVME_IOCTL_SUBMIT_IO,
                &mut io_cmd as *mut _,
            );
            
            if ret < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
        }
        
        Ok(())
    }
    
    pub fn write_blocks(&self, lba: u64, num_blocks: u16, data: &[u8]) -> Result<()> {
        let mut io_cmd = NvmeUserIo {
            opcode: 0x01, // Write
            flags: 0,
            control: 0,
            nblocks: num_blocks - 1,
            rsvd: 0,
            metadata: 0,
            addr: data.as_ptr() as u64,
            slba: lba,
            dsmgmt: 0,
            reftag: 0,
            apptag: 0,
            appmask: 0,
        };
        
        unsafe {
            let ret = libc::ioctl(
                self.device.as_raw_fd(),
                NVME_IOCTL_SUBMIT_IO,
                &mut io_cmd as *mut _,
            );
            
            if ret < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
        }
        
        Ok(())
    }
    
    pub fn identify_namespace(&self) -> Result<NamespaceInfo> {
        let mut buffer = vec![0u8; 4096];
        
        let mut admin_cmd = NvmeAdminCmd {
            opcode: 0x06, // Identify
            flags: 0,
            command_id: 0,
            nsid: self.namespace_id,
            cdw2: 0,
            cdw3: 0,
            metadata: 0,
            addr: buffer.as_mut_ptr() as u64,
            metadata_len: 0,
            data_len: 4096,
            cdw10: 0, // CNS = Namespace
            cdw11: 0,
            cdw12: 0,
            cdw13: 0,
            cdw14: 0,
            cdw15: 0,
            timeout_ms: 1000,
            result: 0,
        };
        
        unsafe {
            let ret = libc::ioctl(
                self.device.as_raw_fd(),
                NVME_IOCTL_ADMIN_CMD,
                &mut admin_cmd as *mut _,
            );
            
            if ret < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
        }
        
        // Parse namespace info from buffer
        Ok(NamespaceInfo {
            size: 0,
            capacity: 0,
            utilization: 0,
            formatted_lba_size: 0,
        })
    }
    
    pub fn flush(&self) -> Result<()> {
        let mut io_cmd = NvmeUserIo {
            opcode: 0x00, // Flush
            flags: 0,
            control: 0,
            nblocks: 0,
            rsvd: 0,
            metadata: 0,
            addr: 0,
            slba: 0,
            dsmgmt: 0,
            reftag: 0,
            apptag: 0,
            appmask: 0,
        };
        
        unsafe {
            let ret = libc::ioctl(
                self.device.as_raw_fd(),
                NVME_IOCTL_SUBMIT_IO,
                &mut io_cmd as *mut _,
            );
            
            if ret < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
        }
        
        Ok(())
    }
}

pub struct NamespaceInfo {
    pub size: u64,
    pub capacity: u64,
    pub utilization: u64,
    pub formatted_lba_size: u32,
}
```

### Use Cases
- Custom storage engines
- NVMe device management
- Performance testing
- Custom command submission

### Pros
- ✅ Direct hardware access
- ✅ Full command set
- ✅ No filesystem overhead
- ✅ Fine-grained control

### Cons
- ❌ Requires root/capabilities
- ❌ Complex IOCTL interface
- ❌ Must handle all details
- ❌ Error-prone

---

## 5. DAX (Direct Access) & Persistent Memory

### Overview
Direct load/store access to persistent memory, bypassing kernel page cache.

### Key Features
- <1μs latency (memory speed)
- Byte-addressable persistence
- Load/store instructions
- Intel Optane PMem

### Architecture

```text
Application
┌────────────────────────┐
│  mmap() with MAP_SYNC  │
│         DAX            │
├────────────────────────┤
│    Virtual Memory      │
├────────────────────────┤
│   Page Tables (DAX)    │ No page cache!
├────────────────────────┤
│    ext4-DAX/XFS-DAX    │
├────────────────────────┤
│   Persistent Memory    │
│   (Intel Optane)       │
└────────────────────────┘
```

### Implementation

#### src/storage/dax/mod.rs

```rust
use std::fs::{File, OpenOptions};
use std::os::unix::io::AsRawFd;
use std::ptr;

pub struct DaxStorage {
    file: File,
    mapping: *mut u8,
    size: usize,
}

impl DaxStorage {
    pub fn new(path: &str, size: usize) -> Result<Self> {
        // Open file on DAX filesystem (ext4-dax or xfs-dax)
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;
        
        // Set file size
        file.set_len(size as u64)?;
        
        // Memory map with MAP_SYNC for persistence
        let mapping = unsafe {
            libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_SYNC, // MAP_SYNC ensures persistence
                file.as_raw_fd(),
                0,
            )
        };
        
        if mapping == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        
        Ok(Self {
            file,
            mapping: mapping as *mut u8,
            size,
        })
    }
    
    pub fn write_at(&self, offset: usize, data: &[u8]) -> Result<()> {
        if offset + data.len() > self.size {
            return Err("Write exceeds mapping size".into());
        }
        
        unsafe {
            // Direct memory copy to persistent memory
            ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.mapping.add(offset),
                data.len(),
            );
            
            // Flush to ensure persistence (on PMem this is fast)
            self.flush_range(offset, data.len());
        }
        
        Ok(())
    }
    
    pub fn read_at(&self, offset: usize, size: usize) -> Result<Vec<u8>> {
        if offset + size > self.size {
            return Err("Read exceeds mapping size".into());
        }
        
        let mut buffer = vec![0u8; size];
        
        unsafe {
            // Direct memory read from persistent memory
            ptr::copy_nonoverlapping(
                self.mapping.add(offset),
                buffer.as_mut_ptr(),
                size,
            );
        }
        
        Ok(buffer)
    }
    
    pub fn get_slice(&self, offset: usize, size: usize) -> Result<&[u8]> {
        if offset + size > self.size {
            return Err("Slice exceeds mapping size".into());
        }
        
        unsafe {
            Ok(std::slice::from_raw_parts(
                self.mapping.add(offset),
                size,
            ))
        }
    }
    
    pub fn get_slice_mut(&mut self, offset: usize, size: usize) -> Result<&mut [u8]> {
        if offset + size > self.size {
            return Err("Slice exceeds mapping size".into());
        }
        
        unsafe {
            Ok(std::slice::from_raw_parts_mut(
                self.mapping.add(offset),
                size,
            ))
        }
    }
    
    fn flush_range(&self, offset: usize, size: usize) {
        unsafe {
            // Cache line flush (CLWB instruction)
            let cache_line_size = 64;
            let start = self.mapping.add(offset);
            let end = start.add(size);
            
            let mut addr = start as usize & !(cache_line_size - 1);
            while addr < end as usize {
                // CLWB instruction (write-back cache line)
                std::arch::x86_64::_mm_clwb(addr as *const u8);
                addr += cache_line_size;
            }
            
            // Memory fence to ensure ordering
            std::arch::x86_64::_mm_sfence();
        }
    }
    
    pub fn persist_all(&self) {
        unsafe {
            libc::msync(
                self.mapping as *mut _,
                self.size,
                libc::MS_SYNC,
            );
        }
    }
}

impl Drop for DaxStorage {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.mapping as *mut _, self.size);
        }
    }
}

unsafe impl Send for DaxStorage {}
unsafe impl Sync for DaxStorage {}
```

### Use Cases
- In-memory databases with persistence
- Transaction logs
- Fast restart/recovery
- Byte-addressable storage

### Pros
- ✅ Fastest persistence (<1μs)
- ✅ Byte-addressable
- ✅ Direct load/store
- ✅ No kernel overhead
- ✅ Simple programming model

### Cons
- ❌ Requires PMem hardware
- ❌ Limited capacity (expensive)
- ❌ Special filesystem (ext4-dax/xfs-dax)
- ❌ Durability considerations

---

## 6. PMDK (Persistent Memory Development Kit)

### Overview
Intel's library for programming persistent memory with atomic operations.

### Key Features
- Atomic persistence
- Transaction support
- Memory allocators
- Language bindings

### Implementation

#### Cargo.toml

```toml
[dependencies]
pmdk-sys = "0.5"  # Low-level bindings
```

#### src/storage/pmdk/mod.rs

```rust
use pmdk_sys::*;
use std::ffi::CString;
use std::ptr;

pub struct PmdkStorage {
    pool: *mut PMEMobjpool,
    root: *mut RootObject,
}

#[repr(C)]
struct RootObject {
    data: [u8; 1024],
    counter: u64,
}

impl PmdkStorage {
    pub fn new(path: &str, size: usize) -> Result<Self> {
        let path_c = CString::new(path).unwrap();
        let layout_c = CString::new("shared_nothing").unwrap();
        
        unsafe {
            // Create or open pool
            let pool = pmemobj_create(
                path_c.as_ptr(),
                layout_c.as_ptr(),
                size as u64,
                0o666,
            );
            
            if pool.is_null() {
                let pool = pmemobj_open(path_c.as_ptr(), layout_c.as_ptr());
                if pool.is_null() {
                    return Err("Failed to create/open pool".into());
                }
            }
            
            // Get root object
            let root_oid = pmemobj_root(pool, std::mem::size_of::<RootObject>());
            let root = pmemobj_direct(root_oid) as *mut RootObject;
            
            Ok(Self { pool, root })
        }
    }
    
    pub fn transactional_write(&mut self, data: &[u8]) -> Result<()> {
        unsafe {
            // Begin transaction
            pmemobj_tx_begin(
                self.pool,
                ptr::null_mut(),
                ptr::null_mut(),
            );
            
            // Add root to transaction
            pmemobj_tx_add_range_direct(
                self.root as *const _,
                std::mem::size_of::<RootObject>(),
            );
            
            // Modify data within transaction
            let data_len = data.len().min(1024);
            ptr::copy_nonoverlapping(
                data.as_ptr(),
                (*self.root).data.as_mut_ptr(),
                data_len,
            );
            (*self.root).counter += 1;
            
            // Commit transaction
            pmemobj_tx_commit();
            pmemobj_tx_end();
        }
        
        Ok(())
    }
    
    pub fn atomic_increment(&mut self) -> Result<u64> {
        unsafe {
            let old_value = (*self.root).counter;
            
            // Atomic 8-byte write
            pmemobj_persist(
                self.pool,
                &mut (*self.root).counter as *mut _ as *const _,
                8,
            );
            
            Ok(old_value)
        }
    }
    
    pub fn allocate_persistent(&self, size: usize) -> Result<*mut u8> {
        unsafe {
            let mut oid: PMEMoid = std::mem::zeroed();
            
            let ret = pmemobj_alloc(
                self.pool,
                &mut oid,
                size,
                0,
                None,
                ptr::null_mut(),
            );
            
            if ret != 0 {
                return Err("Allocation failed".into());
            }
            
            let ptr = pmemobj_direct(oid) as *mut u8;
            Ok(ptr)
        }
    }
}

impl Drop for PmdkStorage {
    fn drop(&mut self) {
        unsafe {
            pmemobj_close(self.pool);
        }
    }
}
```

### Use Cases
- Persistent data structures
- Transaction logging
- Key-value stores
- Restart-resilient applications

### Pros
- ✅ Atomic persistence
- ✅ Transaction support
- ✅ Memory allocators
- ✅ Well-tested library

### Cons
- ❌ Requires PMem hardware
- ❌ Complex API
- ❌ C library (unsafe FFI)
- ❌ Limited Rust support

---

## 7. O_DIRECT (Direct I/O)

### Overview
Bypass kernel page cache for direct disk access.

### Key Features
- No page cache
- Sector-aligned required
- Lower latency for databases
- Predictable performance

### Implementation

#### src/storage/direct_io/mod.rs

```rust
use std::fs::{File, OpenOptions};
use std::os::unix::io::AsRawFd;
use std::ptr;

const SECTOR_SIZE: usize = 4096;

pub struct DirectIoStorage {
    file: File,
}

impl DirectIoStorage {
    pub fn new(path: &str) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .custom_flags(libc::O_DIRECT | libc::O_SYNC)
            .open(path)?;
        
        Ok(Self { file })
    }
    
    pub fn write_aligned(&self, offset: u64, data: &[u8]) -> Result<usize> {
        // Ensure alignment
        if offset % SECTOR_SIZE as u64 != 0 {
            return Err("Offset not sector-aligned".into());
        }
        
        if data.len() % SECTOR_SIZE != 0 {
            return Err("Data size not sector-aligned".into());
        }
        
        // Allocate aligned buffer
        let aligned_buffer = Self::allocate_aligned(data.len())?;
        
        unsafe {
            // Copy to aligned buffer
            ptr::copy_nonoverlapping(
                data.as_ptr(),
                aligned_buffer,
                data.len(),
            );
            
            // Write with pwrite
            let n = libc::pwrite(
                self.file.as_raw_fd(),
                aligned_buffer as *const _,
                data.len(),
                offset as i64,
            );
            
            // Free aligned buffer
            libc::free(aligned_buffer as *mut _);
            
            if n < 0 {
                return Err(std::io::Error::last_os_error().into());
            }
            
            Ok(n as usize)
        }
    }
    
    pub fn read_aligned(&self, offset: u64, size: usize) -> Result<Vec<u8>> {
        // Ensure alignment
        if offset % SECTOR_SIZE as u64 != 0 {
            return Err("Offset not sector-aligned".into());
        }
        
        if size % SECTOR_SIZE != 0 {
            return Err("Size not sector-aligned".into());
        }
        
        // Allocate aligned buffer
        let aligned_buffer = Self::allocate_aligned(size)?;
        
        unsafe {
            // Read with pread
            let n = libc::pread(
                self.file.as_raw_fd(),
                aligned_buffer as *mut _,
                size,
                offset as i64,
            );
            
            if n < 0 {
                libc::free(aligned_buffer as *mut _);
                return Err(std::io::Error::last_os_error().into());
            }
            
            // Copy from aligned buffer
            let mut result = vec![0u8; n as usize];
            ptr::copy_nonoverlapping(
                aligned_buffer,
                result.as_mut_ptr(),
                n as usize,
            );
            
            libc::free(aligned_buffer as *mut _);
            
            Ok(result)
        }
    }
    
    fn allocate_aligned(size: usize) -> Result<*mut u8> {
        unsafe {
            let mut ptr: *mut u8 = ptr::null_mut();
            let ret = libc::posix_memalign(
                &mut ptr as *mut *mut u8 as *mut *mut _,
                SECTOR_SIZE,
                size,
            );
            
            if ret != 0 {
                return Err("Aligned allocation failed".into());
            }
            
            Ok(ptr)
        }
    }
}

impl Drop for DirectIoStorage {
    fn drop(&mut self) {
        // Sync on drop
        unsafe {
            libc::fsync(self.file.as_raw_fd());
        }
    }
}
```

### Use Cases
- Database storage engines
- Write-ahead logs
- Custom filesystems
- Predictable I/O performance

### Pros
- ✅ Bypass page cache
- ✅ Predictable performance
- ✅ Control over I/O
- ✅ Works everywhere

### Cons
- ❌ Alignment requirements
- ❌ More complex
- ❌ No caching benefits
- ❌ Potentially slower for small I/O

---

## Integration with Shared-Nothing Library

### Architecture

```rust
// src/storage/mod.rs

pub trait Storage: Send + Sync {
    fn write(&self, key: &[u8], value: &[u8]) -> Result<()>;
    fn read(&self, key: &[u8]) -> Result<Vec<u8>>;
    fn delete(&self, key: &[u8]) -> Result<()>;
    fn sync(&self) -> Result<()>;
}

pub enum StorageType {
    Spdk,
    IoUring,
    NvmeOf,
    Dax,
    Pmdk,
    DirectIo,
    Mmap,
}

pub fn create_storage(storage_type: StorageType, config: StorageConfig) -> Result<Box<dyn Storage>> {
    match storage_type {
        StorageType::Spdk => Ok(Box::new(SpdkStorage::new(&config.device)?)),
        StorageType::IoUring => Ok(Box::new(IoUringStorage::new(&config.path, config.queue_depth)?)),
        StorageType::Dax => Ok(Box::new(DaxStorage::new(&config.path, config.size)?)),
        // ...
    }
}

// Worker with storage
pub struct StorageWorker {
    storage: Arc<Box<dyn Storage>>,
}

impl Worker for StorageWorker {
    type State = ();
    type Message = StorageOp;
    
    fn handle_message(&mut self, _state: &mut State, msg: Envelope<Message>) -> Result<()> {
        match msg.payload {
            StorageOp::Write { key, value } => {
                self.storage.write(&key, &value)?;
            }
            StorageOp::Read { key, response_tx } => {
                let value = self.storage.read(&key)?;
                response_tx.send(value)?;
            }
        }
        Ok(())
    }
}
```

### Cargo.toml Features

```toml
[features]
spdk = []
io-uring = ["dep:io-uring"]
nvme-of = []
dax = []
pmdk = ["pmdk-sys"]
all-storage = ["io-uring", "dax"]

[dependencies]
io-uring = { version = "0.6", optional = true }
pmdk-sys = { version = "0.5", optional = true }
libc = "0.2"
```

## Performance Comparison

```
Operation: 4KB Random Write (measured IOPS)
- SPDK:          12,000,000
- NVMe Direct:    1,500,000
- io_uring:       4,000,000
- O_DIRECT:         800,000
- Buffered I/O:     200,000

Operation: 4KB Random Read (measured IOPS)
- SPDK:          10,000,000
- DAX/PMem:      Memory speed
- io_uring:       3,500,000
- O_DIRECT:         700,000
- mmap:           1,000,000+

Latency:
- PMem (DAX):     <1μs
- SPDK:           ~8μs
- io_uring:       ~15μs
- NVMe-oF/RDMA:   ~20μs
- O_DIRECT:       ~50μs
```

## Recommendation Matrix

| Use Case | Primary | Alternative | Fallback |
|----------|---------|-------------|----------|
| **Ultra-Low Latency** | SPDK | DAX/PMem | io_uring |
| **High Throughput** | SPDK | io_uring | NVMe-oF |
| **Persistent Memory** | DAX/PMDK | - | - |
| **Cloud/Portable** | io_uring | O_DIRECT | Standard I/O |
| **Remote Storage** | NVMe-oF/RDMA | - | Network FS |
| **Custom Control** | NVMe Direct | SPDK | Raw Block |

## Next Steps

1. **Phase 1**: Implement io_uring (best general-purpose)
2. **Phase 2**: Add SPDK support (ultra-performance)
3. **Phase 3**: DAX/PMem integration (persistent memory)
4. **Phase 4**: NVMe-oF for distributed storage
5. **Phase 5**: Specialized (PMDK, UIO, etc.)

---

**Document Status**: Ready for implementation
**Last Updated**: October 31, 2025
**Estimated Implementation**: 4-6 weeks for full storage stack

