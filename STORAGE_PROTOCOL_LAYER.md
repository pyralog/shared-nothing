# Storage Protocol Layer: Unified Architecture for Storage and Networking

This document defines the integration of **Dedicated Storage I/O Workers** with **Storage as Protocol** pattern, treating storage operations as protocols over the unified transport layer.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Storage I/O Workers](#storage-io-workers)
3. [Storage Protocol Traits](#storage-protocol-traits)
4. [Storage Transport Abstraction](#storage-transport-abstraction)
5. [Storage Protocol Implementations](#storage-protocol-implementations)
6. [Message Flow Patterns](#message-flow-patterns)
7. [Integration with Worker Pool](#integration-with-worker-pool)
8. [Performance Optimizations](#performance-optimizations)
9. [Use Cases and Examples](#use-cases-and-examples)

---

## Architecture Overview

### Unified I/O Worker Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Workers                           │
│              (Business Logic, Data Processing)                   │
│                      Cores 8-63                                  │
└─────────────────────────────────────────────────────────────────┘
                            ▲ │
                            │ │ Messages (Request/Response)
                            │ ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Dedicated I/O Workers                           │
│                                                                  │
│  ┌────────────────────┐              ┌────────────────────┐    │
│  │  Network I/O       │              │  Storage I/O       │    │
│  │  Workers           │              │  Workers           │    │
│  │  (Cores 0-3)       │              │  (Cores 4-7)       │    │
│  │                    │              │                    │    │
│  │  • HTTP Server     │              │  • Block Storage   │    │
│  │  • HTTP Client     │              │  • Object Storage  │    │
│  │  • gRPC            │              │  • KV Storage      │    │
│  │  • Redis           │              │  • File Storage    │    │
│  │  • PostgreSQL      │              │  • Cache           │    │
│  └────────────────────┘              └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                            ▲ │
                            │ │ Protocol Layer (Unified)
                            │ ▼
┌─────────────────────────────────────────────────────────────────┐
│              Transport Abstraction Layer                         │
│         (Unified interface for Network + Storage)                │
│                                                                  │
│  ┌────────────────────┐              ┌────────────────────┐    │
│  │  Network Transport │              │  Storage Transport │    │
│  │  • TCP/UDP         │              │  • Block Device    │    │
│  │  • RDMA            │              │  • Character Dev   │    │
│  │  • Sockets         │              │  • Memory Map      │    │
│  └────────────────────┘              └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                            ▲ │
                            │ │ Low-Level I/O
                            │ ▼
┌─────────────────────────────────────────────────────────────────┐
│                Low-Level I/O Stack                               │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  io_uring (Unified for Network + Storage)               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌────────────────────┐              ┌────────────────────┐    │
│  │  Network Layer     │              │  Storage Layer     │    │
│  │  • DPDK            │              │  • SPDK            │    │
│  │  • AF_XDP          │              │  • DAX/PMem        │    │
│  │  • Raw Sockets     │              │  • NVMe-oF         │    │
│  │  • RDMA            │              │  • O_DIRECT        │    │
│  └────────────────────┘              └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                            ▲ │
                            │ ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Hardware                                   │
│  Network: NICs, RDMA HCAs  │  Storage: NVMe SSDs, PMem          │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Unified Protocol Model**: Storage operations as protocols (like HTTP, gRPC)
2. **Dedicated Workers**: Separate CPU cores for storage I/O
3. **Transport Independence**: Storage protocols work over any transport
4. **Zero-Copy**: Minimize data copying between layers
5. **Composable**: Mix storage and network operations seamlessly
6. **Symmetric**: Same patterns for network and storage I/O

---

## Storage I/O Workers

### Worker Allocation Strategy

**64-Core Server Example:**

```
Cores 0-3:    Network I/O Workers (Server + Client)
              - HTTP/gRPC inbound requests
              - External API calls
              - Database client connections
              
Cores 4-7:    Storage I/O Workers
              - Block storage operations
              - Object storage
              - Key-value operations
              - File I/O
              
Cores 8-63:   Application Workers
              - Business logic
              - Data processing
              - Computations
              
Cores 62-63:  System/Monitoring (optional)
              - Metrics collection
              - Health checks
```

**NUMA Optimization:**

```
NUMA Node 0:
  - Cores 0-31
  - NIC 0 (network)
  - NVMe 0, 1 (storage)
  → Network I/O Workers: Cores 0-1
  → Storage I/O Workers: Cores 4-5
  → App Workers: Cores 8-31

NUMA Node 1:
  - Cores 32-63
  - NIC 1 (network)
  - NVMe 2, 3 (storage)
  → Network I/O Workers: Cores 32-33
  → Storage I/O Workers: Cores 36-37
  → App Workers: Cores 40-63
```

### Storage Worker Responsibilities

**Primary Duties:**
- Accept storage requests from application workers
- Translate to low-level storage operations
- Manage storage connections/sessions
- Handle batching and coalescing
- Implement caching strategies
- Error handling and retry logic
- Metrics and monitoring

**State Management:**
- Per-worker storage backend instances
- Connection pools for remote storage
- Buffer pools for zero-copy
- Cache for hot data
- Metadata index

---

## Storage Protocol Traits

### Core Storage Protocol Hierarchy

```rust
/// Base trait for all storage operations
pub trait StorageProtocol: Protocol {
    /// Protocol-specific addressing
    type Address;
    
    /// Protocol-specific metadata
    type Metadata;
    
    /// Addressing scheme (block/object/key-value)
    fn addressing_model(&self) -> AddressingModel;
    
    /// Consistency guarantees
    fn consistency_model(&self) -> ConsistencyModel;
}

/// Addressing models for different storage types
pub enum AddressingModel {
    Block {
        block_size: usize,
        addressing: BlockAddressing,
    },
    Object {
        namespace: String,
        hierarchical: bool,
    },
    KeyValue {
        partitioned: bool,
        sorted: bool,
    },
    File {
        hierarchical: bool,
        permissions: bool,
    },
}

pub enum BlockAddressing {
    Lba,      // Logical Block Address
    Physical, // Physical address
    Virtual,  // Virtual address space
}

pub enum ConsistencyModel {
    StrongConsistency,
    EventualConsistency,
    CausalConsistency,
    SessionConsistency,
}
```

### Block Storage Protocol

```rust
/// Block storage protocol (SPDK, NVMe, raw block devices)
pub trait BlockStorageProtocol: StorageProtocol {
    /// Read blocks
    fn read_blocks(
        &self,
        lba: u64,
        num_blocks: u32,
    ) -> impl Future<Output = Result<Vec<u8>>>;
    
    /// Write blocks
    fn write_blocks(
        &self,
        lba: u64,
        data: &[u8],
    ) -> impl Future<Output = Result<()>>;
    
    /// Vectored read (scatter)
    fn read_vectored(
        &self,
        requests: &[(u64, u32)],
    ) -> impl Future<Output = Result<Vec<Vec<u8>>>>;
    
    /// Vectored write (gather)
    fn write_vectored(
        &self,
        requests: &[(u64, &[u8])],
    ) -> impl Future<Output = Result<()>>;
    
    /// Flush/sync
    fn flush(&self) -> impl Future<Output = Result<()>>;
    
    /// Trim/discard
    fn trim(&self, lba: u64, num_blocks: u32) -> impl Future<Output = Result<()>>;
    
    /// Get block size
    fn block_size(&self) -> usize;
    
    /// Get total capacity
    fn capacity(&self) -> u64;
}
```

### Object Storage Protocol

```rust
/// Object storage protocol (S3-like, blob storage)
pub trait ObjectStorageProtocol: StorageProtocol {
    /// Put object
    fn put_object(
        &self,
        key: &str,
        data: &[u8],
        metadata: ObjectMetadata,
    ) -> impl Future<Output = Result<ObjectId>>;
    
    /// Get object
    fn get_object(
        &self,
        key: &str,
    ) -> impl Future<Output = Result<ObjectData>>;
    
    /// Delete object
    fn delete_object(&self, key: &str) -> impl Future<Output = Result<()>>;
    
    /// List objects
    fn list_objects(
        &self,
        prefix: &str,
        limit: usize,
    ) -> impl Future<Output = Result<Vec<ObjectInfo>>>;
    
    /// Multipart upload
    fn start_multipart(&self, key: &str) -> impl Future<Output = Result<UploadId>>;
    fn upload_part(
        &self,
        upload_id: UploadId,
        part_number: u32,
        data: &[u8],
    ) -> impl Future<Output = Result<()>>;
    fn complete_multipart(&self, upload_id: UploadId) -> impl Future<Output = Result<()>>;
    
    /// Stream read (for large objects)
    fn stream_object(
        &self,
        key: &str,
    ) -> impl Stream<Item = Result<Bytes>>;
}

pub struct ObjectMetadata {
    pub content_type: String,
    pub content_encoding: Option<String>,
    pub custom: HashMap<String, String>,
}

pub struct ObjectData {
    pub data: Vec<u8>,
    pub metadata: ObjectMetadata,
    pub etag: String,
}
```

### Key-Value Storage Protocol

```rust
/// Key-Value storage protocol (Redis-like, RocksDB)
pub trait KeyValueProtocol: StorageProtocol {
    /// Get value by key
    fn get(&self, key: &[u8]) -> impl Future<Output = Result<Option<Vec<u8>>>>;
    
    /// Put key-value pair
    fn put(&self, key: &[u8], value: &[u8]) -> impl Future<Output = Result<()>>;
    
    /// Delete key
    fn delete(&self, key: &[u8]) -> impl Future<Output = Result<()>>;
    
    /// Batch operations
    fn batch(&self, ops: &[KvOperation]) -> impl Future<Output = Result<()>>;
    
    /// Range scan (for sorted KV stores)
    fn scan(
        &self,
        start: &[u8],
        end: &[u8],
        limit: usize,
    ) -> impl Future<Output = Result<Vec<(Vec<u8>, Vec<u8>)>>>;
    
    /// Atomic operations
    fn compare_and_swap(
        &self,
        key: &[u8],
        expected: &[u8],
        new: &[u8],
    ) -> impl Future<Output = Result<bool>>;
    
    /// Transactions (optional)
    fn begin_transaction(&self) -> impl Future<Output = Result<TransactionId>>;
    fn commit_transaction(&self, txn_id: TransactionId) -> impl Future<Output = Result<()>>;
    fn rollback_transaction(&self, txn_id: TransactionId) -> impl Future<Output = Result<()>>;
}

pub enum KvOperation {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
    Merge { key: Vec<u8>, value: Vec<u8> },
}
```

### File Storage Protocol

```rust
/// File storage protocol (POSIX-like)
pub trait FileStorageProtocol: StorageProtocol {
    /// Open file
    fn open(
        &self,
        path: &str,
        flags: OpenFlags,
    ) -> impl Future<Output = Result<FileHandle>>;
    
    /// Read from file
    fn read(
        &self,
        handle: FileHandle,
        offset: u64,
        size: usize,
    ) -> impl Future<Output = Result<Vec<u8>>>;
    
    /// Write to file
    fn write(
        &self,
        handle: FileHandle,
        offset: u64,
        data: &[u8],
    ) -> impl Future<Output = Result<usize>>;
    
    /// Close file
    fn close(&self, handle: FileHandle) -> impl Future<Output = Result<()>>;
    
    /// Metadata operations
    fn stat(&self, path: &str) -> impl Future<Output = Result<FileMetadata>>;
    fn mkdir(&self, path: &str) -> impl Future<Output = Result<()>>;
    fn readdir(&self, path: &str) -> impl Future<Output = Result<Vec<DirEntry>>>;
    
    /// Sync operations
    fn fsync(&self, handle: FileHandle) -> impl Future<Output = Result<()>>;
    fn fdatasync(&self, handle: FileHandle) -> impl Future<Output = Result<()>>;
}

pub struct FileHandle(u64);

pub struct FileMetadata {
    pub size: u64,
    pub created: SystemTime,
    pub modified: SystemTime,
    pub permissions: u32,
}
```

---

## Storage Transport Abstraction

### Unified Transport for Storage

```rust
/// Storage-specific transport operations
pub trait StorageTransport: Transport {
    /// Direct memory access (zero-copy)
    fn map_memory(
        &self,
        offset: u64,
        size: usize,
        flags: MapFlags,
    ) -> Result<MemoryMap>;
    
    /// Scatter-gather I/O
    fn readv(
        &self,
        iovec: &[IoVec],
    ) -> impl Future<Output = Result<usize>>;
    
    fn writev(
        &self,
        iovec: &[IoVec],
    ) -> impl Future<Output = Result<usize>>;
    
    /// Direct buffer management
    fn allocate_buffer(
        &self,
        size: usize,
        alignment: usize,
    ) -> Result<DmaBuffer>;
    
    /// Atomic operations (for persistent memory)
    fn atomic_write(
        &self,
        offset: u64,
        data: &[u8],
    ) -> impl Future<Output = Result<()>>;
    
    /// Flush/persist operations
    fn flush_range(
        &self,
        offset: u64,
        size: usize,
    ) -> impl Future<Output = Result<()>>;
    
    /// Storage-specific properties
    fn block_size(&self) -> usize;
    fn is_persistent(&self) -> bool;
    fn supports_atomic_writes(&self) -> bool;
}

pub struct IoVec {
    pub base: *const u8,
    pub len: usize,
}

pub struct DmaBuffer {
    ptr: *mut u8,
    size: usize,
    alignment: usize,
}

pub struct MemoryMap {
    addr: *mut u8,
    size: usize,
    offset: u64,
}

pub enum MapFlags {
    ReadOnly,
    ReadWrite,
    Private,
    Shared,
    Sync,  // MAP_SYNC for DAX
}
```

### Storage Transport Implementations

```rust
/// io_uring transport (network + storage unified)
pub struct IoUringTransport {
    ring: IoUring,
    fd: RawFd,
    transport_type: TransportType,
}

pub enum TransportType {
    Network(NetworkType),
    Storage(StorageType),
}

pub enum NetworkType {
    TcpSocket,
    UdpSocket,
    UnixSocket,
}

pub enum StorageType {
    BlockDevice,
    RegularFile,
    CharDevice,
    DirectIo,
}

impl IoUringTransport {
    /// Unified read for network or storage
    pub async fn read(&self, buf: &mut [u8]) -> Result<usize> {
        match self.transport_type {
            TransportType::Network(_) => self.read_network(buf).await,
            TransportType::Storage(_) => self.read_storage(buf).await,
        }
    }
    
    /// Unified write for network or storage
    pub async fn write(&self, buf: &[u8]) -> Result<usize> {
        match self.transport_type {
            TransportType::Network(_) => self.write_network(buf).await,
            TransportType::Storage(_) => self.write_storage(buf).await,
        }
    }
    
    /// Storage-specific read at offset
    pub async fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<usize> {
        let fd = types::Fd(self.fd);
        let read_e = opcode::Read::new(fd, buf.as_mut_ptr(), buf.len() as _)
            .offset(offset)
            .build();
        
        // Submit and wait for completion
        // ... io_uring operation ...
        Ok(0)
    }
}

/// SPDK transport (storage only)
pub struct SpdkTransport {
    controller: *mut spdk_nvme_ctrlr,
    namespace: *mut spdk_nvme_ns,
    qpair: *mut spdk_nvme_qpair,
}

impl StorageTransport for SpdkTransport {
    fn block_size(&self) -> usize {
        unsafe {
            spdk_nvme_ns_get_sector_size(self.namespace) as usize
        }
    }
    
    fn is_persistent(&self) -> bool {
        true
    }
    
    fn supports_atomic_writes(&self) -> bool {
        false // NVMe doesn't guarantee atomicity > 1 block
    }
}

/// DAX/PMem transport (direct memory access)
pub struct DaxTransport {
    mapping: *mut u8,
    size: usize,
    file: File,
}

impl StorageTransport for DaxTransport {
    fn block_size(&self) -> usize {
        1 // Byte-addressable
    }
    
    fn is_persistent(&self) -> bool {
        true
    }
    
    fn supports_atomic_writes(&self) -> bool {
        true // Up to cache line size (64 bytes)
    }
    
    fn map_memory(&self, offset: u64, size: usize, _flags: MapFlags) -> Result<MemoryMap> {
        Ok(MemoryMap {
            addr: unsafe { self.mapping.add(offset as usize) },
            size,
            offset,
        })
    }
}

/// NVMe-oF transport (remote storage)
pub struct NvmeOfTransport {
    rdma_connection: Option<RdmaConnection>,
    tcp_connection: Option<TcpStream>,
    transport_type: NvmeOfTransportType,
}

pub enum NvmeOfTransportType {
    Rdma,
    Tcp,
    Fc,
}

/// RDMA transport (can be used for both network and storage)
pub struct RdmaTransport {
    qp: *mut ibv_qp,
    local_mr: MemoryRegion,
    remote_mr: Option<RemoteMemoryRegion>,
    usage: RdmaUsage,
}

pub enum RdmaUsage {
    Network,
    Storage,
    Both,
}
```

---

## Storage Protocol Implementations

### 1. SPDK Block Storage Protocol

```rust
pub struct SpdkBlockProtocol {
    transport: Arc<SpdkTransport>,
    block_size: usize,
    capacity: u64,
}

impl BlockStorageProtocol for SpdkBlockProtocol {
    async fn read_blocks(&self, lba: u64, num_blocks: u32) -> Result<Vec<u8>> {
        let size = num_blocks as usize * self.block_size;
        let mut buffer = self.transport.allocate_buffer(size, self.block_size)?;
        
        // Submit SPDK read command
        unsafe {
            let rc = spdk_nvme_ns_cmd_read(
                self.transport.namespace,
                self.transport.qpair,
                buffer.ptr as *mut _,
                lba,
                num_blocks,
                Some(completion_callback),
                std::ptr::null_mut(),
                0,
            );
            
            if rc != 0 {
                return Err("SPDK read failed".into());
            }
            
            // Poll for completion
            while spdk_nvme_qpair_process_completions(self.transport.qpair, 0) == 0 {
                // Busy poll
            }
        }
        
        // Convert DMA buffer to Vec
        Ok(buffer.to_vec())
    }
    
    async fn write_blocks(&self, lba: u64, data: &[u8]) -> Result<()> {
        let num_blocks = (data.len() + self.block_size - 1) / self.block_size;
        let mut buffer = self.transport.allocate_buffer(data.len(), self.block_size)?;
        
        // Copy to DMA buffer
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), buffer.ptr, data.len());
            
            // Submit SPDK write command
            let rc = spdk_nvme_ns_cmd_write(
                self.transport.namespace,
                self.transport.qpair,
                buffer.ptr as *mut _,
                lba,
                num_blocks as u32,
                Some(completion_callback),
                std::ptr::null_mut(),
                0,
            );
            
            if rc != 0 {
                return Err("SPDK write failed".into());
            }
            
            // Poll for completion
            while spdk_nvme_qpair_process_completions(self.transport.qpair, 0) == 0 {
                // Busy poll
            }
        }
        
        Ok(())
    }
    
    async fn write_vectored(&self, requests: &[(u64, &[u8])]) -> Result<()> {
        // Batch multiple writes
        for (lba, data) in requests {
            // Submit without waiting
            self.submit_write(*lba, data)?;
        }
        
        // Wait for all completions
        self.wait_all_completions(requests.len())?;
        
        Ok(())
    }
    
    fn block_size(&self) -> usize {
        self.block_size
    }
    
    fn capacity(&self) -> u64 {
        self.capacity
    }
}
```

### 2. io_uring Block Storage Protocol

```rust
pub struct IoUringBlockProtocol {
    transport: Arc<IoUringTransport>,
    block_size: usize,
}

impl BlockStorageProtocol for IoUringBlockProtocol {
    async fn read_blocks(&self, lba: u64, num_blocks: u32) -> Result<Vec<u8>> {
        let offset = lba * self.block_size as u64;
        let size = num_blocks as usize * self.block_size;
        
        let mut buffer = vec![0u8; size];
        
        // Use io_uring read with offset
        self.transport.read_at(offset, &mut buffer).await?;
        
        Ok(buffer)
    }
    
    async fn write_blocks(&self, lba: u64, data: &[u8]) -> Result<()> {
        let offset = lba * self.block_size as u64;
        
        // Use io_uring write with offset
        self.transport.write_at(offset, data).await?;
        
        Ok(())
    }
    
    async fn read_vectored(&self, requests: &[(u64, u32)]) -> Result<Vec<Vec<u8>>> {
        // Convert to iovec
        let mut iovecs = Vec::new();
        let mut buffers = Vec::new();
        
        for (lba, num_blocks) in requests {
            let size = *num_blocks as usize * self.block_size;
            let buffer = vec![0u8; size];
            iovecs.push(IoVec {
                base: buffer.as_ptr(),
                len: size,
            });
            buffers.push(buffer);
        }
        
        // Single io_uring readv operation
        self.transport.readv(&iovecs).await?;
        
        Ok(buffers)
    }
    
    async fn flush(&self) -> Result<()> {
        // Use io_uring fsync
        self.transport.sync().await?;
        Ok(())
    }
    
    fn block_size(&self) -> usize {
        self.block_size
    }
}
```

### 3. DAX/PMem Object Storage Protocol

```rust
pub struct DaxObjectProtocol {
    transport: Arc<DaxTransport>,
    index: Arc<Mutex<HashMap<String, ObjectLocation>>>,
    allocator: Arc<Mutex<Allocator>>,
}

struct ObjectLocation {
    offset: u64,
    size: usize,
    metadata_offset: u64,
}

impl ObjectStorageProtocol for DaxObjectProtocol {
    async fn put_object(
        &self,
        key: &str,
        data: &[u8],
        metadata: ObjectMetadata,
    ) -> Result<ObjectId> {
        // Allocate space in persistent memory
        let location = self.allocator.lock().allocate(data.len() + 4096)?;
        
        // Get memory map
        let map = self.transport.map_memory(
            location.offset,
            data.len(),
            MapFlags::ReadWrite | MapFlags::Sync,
        )?;
        
        // Direct memory copy (persistent)
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), map.addr, data.len());
            
            // Flush to persistence
            self.transport.flush_range(location.offset, data.len()).await?;
        }
        
        // Update index
        self.index.lock().insert(key.to_string(), ObjectLocation {
            offset: location.offset,
            size: data.len(),
            metadata_offset: location.offset + data.len() as u64,
        });
        
        Ok(ObjectId(location.offset))
    }
    
    async fn get_object(&self, key: &str) -> Result<ObjectData> {
        // Lookup in index
        let location = self.index.lock()
            .get(key)
            .cloned()
            .ok_or("Object not found")?;
        
        // Direct memory access (zero-copy)
        let map = self.transport.map_memory(
            location.offset,
            location.size,
            MapFlags::ReadOnly,
        )?;
        
        // Read from persistent memory
        let data = unsafe {
            std::slice::from_raw_parts(map.addr, location.size).to_vec()
        };
        
        Ok(ObjectData {
            data,
            metadata: ObjectMetadata::default(),
            etag: format!("{:x}", location.offset),
        })
    }
    
    fn stream_object(&self, key: &str) -> impl Stream<Item = Result<Bytes>> {
        // Return stream that yields chunks from PMem
        stream! {
            let location = self.index.lock().get(key).cloned();
            if let Some(loc) = location {
                const CHUNK_SIZE: usize = 64 * 1024;
                for offset in (0..loc.size).step_by(CHUNK_SIZE) {
                    let chunk_size = std::cmp::min(CHUNK_SIZE, loc.size - offset);
                    let map = self.transport.map_memory(
                        loc.offset + offset as u64,
                        chunk_size,
                        MapFlags::ReadOnly,
                    )?;
                    
                    let chunk = unsafe {
                        std::slice::from_raw_parts(map.addr, chunk_size)
                    };
                    yield Ok(Bytes::copy_from_slice(chunk));
                }
            }
        }
    }
}
```

### 4. Key-Value over Block Storage

```rust
pub struct BlockKvProtocol {
    block_protocol: Arc<dyn BlockStorageProtocol>,
    index: Arc<RwLock<BTreeMap<Vec<u8>, u64>>>,  // key -> LBA
    allocator: Arc<Mutex<BlockAllocator>>,
}

impl KeyValueProtocol for BlockKvProtocol {
    async fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        // Lookup LBA in index
        let lba = {
            let index = self.index.read();
            match index.get(key) {
                Some(&lba) => lba,
                None => return Ok(None),
            }
        };
        
        // Read block
        let block_size = self.block_protocol.block_size();
        let data = self.block_protocol.read_blocks(lba, 1).await?;
        
        // Parse block (first 4 bytes = size)
        let size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        
        Ok(Some(data[4..4 + size].to_vec()))
    }
    
    async fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        // Allocate block
        let lba = self.allocator.lock().allocate()?;
        
        // Prepare block (size + data)
        let block_size = self.block_protocol.block_size();
        let mut block = vec![0u8; block_size];
        let size = value.len() as u32;
        block[0..4].copy_from_slice(&size.to_le_bytes());
        block[4..4 + value.len()].copy_from_slice(value);
        
        // Write block
        self.block_protocol.write_blocks(lba, &block).await?;
        
        // Update index
        self.index.write().insert(key.to_vec(), lba);
        
        Ok(())
    }
    
    async fn delete(&self, key: &[u8]) -> Result<()> {
        // Remove from index
        let lba = self.index.write().remove(key)
            .ok_or("Key not found")?;
        
        // Free block
        self.allocator.lock().free(lba)?;
        
        Ok(())
    }
    
    async fn scan(
        &self,
        start: &[u8],
        end: &[u8],
        limit: usize,
    ) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let index = self.index.read();
        let mut results = Vec::new();
        
        for (key, &lba) in index.range(start.to_vec()..end.to_vec()).take(limit) {
            let value = self.get(key).await?.unwrap_or_default();
            results.push((key.clone(), value));
        }
        
        Ok(results)
    }
}
```

### 5. NVMe-oF Remote Block Storage

```rust
pub struct NvmeOfBlockProtocol {
    transport: Arc<NvmeOfTransport>,
    subsystem_nqn: String,
    block_size: usize,
}

impl BlockStorageProtocol for NvmeOfBlockProtocol {
    async fn read_blocks(&self, lba: u64, num_blocks: u32) -> Result<Vec<u8>> {
        match &self.transport.transport_type {
            NvmeOfTransportType::Rdma => {
                // RDMA read (zero-copy)
                let rdma_conn = self.transport.rdma_connection.as_ref().unwrap();
                let remote_addr = lba * self.block_size as u64;
                let size = num_blocks as usize * self.block_size;
                
                rdma_conn.read(remote_addr, size).await
            }
            NvmeOfTransportType::Tcp => {
                // TCP-based NVMe command
                self.send_nvme_read_command(lba, num_blocks).await
            }
            NvmeOfTransportType::Fc => {
                // Fibre Channel
                unimplemented!("FC support")
            }
        }
    }
    
    async fn write_blocks(&self, lba: u64, data: &[u8]) -> Result<()> {
        match &self.transport.transport_type {
            NvmeOfTransportType::Rdma => {
                // RDMA write (zero-copy)
                let rdma_conn = self.transport.rdma_connection.as_ref().unwrap();
                let remote_addr = lba * self.block_size as u64;
                
                rdma_conn.write(remote_addr, data).await
            }
            NvmeOfTransportType::Tcp => {
                // TCP-based NVMe command
                self.send_nvme_write_command(lba, data).await
            }
            NvmeOfTransportType::Fc => {
                unimplemented!("FC support")
            }
        }
    }
}
```

---

## Message Flow Patterns

### Pattern 1: Simple Block Write

```
┌─────────────┐  1. Write Request   ┌─────────────┐
│ Application │ ───────────────────> │  Storage    │
│   Worker    │                      │ I/O Worker  │
│             │  4. Ack Response     │             │
│             │ <─────────────────── │             │
└─────────────┘                      └─────────────┘
                                             │
                                             │ 2. Block Write
                                             │    (SPDK/io_uring)
                                             ▼
                                     ┌─────────────┐
                                     │   Storage   │
                                     │   Backend   │
                                     │  (NVMe SSD) │
                                     └─────────────┘
                                             │
                                             │ 3. Completion
                                             ▼
```

**Message Sequence:**

```rust
// Application Worker
let msg = StorageMessage::BlockWrite {
    lba: 1000,
    data: vec![...],
    correlation_id: 12345,
};
storage_worker_tx.send(msg).await?;

// Wait for response
let response = response_rx.recv().await?;

// Storage I/O Worker
match msg {
    StorageMessage::BlockWrite { lba, data, correlation_id } => {
        // Use block protocol
        block_protocol.write_blocks(lba, &data).await?;
        
        // Send ack back
        let response = StorageResponse::WriteAck {
            correlation_id,
            result: Ok(()),
        };
        app_worker_tx.send(response).await?;
    }
}
```

### Pattern 2: Read with Caching

```
┌─────────────┐  1. Read Request    ┌─────────────┐
│ Application │ ───────────────────> │  Storage    │
│   Worker    │                      │ I/O Worker  │
│             │                      │             │
│             │                      │ ┌─────────┐ │
│             │  2. Cache Hit!       │ │  Cache  │ │
│             │                      │ └─────────┘ │
│             │  3. Return Data      │             │
│             │ <─────────────────── │             │
└─────────────┘                      └─────────────┘
                                      (No storage access)

Or on cache miss:

┌─────────────┐  1. Read Request    ┌─────────────┐
│ Application │ ───────────────────> │  Storage    │
│   Worker    │                      │ I/O Worker  │
│             │                      │             │
│             │  5. Return Data      │ ┌─────────┐ │
│             │ <─────────────────── │ │  Cache  │ │
│             │                      │ └─────────┘ │
└─────────────┘                      └──────┬──────┘
                                            │ 2. Cache Miss
                                            │ 3. Read from Storage
                                            ▼
                                     ┌─────────────┐
                                     │   Storage   │
                                     │   Backend   │
                                     └─────────────┘
                                            │ 4. Data
                                            ▼ (Update cache)
```

### Pattern 3: Batch Operations

```
┌─────────────┐  1. Multiple Writes  ┌─────────────┐
│ Application │ ───────────────────> │  Storage    │
│  Worker A   │     (write lba=100)  │ I/O Worker  │
└─────────────┘                      │             │
                                     │  ┌────────┐ │
┌─────────────┐  2. More Writes      │  │ Batch  │ │
│ Application │ ───────────────────> │  │ Queue  │ │
│  Worker B   │     (write lba=200)  │  └────────┘ │
└─────────────┘                      │             │
                                     │  3. Flush    │
┌─────────────┐  3. Even More        │  (vectored)  │
│ Application │ ───────────────────> │             │
│  Worker C   │     (write lba=300)  └──────┬──────┘
└─────────────┘                             │
                                            │ 4. Batch Write
                                            ▼
                                     ┌─────────────┐
                                     │   Storage   │
                                     │   Backend   │
                                     └─────────────┘
```

**Implementation:**

```rust
pub struct StorageIoWorker {
    batch_queue: VecDeque<StorageOperation>,
    batch_timeout: Duration,
    max_batch_size: usize,
}

impl StorageIoWorker {
    async fn run(&mut self) {
        loop {
            select! {
                // Receive new operation
                msg = self.rx.recv() => {
                    self.batch_queue.push_back(msg);
                    
                    // Flush if batch full
                    if self.batch_queue.len() >= self.max_batch_size {
                        self.flush_batch().await?;
                    }
                }
                
                // Timeout - flush partial batch
                _ = sleep(self.batch_timeout) => {
                    if !self.batch_queue.is_empty() {
                        self.flush_batch().await?;
                    }
                }
            }
        }
    }
    
    async fn flush_batch(&mut self) -> Result<()> {
        // Collect all writes
        let writes: Vec<_> = self.batch_queue
            .drain(..)
            .filter_map(|op| match op {
                StorageOperation::Write { lba, data } => Some((lba, data)),
                _ => None,
            })
            .collect();
        
        // Single vectored write
        self.block_protocol.write_vectored(&writes).await?;
        
        // Send all acks
        // ...
        
        Ok(())
    }
}
```

### Pattern 4: Tiered Storage

```
┌─────────────┐  1. Write          ┌─────────────┐
│ Application │ ─────────────────> │  Storage    │
│   Worker    │                    │ I/O Worker  │
│             │                    │             │
│             │                    │  2. Tier    │
│             │                    │  Decision   │
│             │                    └──────┬──────┘
└─────────────┘                           │
                         ┌────────────────┼────────────────┐
                         │                │                │
                    3a. Hot Data     3b. Warm Data   3c. Cold Data
                         │                │                │
                         ▼                ▼                ▼
                  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
                  │    PMem     │  │  Local NVMe │  │  NVMe-oF    │
                  │ (DAX/PMDK)  │  │ (SPDK/      │  │  (Remote)   │
                  │             │  │  io_uring)  │  │             │
                  │  <1μs       │  │  ~10μs      │  │  ~100μs     │
                  └─────────────┘  └─────────────┘  └─────────────┘
```

**Tier Management:**

```rust
pub struct TieredStorageProtocol {
    hot_tier: Arc<DaxObjectProtocol>,
    warm_tier: Arc<SpdkBlockProtocol>,
    cold_tier: Arc<NvmeOfBlockProtocol>,
    tier_manager: Arc<TierManager>,
}

impl KeyValueProtocol for TieredStorageProtocol {
    async fn put(&self, key: &[u8], value: &[u8]) -> Result<()> {
        // Decide tier based on access pattern
        let tier = self.tier_manager.select_tier(key, value.len());
        
        match tier {
            Tier::Hot => {
                // Write to PMem (fastest)
                self.hot_tier.put_object(
                    &String::from_utf8_lossy(key),
                    value,
                    ObjectMetadata::default(),
                ).await?;
            }
            Tier::Warm => {
                // Write to local NVMe
                let lba = self.allocate_lba(value.len())?;
                self.warm_tier.write_blocks(lba, value).await?;
            }
            Tier::Cold => {
                // Write to remote storage
                let lba = self.allocate_remote_lba(value.len())?;
                self.cold_tier.write_blocks(lba, value).await?;
            }
        }
        
        // Update tier metadata
        self.tier_manager.record_write(key, tier);
        
        Ok(())
    }
    
    async fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        // Check tiers in order (hot -> warm -> cold)
        
        // Try hot tier
        if let Some(data) = self.hot_tier.get_object(
            &String::from_utf8_lossy(key)
        ).await.ok() {
            return Ok(Some(data.data));
        }
        
        // Try warm tier
        if let Some(lba) = self.tier_manager.get_warm_lba(key) {
            if let Ok(data) = self.warm_tier.read_blocks(lba, 1).await {
                // Promote to hot if frequently accessed
                if self.tier_manager.should_promote(key) {
                    self.promote_to_hot(key, &data).await?;
                }
                return Ok(Some(data));
            }
        }
        
        // Try cold tier
        if let Some(lba) = self.tier_manager.get_cold_lba(key) {
            if let Ok(data) = self.cold_tier.read_blocks(lba, 1).await {
                return Ok(Some(data));
            }
        }
        
        Ok(None)
    }
}
```

### Pattern 5: Replication with Consensus

```
┌─────────────┐  1. Write Request  ┌─────────────┐
│ Application │ ─────────────────> │  Storage    │
│   Worker    │                    │ I/O Worker  │
│             │                    │  (Leader)   │
└─────────────┘                    └──────┬──────┘
                                          │
                   2. Replicate to quorum │
                ┌──────────────┬──────────┼──────────┐
                │              │          │          │
                ▼              ▼          ▼          ▼
         ┌───────────┐  ┌───────────┐  ┌───────────┐
         │ Storage   │  │ Storage   │  │ Storage   │
         │ Worker    │  │ Worker    │  │ Worker    │
         │(Follower) │  │(Follower) │  │(Follower) │
         └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
               │              │              │
               │ 3. Write     │              │
               ▼              ▼              ▼
         ┌───────────┐  ┌───────────┐  ┌───────────┐
         │ Local     │  │ Local     │  │ Local     │
         │ Storage   │  │ Storage   │  │ Storage   │
         └───────────┘  └───────────┘  └───────────┘
               │              │              │
               └──────────────┴──────────────┘
                              │
                     4. Quorum achieved
                              │
                              ▼
                        5. Commit & Ack
```

---

## Integration with Worker Pool

### Storage Worker Pool Configuration

```rust
pub struct StorageWorkerPoolConfig {
    /// Number of storage I/O workers
    pub num_workers: usize,
    
    /// CPU cores to pin workers to
    pub cpu_cores: Vec<usize>,
    
    /// Storage backend type
    pub backend: StorageBackend,
    
    /// Protocol type
    pub protocol: StorageProtocolType,
    
    /// Performance tuning
    pub batch_size: usize,
    pub batch_timeout: Duration,
    pub cache_size: usize,
    
    /// Replication
    pub replication_factor: usize,
    pub consistency: ConsistencyLevel,
}

pub enum StorageBackend {
    Spdk { device: String },
    IoUring { path: String, queue_depth: u32 },
    Dax { path: String, size: usize },
    NvmeOf { address: String, subsystem_nqn: String },
    Tiered {
        hot: Box<StorageBackend>,
        warm: Box<StorageBackend>,
        cold: Box<StorageBackend>,
    },
}

pub enum StorageProtocolType {
    Block,
    Object,
    KeyValue,
    File,
}
```

### Creating Storage Worker Pool

```rust
pub fn create_storage_worker_pool(
    config: StorageWorkerPoolConfig,
) -> Result<StorageWorkerPool> {
    let mut workers = Vec::new();
    
    // Create storage backend
    let backend = match config.backend {
        StorageBackend::Spdk { device } => {
            let transport = SpdkTransport::new(&device)?;
            let protocol = SpdkBlockProtocol::new(Arc::new(transport))?;
            Arc::new(protocol) as Arc<dyn BlockStorageProtocol>
        }
        StorageBackend::IoUring { path, queue_depth } => {
            let transport = IoUringTransport::new_storage(&path, queue_depth)?;
            let protocol = IoUringBlockProtocol::new(Arc::new(transport))?;
            Arc::new(protocol) as Arc<dyn BlockStorageProtocol>
        }
        // ... other backends
    };
    
    // Create workers
    for i in 0..config.num_workers {
        let worker = StorageIoWorker::new(
            i,
            backend.clone(),
            config.cpu_cores.get(i).copied(),
        )?;
        
        let handle = worker.spawn()?;
        workers.push(handle);
    }
    
    Ok(StorageWorkerPool {
        workers,
        partitioner: Arc::new(HashPartitioner::default()),
    })
}
```

### Application Worker Integration

```rust
pub struct AppWorkerWithStorage {
    worker_id: usize,
    storage_pool: Arc<StorageWorkerPool>,
}

impl Worker for AppWorkerWithStorage {
    type State = AppState;
    type Message = AppMessage;
    
    fn handle_message(&mut self, state: &mut Self::State, msg: Envelope<Self::Message>) -> Result<()> {
        match msg.payload {
            AppMessage::ProcessData { key, data } => {
                // Process data
                let result = self.process(data);
                
                // Store result via storage workers
                self.storage_pool.put(&key, &result).await?;
                
                Ok(())
            }
            AppMessage::QueryData { key, response_tx } => {
                // Fetch from storage
                let data = self.storage_pool.get(&key).await?;
                
                // Send response
                response_tx.send(data)?;
                
                Ok(())
            }
        }
    }
}
```

### Unified I/O Worker Manager

```rust
pub struct IoWorkerManager {
    network_workers: Vec<NetworkIoWorker>,
    storage_workers: Vec<StorageIoWorker>,
    cpu_topology: CpuTopology,
}

impl IoWorkerManager {
    pub fn new(config: IoWorkerConfig) -> Result<Self> {
        let cpu_topology = CpuTopology::detect()?;
        
        // Allocate cores based on NUMA topology
        let (network_cores, storage_cores) = Self::allocate_cores(
            &cpu_topology,
            config.num_network_workers,
            config.num_storage_workers,
        );
        
        // Create network workers
        let network_workers = Self::create_network_workers(
            config.network_config,
            &network_cores,
        )?;
        
        // Create storage workers
        let storage_workers = Self::create_storage_workers(
            config.storage_config,
            &storage_cores,
        )?;
        
        Ok(Self {
            network_workers,
            storage_workers,
            cpu_topology,
        })
    }
    
    fn allocate_cores(
        topology: &CpuTopology,
        num_network: usize,
        num_storage: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let mut network_cores = Vec::new();
        let mut storage_cores = Vec::new();
        
        // For each NUMA node
        for node in topology.nodes() {
            // Network workers on cores near NICs
            for core in node.cores_near_devices(&["network"]).take(num_network / 2) {
                network_cores.push(core);
            }
            
            // Storage workers on cores near storage controllers
            for core in node.cores_near_devices(&["storage"]).take(num_storage / 2) {
                storage_cores.push(core);
            }
        }
        
        (network_cores, storage_cores)
    }
}
```

---

## Performance Optimizations

### 1. Zero-Copy Storage

```rust
pub struct ZeroCopyStorageProtocol {
    transport: Arc<DaxTransport>,
}

impl ZeroCopyStorageProtocol {
    /// Get direct memory reference (no copy)
    pub fn get_zero_copy(&self, key: &[u8]) -> Result<&[u8]> {
        let location = self.index.get(key)?;
        
        // Return slice directly into persistent memory
        unsafe {
            Ok(std::slice::from_raw_parts(
                self.transport.mapping.add(location.offset as usize),
                location.size,
            ))
        }
    }
    
    /// Put with zero-copy (caller provides aligned buffer)
    pub fn put_zero_copy(&self, key: &[u8], buffer: DmaBuffer) -> Result<()> {
        // Buffer already in correct location
        // Just update index
        self.index.insert(key, buffer.location());
        
        // Ensure persistence
        self.transport.flush_range(buffer.offset(), buffer.size())?;
        
        Ok(())
    }
}
```

### 2. Storage Buffer Pooling

```rust
pub struct StorageBufferPool {
    block_size: usize,
    pools: Vec<Mutex<Vec<DmaBuffer>>>,
}

impl StorageBufferPool {
    pub fn allocate(&self, size: usize) -> Result<DmaBuffer> {
        let pool_idx = size / self.block_size;
        
        // Try to get from pool
        if let Some(buffer) = self.pools[pool_idx].lock().pop() {
            return Ok(buffer);
        }
        
        // Allocate new aligned buffer
        self.allocate_new(size)
    }
    
    pub fn release(&self, buffer: DmaBuffer) {
        let pool_idx = buffer.size() / self.block_size;
        self.pools[pool_idx].lock().push(buffer);
    }
}
```

### 3. Prefetching

```rust
pub struct PrefetchingStorageProtocol {
    protocol: Arc<dyn BlockStorageProtocol>,
    access_predictor: AccessPredictor,
    prefetch_queue: Arc<Mutex<VecDeque<PrefetchRequest>>>,
}

impl PrefetchingStorageProtocol {
    pub async fn get_with_prefetch(&self, key: &[u8]) -> Result<Vec<u8>> {
        // Prefetch likely next accesses
        let predictions = self.access_predictor.predict_next(key);
        
        for next_key in predictions {
            self.prefetch_queue.lock().push_back(PrefetchRequest {
                key: next_key,
                priority: PrefetchPriority::Medium,
            });
        }
        
        // Actual read
        self.protocol.get(key).await
    }
    
    async fn prefetch_worker(&self) {
        loop {
            if let Some(req) = self.prefetch_queue.lock().pop_front() {
                // Load into cache
                let _ = self.protocol.get(&req.key).await;
            } else {
                sleep(Duration::from_micros(100)).await;
            }
        }
    }
}
```

### 4. Write Combining

```rust
pub struct WriteCombiningProtocol {
    protocol: Arc<dyn BlockStorageProtocol>,
    write_buffer: Arc<Mutex<WriteBuffer>>,
}

struct WriteBuffer {
    entries: HashMap<u64, Vec<u8>>,  // LBA -> pending data
    dirty_lbas: BTreeSet<u64>,
}

impl WriteCombiningProtocol {
    pub async fn write(&self, lba: u64, data: &[u8]) -> Result<()> {
        {
            let mut buffer = self.write_buffer.lock();
            
            // Combine with existing write to same LBA
            buffer.entries.entry(lba)
                .and_modify(|existing| {
                    // Merge writes
                    existing.copy_from_slice(data);
                })
                .or_insert_with(|| data.to_vec());
            
            buffer.dirty_lbas.insert(lba);
        }
        
        // Flush if too many pending
        if self.write_buffer.lock().dirty_lbas.len() > 256 {
            self.flush().await?;
        }
        
        Ok(())
    }
    
    async fn flush(&self) -> Result<()> {
        let writes = {
            let mut buffer = self.write_buffer.lock();
            
            // Drain pending writes
            let writes: Vec<_> = buffer.dirty_lbas.iter()
                .map(|&lba| (lba, buffer.entries.get(&lba).unwrap().as_slice()))
                .collect();
            
            buffer.dirty_lbas.clear();
            buffer.entries.clear();
            
            writes
        };
        
        // Single vectored write
        self.protocol.write_vectored(&writes).await
    }
}
```

---

## Use Cases and Examples

### Use Case 1: High-Performance Key-Value Store

```rust
pub struct KvStoreService {
    app_workers: WorkerPool<KvWorker>,
    storage_workers: StorageWorkerPool,
    network_workers: NetworkWorkerPool,
}

impl KvStoreService {
    pub async fn new() -> Result<Self> {
        // Storage workers with SPDK
        let storage_config = StorageWorkerPoolConfig {
            num_workers: 4,
            cpu_cores: vec![4, 5, 6, 7],
            backend: StorageBackend::Spdk {
                device: "0000:03:00.0".to_string(),
            },
            protocol: StorageProtocolType::KeyValue,
            batch_size: 256,
            batch_timeout: Duration::from_micros(100),
            cache_size: 1_000_000,
            replication_factor: 3,
            consistency: ConsistencyLevel::Quorum,
        };
        
        let storage_workers = create_storage_worker_pool(storage_config)?;
        
        // Network workers with io_uring
        let network_config = NetworkWorkerPoolConfig {
            num_workers: 4,
            cpu_cores: vec![0, 1, 2, 3],
            protocol: ProtocolType::Http2,
            transport: TransportType::IoUring,
        };
        
        let network_workers = create_network_worker_pool(network_config)?;
        
        // Application workers
        let app_config = WorkerPoolConfig {
            num_workers: 56,
            cpu_cores: (8..64).collect(),
        };
        
        let app_workers = WorkerPool::new(app_config, || {
            KvWorker::new(storage_workers.clone())
        })?;
        
        Ok(Self {
            app_workers,
            storage_workers,
            network_workers,
        })
    }
}

// Latency target: < 100μs p99
// Throughput target: 10M ops/sec
```

### Use Case 2: Object Storage with Tiering

```rust
pub struct ObjectStoreService {
    storage_config: TieredStorageConfig,
}

pub struct TieredStorageConfig {
    hot_tier: DaxConfig,       // PMem for hot data
    warm_tier: SpdkConfig,     // Local NVMe
    cold_tier: NvmeOfConfig,   // Remote storage
}

impl ObjectStoreService {
    pub async fn put_object(&self, key: &str, data: &[u8]) -> Result<()> {
        let size = data.len();
        
        // Small objects -> PMem (hot)
        if size < 64 * 1024 {
            self.hot_tier.put(key, data).await?;
        }
        // Medium objects -> Local NVMe (warm)
        else if size < 10 * 1024 * 1024 {
            self.warm_tier.put(key, data).await?;
        }
        // Large objects -> Remote storage (cold)
        else {
            self.cold_tier.put(key, data).await?;
        }
        
        Ok(())
    }
}
```

### Use Case 3: Time-Series Database

```rust
pub struct TimeSeriesDb {
    write_log: Arc<DaxWriteLog>,          // PMem write-ahead log
    compaction_worker: CompactionWorker,
    storage_tiers: TieredBlockStorage,
}

impl TimeSeriesDb {
    pub async fn insert(&self, timestamp: u64, data: &[u8]) -> Result<()> {
        // Fast write to PMem WAL
        self.write_log.append(timestamp, data).await?;
        
        // Background compaction to NVMe
        self.compaction_worker.notify();
        
        Ok(())
    }
    
    pub async fn query(&self, start: u64, end: u64) -> Result<Vec<DataPoint>> {
        // Query recent data from PMem
        let mut results = self.write_log.scan(start, end).await?;
        
        // Query older data from NVMe tiers
        results.extend(self.storage_tiers.scan(start, end).await?);
        
        Ok(results)
    }
}

// Performance:
// - Writes: <1μs (PMem)
// - Recent queries: <10μs (PMem)
// - Historical queries: <1ms (NVMe)
```

### Use Case 4: Distributed File System

```rust
pub struct DistributedFs {
    metadata_workers: WorkerPool<MetadataWorker>,
    data_workers: StorageWorkerPool,
    replication_manager: ReplicationManager,
}

impl DistributedFs {
    pub async fn write_file(&self, path: &str, data: &[u8]) -> Result<()> {
        // Split into chunks
        const CHUNK_SIZE: usize = 1024 * 1024;
        let chunks: Vec<_> = data.chunks(CHUNK_SIZE).collect();
        
        // Write chunks in parallel
        let mut chunk_handles = Vec::new();
        
        for (i, chunk) in chunks.iter().enumerate() {
            // Replicate each chunk
            let replicas = self.replication_manager
                .select_replicas(path, i, 3)?;
            
            for replica in replicas {
                let handle = replica.write_chunk(path, i, chunk);
                chunk_handles.push(handle);
            }
        }
        
        // Wait for all replicas
        futures::future::join_all(chunk_handles).await;
        
        // Update metadata
        self.metadata_workers.update_file_metadata(path, chunks.len()).await?;
        
        Ok(())
    }
}
```

---

## Summary

### Architecture Benefits

✅ **Unified Design**: Storage treated like any other protocol  
✅ **Performance**: Dedicated workers, zero-copy, batching  
✅ **Flexibility**: Mix protocols, transports, backends  
✅ **Scalability**: Independent scaling of I/O types  
✅ **Isolation**: Failures contained to I/O workers  
✅ **Composability**: Tiering, replication, caching  

### Implementation Priority

**Phase 1** (Month 1-2):
- Storage protocol trait definitions
- io_uring storage transport
- Basic block storage protocol
- Integration with worker pool

**Phase 2** (Month 3-4):
- SPDK transport and protocol
- Key-value over block storage
- Caching and batching
- Storage buffer pooling

**Phase 3** (Month 5-6):
- DAX/PMem support
- Object storage protocol
- Tiered storage
- Prefetching and optimization

**Phase 4** (Month 7-8):
- NVMe-oF remote storage
- Replication with consensus
- Advanced caching strategies
- Production hardening

---

**Document Version**: 1.0  
**Last Updated**: October 31, 2025  
**Status**: Design Complete - Ready for Implementation  
**Integration**: Extends PROTOCOL_LAYER.md with storage protocols

