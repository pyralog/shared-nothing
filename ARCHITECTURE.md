# Architecture Design Document

## Overview

The shared-nothing architecture library is designed to provide maximum performance for concurrent workloads by eliminating all shared state between workers. This document describes the architectural decisions and performance optimizations.

## Core Design Principles

### 1. Zero Sharing

**Principle**: Workers never share memory, preventing contention and data races.

**Implementation**:
- Each worker runs in its own thread with isolated state
- State types must be `Send + 'static` but not `Sync`
- Communication happens only through message passing
- No `Arc<Mutex<T>>` or other shared memory primitives

**Benefits**:
- Zero lock contention
- Perfect cache locality
- Linear scalability
- Fault isolation

### 2. Lock-Free Message Passing

**Principle**: Use lock-free data structures for inter-worker communication.

**Implementation**:
- Built on `flume` and `crossbeam` channels
- SPSC, MPSC, and MPMC variants
- Bounded and unbounded options
- Cache-line aligned for performance

**Performance Characteristics**:
```
SPSC: ~10ns per message (single core)
MPSC: ~30ns per message (multi-core)
MPMC: ~50ns per message (multi-core)
```

### 3. Cache Optimization

**Principle**: Minimize cache coherency traffic between cores.

**Implementation**:
- Cache-line padding (64 bytes) for shared structures
- Worker state aligned to cache lines
- Statistics counters use separate cache lines
- CPU affinity to keep workers on same core

**Memory Layout**:
```text
┌────────────────────────────────────────────┐  Cache Line 0
│  Worker State (exclusive to core)          │
├────────────────────────────────────────────┤  Cache Line 1
│  Padding (prevents false sharing)          │
├────────────────────────────────────────────┤  Cache Line 2
│  Channel metadata (shared)                 │
└────────────────────────────────────────────┘
```

### 4. Data Partitioning

**Principle**: Distribute work evenly while maintaining key affinity.

**Strategies**:

#### Hash Partitioning
- **Use Case**: General purpose, consistent mapping
- **Algorithm**: `hash(key) % num_workers`
- **Pros**: Simple, fast, uniform distribution
- **Cons**: All keys reassigned if workers change

#### Consistent Hashing
- **Use Case**: Dynamic worker pools
- **Algorithm**: Virtual nodes on hash ring
- **Pros**: Minimal redistribution on worker changes
- **Cons**: Slightly more overhead

#### Range Partitioning
- **Use Case**: Ordered data, range queries
- **Algorithm**: Map ranges to workers
- **Pros**: Locality for range operations
- **Cons**: Can create hot spots

#### Round Robin
- **Use Case**: Load balancing without affinity
- **Algorithm**: Sequential distribution
- **Pros**: Perfect balance
- **Cons**: No key affinity

## Component Architecture

### Worker Lifecycle

```text
┌─────────┐     spawn()     ┌─────────────┐
│ Factory │ ───────────────>│  Thread     │
└─────────┘                 │  Spawn      │
                            └──────┬──────┘
                                   │
                            ┌──────▼──────┐
                            │   init()    │
                            │  (Setup)    │
                            └──────┬──────┘
                                   │
                            ┌──────▼──────┐
                            │ Message     │
                            │ Loop        │<───┐
                            └──────┬──────┘    │
                                   │           │
                            ┌──────▼──────┐    │
                            │ handle_     │    │
                            │ message()   │────┘
                            └──────┬──────┘
                                   │
                            ┌──────▼──────┐
                            │ shutdown()  │
                            └──────┬──────┘
                                   │
                            ┌──────▼──────┐
                            │ Thread      │
                            │ Join        │
                            └─────────────┘
```

### Message Flow

```text
Client
  │
  │ send_partitioned(key, msg)
  ▼
PartitionerMessageRouter
  │
  │ partition(key) → worker_id
  ▼
Channel (Lock-Free)
  │
  │ Bounded queue (cache-aligned)
  ▼
Worker Thread
  │
  │ recv() → Message
  ▼
Message Handler
  │
  │ Process with isolated state
  ▼
[Optional: Send to other workers]
```

### Memory Model

#### Thread-Local State
Each worker has exclusive ownership of its state:

```rust
struct WorkerState {
    // All fields are owned, no Arc/Mutex needed
    data: HashMap<K, V>,
    counters: Vec<u64>,
    cache: LruCache<K, V>,
}
```

#### Message Passing
Messages are moved (not cloned) when possible:

```rust
// Message is moved into channel
tx.send(expensive_message)?;

// Receiver takes ownership
let msg = rx.recv()?;
```

#### Shared Statistics
Read-only or atomic-only access:

```rust
#[repr(align(64))]
struct ChannelStats {
    messages_sent: AtomicU64,    // Atomic updates
    _padding: [u8; 56],          // Prevent false sharing
}
```

## Performance Optimizations

### 1. Channel Selection

| Scenario | Channel Type | Reason |
|----------|-------------|---------|
| Single sender, single receiver | SPSC | Fastest, no contention |
| Multiple senders, single receiver | MPSC | Common pattern, optimized |
| Multiple senders, multiple receivers | MPMC | Most flexible |

### 2. CPU Affinity

Pin workers to specific CPU cores:

```rust
WorkerConfig::new()
    .with_cpu_affinity(core_id)
```

**Benefits**:
- Warmer caches
- Reduced context switching
- Predictable performance
- Better NUMA locality

### 3. Batch Processing

Process multiple messages per iteration:

```rust
fn handle_batch(&mut self, state: &mut State) -> Result<()> {
    let mut batch = Vec::with_capacity(100);
    
    // Drain available messages
    while let Ok(msg) = self.rx.try_recv() {
        batch.push(msg);
        if batch.len() >= 100 { break; }
    }
    
    // Process as batch
    self.process_batch(state, batch)
}
```

### 4. Zero-Copy Techniques

- Use `&[u8]` for large data with `Bytes` crate
- Pass ownership instead of cloning
- Use `MaybeUninit` for uninitialized buffers
- Memory-map files for large datasets

### 5. Profiling Hooks

Built-in statistics for monitoring:

```rust
let stats = channel.stats();
println!("Sent: {}, Received: {}", 
    stats.sent(), 
    stats.received()
);
```

## Scalability Analysis

### Vertical Scaling (Single Machine)

**Linear until**: Number of workers = physical cores

**Bottlenecks**:
- Memory bandwidth (>16 cores)
- Cache coherency (>32 cores)
- NUMA effects (>64 cores)

**Mitigation**:
- Use CPU affinity
- NUMA-aware allocation
- Minimize cross-core communication

### Horizontal Scaling (Multiple Machines)

The library provides building blocks for distributed systems:

1. **Serialize messages** (with `serialization` feature)
2. **Network workers** handle socket I/O
3. **Partitioning** extends across machines
4. **Consistent hashing** handles machine failures

## Error Handling

### Error Types

```rust
pub enum Error {
    WorkerNotRunning,        // Worker lifecycle
    WorkerAlreadyRunning,
    WorkerPanicked(String),
    
    SendError(String),       // Channel errors
    ReceiveError(String),
    Timeout,
    
    InvalidConfig(String),   // Configuration
    PoolFull,
    WorkerNotFound(u64),
    
    PartitionError(String),  // Partitioning
    Other(String),           // Catch-all
}
```

### Fault Isolation

Workers are isolated:
- Panic in one worker doesn't affect others
- Channel disconnection is handled gracefully
- Pool continues with remaining workers

### Recovery Strategies

1. **Retry**: Resend message to same worker
2. **Failover**: Send to different worker
3. **Restart**: Spawn new worker
4. **Circuit Breaker**: Stop sending after N failures

## Testing Strategy

### Unit Tests
- Individual components in isolation
- Property-based testing with `proptest`
- Edge cases and error conditions

### Integration Tests
- Multi-worker scenarios
- Message ordering guarantees
- Shutdown sequences

### Benchmarks
- Throughput measurements
- Latency percentiles (p50, p99, p999)
- Comparison with alternatives
- Scaling characteristics

### Stress Tests
- Long-running scenarios
- High message rates
- Memory leak detection
- Thread safety verification

## Future Enhancements

### Planned Features

1. **Async/Await Support**
   - Tokio integration
   - Async message handlers
   - Async I/O workers

2. **Network Distribution**
   - TCP/UDP transport
   - Protocol buffers
   - Service discovery

3. **Monitoring**
   - Prometheus metrics
   - Distributed tracing
   - Health checks

4. **Advanced Partitioning**
   - Weighted partitioning
   - Geo-aware routing
   - Priority queues

5. **Persistence**
   - Message durability
   - State checkpointing
   - Recovery from crashes

### Research Areas

- **SIMD Processing**: Vectorized message processing
- **GPU Offload**: Move compute to GPU workers
- **RDMA**: Zero-copy network transfers
- **eBPF**: Kernel-level message routing

## Conclusion

This architecture achieves high performance through:

1. **Elimination of locks**: Lock-free data structures
2. **Cache optimization**: Alignment and affinity
3. **Data locality**: Partitioning strategies
4. **Zero sharing**: Complete worker isolation
5. **Efficient messaging**: Optimized channels

The result is a library that scales linearly with cores and provides predictable, low-latency performance for concurrent workloads.

## References

- [Mechanical Sympathy Blog](https://mechanical-sympathy.blogspot.com/)
- [Preshing on Programming](https://preshing.com/)
- [The Art of Multiprocessor Programming](https://www.elsevier.com/books/the-art-of-multiprocessor-programming/herlihy/978-0-12-415950-1)
- [Rust Atomics and Locks](https://marabos.nl/atomics/)
- [Lock-Free Programming](https://www.1024cores.net/home/lock-free-algorithms)

