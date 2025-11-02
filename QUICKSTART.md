# Quick Start Guide

Get up and running with shared-nothing in 5 minutes.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
shared-nothing = "0.1"
```

Or use cargo:

```bash
cargo add shared-nothing
```

## Basic Usage

### 1. Simple Worker

```rust
use shared_nothing::prelude::*;

// Define your worker
struct MyWorker;

impl Worker for MyWorker {
    type State = u64;        // Worker's state type
    type Message = String;   // Message type it processes
    
    fn init(&mut self) -> Result<Self::State> {
        Ok(0)  // Initialize state
    }
    
    fn handle_message(&mut self, state: &mut Self::State, msg: Envelope<Self::Message>) -> Result<()> {
        println!("Received: {}", msg.payload);
        *state += 1;
        Ok(())
    }
}

fn main() -> Result<()> {
    // Create and spawn worker
    let mut worker = shared_nothing::worker::spawn(
        MyWorker,
        WorkerConfig::new().with_name("my-worker")
    )?;
    
    // Send messages
    worker.send("Hello".to_string())?;
    worker.send("World".to_string())?;
    
    // Stop worker
    std::thread::sleep(std::time::Duration::from_millis(100));
    worker.stop()?;
    
    Ok(())
}
```

### 2. Worker Pool

```rust
use shared_nothing::prelude::*;

struct DataWorker { id: usize }

impl Worker for DataWorker {
    type State = Vec<i32>;
    type Message = i32;
    
    fn init(&mut self) -> Result<Self::State> {
        println!("Worker {} starting", self.id);
        Ok(Vec::new())
    }
    
    fn handle_message(&mut self, state: &mut Self::State, msg: Envelope<Self::Message>) -> Result<()> {
        state.push(msg.payload);
        Ok(())
    }
}

fn main() -> Result<()> {
    // Create pool with 4 workers
    let mut pool = shared_nothing::pool::WorkerPool::new(
        shared_nothing::pool::PoolConfig::new().with_num_workers(4),
        |i| DataWorker { id: i }
    )?;
    
    // Send messages (automatically partitioned)
    for i in 0..100 {
        pool.send_partitioned(&i, i)?;
    }
    
    std::thread::sleep(std::time::Duration::from_millis(100));
    pool.stop_all()?;
    
    Ok(())
}
```

### 3. Custom Partitioning

```rust
use shared_nothing::prelude::*;
use shared_nothing::partition::ConsistentHashPartitioner;
use std::sync::Arc;

fn main() -> Result<()> {
    // Use consistent hashing
    let pool = shared_nothing::pool::WorkerPool::with_partitioner(
        shared_nothing::pool::PoolConfig::new().with_num_workers(4),
        |i| DataWorker { id: i },
        Arc::new(ConsistentHashPartitioner::new(4, 100))  // 100 virtual nodes
    )?;
    
    // Messages with same key go to same worker
    pool.send_partitioned(&"user-123", 42)?;
    pool.send_partitioned(&"user-123", 43)?;  // Same worker as above
    
    Ok(())
}
```

## Common Patterns

### Pattern 1: Pipeline

```rust
// Worker 1: Read data
// Worker 2: Process data
// Worker 3: Write results

let (tx1, rx1) = Channel::mpsc(100);
let (tx2, rx2) = Channel::mpsc(100);

struct ReadWorker { tx: Sender<Envelope<Data>> }
struct ProcessWorker { tx: Sender<Envelope<Result>> }
struct WriteWorker { }

// Chain workers together
```

### Pattern 2: Fan-Out / Fan-In

```rust
// One coordinator sends to many workers
// Workers send results back to collector

let pool = WorkerPool::new(config, factory)?;

// Fan-out
for item in data {
    pool.send_partitioned(&item.key, item)?;
}

// Fan-in: workers send to collector channel
```

### Pattern 3: Request-Response

```rust
struct Request {
    id: u64,
    data: Vec<u8>,
    response_tx: Sender<Response>,
}

// Send request with return channel
worker.send(Request {
    id: 1,
    data: vec![1, 2, 3],
    response_tx: my_response_channel.clone(),
})?;

// Worker processes and sends response back
```

## Configuration

### Worker Configuration

```rust
let config = WorkerConfig::new()
    .with_name("worker-1")           // Name for debugging
    .with_queue_capacity(1024)       // Message queue size
    .with_cpu_affinity(2)            // Pin to CPU core 2
    .with_stack_size(2 * 1024 * 1024); // 2MB stack
```

### Pool Configuration

```rust
let config = PoolConfig::new()
    .with_num_workers(8)             // Number of workers
    .with_cpu_affinity(true)         // Enable CPU pinning
    .with_worker_config(               // Template for workers
        WorkerConfig::new()
            .with_queue_capacity(1024)
    );
```

### Channel Configuration

```rust
// Bounded channel
let (tx, rx) = Channel::mpsc(1024);

// Unbounded channel
let (tx, rx) = Channel::mpsc_unbounded();

// With timeout
let (tx, rx) = Channel::new(
    ChannelConfig::new()
        .with_capacity(1024)
        .with_timeout(Duration::from_secs(5)),
    ChannelType::MPSC
);
```

## Monitoring

```rust
// Get channel statistics
let stats = tx.stats();
println!("Messages sent: {}", stats.sent());
println!("Messages received: {}", stats.received());
println!("Errors: {}", stats.send_errors());

// Check worker status
if worker.is_running() {
    println!("Worker is running");
}

// Pool statistics
println!("Active workers: {}", pool.num_workers());
println!("All running: {}", pool.all_running());
```

## Error Handling

```rust
match worker.send(message) {
    Ok(_) => println!("Sent successfully"),
    Err(Error::SendError(msg)) => eprintln!("Send failed: {}", msg),
    Err(Error::WorkerNotRunning) => eprintln!("Worker stopped"),
    Err(e) => eprintln!("Other error: {}", e),
}
```

## Performance Tips

### 1. Choose Right Channel Type

```rust
// Fastest: Single producer, single consumer
let (tx, rx) = Channel::spsc(1024);

// Common: Multiple producers, single consumer
let (tx, rx) = Channel::mpsc(1024);

// Flexible: Multiple producers, multiple consumers
let (tx, rx) = Channel::mpmc(1024);
```

### 2. Enable CPU Affinity

```rust
let config = PoolConfig::new()
    .with_num_workers(4)
    .with_cpu_affinity(true);  // 10-30% performance boost
```

### 3. Batch Processing

```rust
impl Worker for MyWorker {
    fn tick(&mut self, state: &mut State) -> Result<()> {
        // Process multiple messages at once
        let mut batch = Vec::with_capacity(100);
        while let Ok(msg) = self.try_recv() {
            batch.push(msg);
            if batch.len() >= 100 { break; }
        }
        self.process_batch(state, batch)
    }
}
```

### 4. Keep Messages Small

```rust
// Good: Small message
#[derive(Clone, Copy)]
struct Message {
    id: u64,
    value: i32,
}

// Better for large data: Use indirection
struct LargeMessage {
    data: Box<Vec<u8>>,  // Heap allocation
}
```

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_worker() {
        let mut worker = spawn(MyWorker, WorkerConfig::new()).unwrap();
        
        worker.send(42).unwrap();
        std::thread::sleep(Duration::from_millis(50));
        
        worker.stop().unwrap();
    }
}
```

## Debugging

```rust
// Enable debug logging
env_logger::init();

// Named workers for easier debugging
WorkerConfig::new().with_name("debug-worker")

// Add debug output in worker
fn handle_message(&mut self, state: &mut State, msg: Envelope<Message>) -> Result<()> {
    eprintln!("Worker received: {:?}", msg.payload);
    // ... process message
    Ok(())
}
```

## Common Issues

### Issue: Worker Not Processing Messages

**Solution**: Make sure worker is running and messages are being sent correctly
```rust
assert!(worker.is_running());
worker.send(msg)?;
thread::sleep(Duration::from_millis(10)); // Give time to process
```

### Issue: Channel Full

**Solution**: Increase capacity or use unbounded channel
```rust
// Option 1: Increase capacity
WorkerConfig::new().with_queue_capacity(10000)

// Option 2: Use unbounded
let (tx, rx) = Channel::mpsc_unbounded();
```

### Issue: High Latency

**Solution**: Reduce queue size and enable CPU affinity
```rust
WorkerConfig::new()
    .with_queue_capacity(64)  // Smaller queue
    .with_cpu_affinity(0)     // Pin to core
```

## Next Steps

1. **Read the docs**: `cargo doc --open`
2. **Run examples**: `cargo run --example basic_worker`
3. **Check benchmarks**: `cargo bench`
4. **Read guides**:
   - [Architecture Guide](ARCHITECTURE.md)
   - [Performance Guide](PERFORMANCE.md)
   - [Full README](README.md)

## Getting Help

- **Documentation**: Run `cargo doc --open`
- **Examples**: Check `examples/` directory
- **Issues**: Open an issue on GitHub
- **Performance**: See PERFORMANCE.md

---

**Happy coding with shared-nothing!** 🚀

