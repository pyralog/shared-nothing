//! Data processing example using worker pool with partitioning

use shared_nothing::prelude::*;
use shared_nothing::partition::HashPartitioner;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Data processing message
#[derive(Clone, Debug)]
struct DataItem {
    key: String,
    value: i64,
}

/// Worker that processes data items
struct DataProcessor {
    worker_id: usize,
}

impl Worker for DataProcessor {
    type State = Vec<DataItem>;
    type Message = DataItem;
    
    fn init(&mut self) -> Result<Self::State> {
        println!("DataProcessor {} initialized", self.worker_id);
        Ok(Vec::new())
    }
    
    fn handle_message(&mut self, state: &mut Self::State, message: Envelope<Self::Message>) -> Result<()> {
        let item = message.payload;
        println!(
            "Worker {} processing: key={}, value={}",
            self.worker_id, item.key, item.value
        );
        
        // Simulate some processing
        thread::sleep(Duration::from_millis(10));
        
        state.push(item);
        Ok(())
    }
    
    fn shutdown(&mut self, state: Self::State) -> Result<()> {
        println!(
            "DataProcessor {} shutting down. Processed {} items",
            self.worker_id,
            state.len()
        );
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("=== Data Processing with Worker Pool ===\n");
    
    // Configure the worker pool
    let pool_config = shared_nothing::pool::PoolConfig::new()
        .with_num_workers(4)
        .with_worker_config(
            WorkerConfig::new()
                .with_queue_capacity(100)
        );
    
    // Create the worker pool
    let mut pool = shared_nothing::pool::WorkerPool::with_partitioner(
        pool_config,
        |i| DataProcessor { worker_id: i },
        Arc::new(HashPartitioner::new()),
    )?;
    
    println!("Created pool with {} workers\n", pool.num_workers());
    
    // Generate and send data items
    println!("Sending data items...");
    for i in 0..20 {
        let item = DataItem {
            key: format!("key-{}", i),
            value: i * 10,
        };
        
        // Messages with the same key go to the same worker
        pool.send_partitioned(&item.key, item)?;
    }
    
    // Give workers time to process
    thread::sleep(Duration::from_millis(500));
    
    println!("\nStopping all workers...");
    pool.stop_all()?;
    
    println!("All workers stopped successfully!");
    
    Ok(())
}

