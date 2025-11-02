//! Basic worker example demonstrating the core concepts

use shared_nothing::prelude::*;
use std::thread;
use std::time::Duration;

/// A simple counter worker that accumulates values
struct CounterWorker;

impl Worker for CounterWorker {
    type State = u64;
    type Message = u64;
    
    fn init(&mut self) -> Result<Self::State> {
        println!("CounterWorker initialized");
        Ok(0)
    }
    
    fn handle_message(&mut self, state: &mut Self::State, message: Envelope<Self::Message>) -> Result<()> {
        *state += message.payload;
        println!("CounterWorker received: {}, total: {}", message.payload, *state);
        Ok(())
    }
    
    fn shutdown(&mut self, state: Self::State) -> Result<()> {
        println!("CounterWorker shutting down. Final count: {}", state);
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("=== Basic Worker Example ===\n");
    
    // Create worker configuration
    let config = WorkerConfig::new()
        .with_name("counter")
        .with_queue_capacity(100);
    
    // Spawn the worker
    let mut worker = shared_nothing::worker::spawn(CounterWorker, config)?;
    
    println!("Worker spawned with ID: {}\n", worker.id());
    
    // Send some messages
    println!("Sending messages...");
    for i in 1..=10 {
        worker.send(i)?;
        thread::sleep(Duration::from_millis(50));
    }
    
    // Give worker time to process
    thread::sleep(Duration::from_millis(500));
    
    println!("\nStopping worker...");
    worker.stop()?;
    
    println!("Worker stopped successfully!");
    
    Ok(())
}

