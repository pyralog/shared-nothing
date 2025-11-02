//! Distributed computation example using worker communication

use shared_nothing::prelude::*;
use shared_nothing::channel::Channel;
use std::thread;
use std::time::Duration;

/// Computation task
#[derive(Clone, Debug)]
struct Task {
    id: u64,
    data: Vec<i64>,
}

/// Computation result
#[derive(Clone, Debug)]
struct TaskResult {
    task_id: u64,
    sum: i64,
    count: usize,
    average: f64,
}

/// Worker that performs computations
struct ComputeWorker {
    worker_id: usize,
    result_sender: shared_nothing::channel::Sender<Envelope<TaskResult>>,
}

impl Worker for ComputeWorker {
    type State = usize; // Count of processed tasks
    type Message = Task;
    
    fn init(&mut self) -> Result<Self::State> {
        println!("ComputeWorker {} initialized", self.worker_id);
        Ok(0)
    }
    
    fn handle_message(&mut self, state: &mut Self::State, message: Envelope<Self::Message>) -> Result<()> {
        let task = message.payload;
        
        println!("Worker {} computing task {}", self.worker_id, task.id);
        
        // Perform computation
        let sum: i64 = task.data.iter().sum();
        let count = task.data.len();
        let average = if count > 0 {
            sum as f64 / count as f64
        } else {
            0.0
        };
        
        // Simulate computation time
        thread::sleep(Duration::from_millis(50));
        
        // Send result
        let result = TaskResult {
            task_id: task.id,
            sum,
            count,
            average,
        };
        
        self.result_sender.send(Envelope::new(result))
            .map_err(|e| shared_nothing::error::Error::Other(e.to_string()))?;
        
        *state += 1;
        Ok(())
    }
    
    fn shutdown(&mut self, state: Self::State) -> Result<()> {
        println!("ComputeWorker {} completed {} tasks", self.worker_id, state);
        Ok(())
    }
}

/// Result collector worker
struct ResultCollector;

impl Worker for ResultCollector {
    type State = Vec<TaskResult>;
    type Message = TaskResult;
    
    fn init(&mut self) -> Result<Self::State> {
        println!("ResultCollector initialized");
        Ok(Vec::new())
    }
    
    fn handle_message(&mut self, state: &mut Self::State, message: Envelope<Self::Message>) -> Result<()> {
        let result = message.payload;
        println!(
            "Result received: task_id={}, sum={}, avg={:.2}",
            result.task_id, result.sum, result.average
        );
        state.push(result);
        Ok(())
    }
    
    fn shutdown(&mut self, state: Self::State) -> Result<()> {
        println!("\nResultCollector summary:");
        println!("Total tasks completed: {}", state.len());
        let total_sum: i64 = state.iter().map(|r| r.sum).sum();
        println!("Grand total sum: {}", total_sum);
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("=== Distributed Computation Example ===\n");
    
    // Create result channel
    let (result_tx, result_rx) = Channel::mpmc(100);
    
    // Spawn result collector
    let collector_config = WorkerConfig::new().with_name("collector");
    let mut collector_handle = shared_nothing::worker::spawn(ResultCollector, collector_config)?;
    
    // Create compute worker pool
    let pool_config = shared_nothing::pool::PoolConfig::new()
        .with_num_workers(3)
        .with_worker_config(WorkerConfig::new().with_queue_capacity(50));
    
    let mut pool = shared_nothing::pool::WorkerPool::new(
        pool_config,
        |i| ComputeWorker {
            worker_id: i,
            result_sender: result_tx.clone(),
        },
    )?;
    
    println!("System initialized with {} compute workers\n", pool.num_workers());
    
    // Spawn result receiver thread
    let collector_sender = collector_handle.sender();
    thread::spawn(move || {
        while let Ok(result) = result_rx.recv() {
            let _ = collector_sender.send(result);
        }
    });
    
    // Generate and distribute tasks
    println!("Distributing tasks...");
    for i in 0..10 {
        let task = Task {
            id: i,
            data: (0..100).map(|x| x * (i as i64 + 1)).collect(),
        };
        
        pool.send_partitioned(&task.id, task)?;
    }
    
    // Wait for processing
    println!("\nProcessing...");
    thread::sleep(Duration::from_secs(2));
    
    // Shutdown
    println!("\nShutting down...");
    pool.stop_all()?;
    drop(result_tx); // Close the channel
    
    thread::sleep(Duration::from_millis(500));
    collector_handle.stop()?;
    
    println!("\nSystem shutdown complete!");
    
    Ok(())
}

