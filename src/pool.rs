//! Worker pool management for shared-nothing architecture
//!
//! The pool manages multiple workers and provides routing of messages
//! based on partitioning strategies.

use crate::channel::Sender;
use crate::error::{Error, Result};
use crate::message::Envelope;
use crate::partition::{HashPartitioner, Partitioner, PartitionerExt};
use crate::worker::{Worker, WorkerConfig, WorkerHandle, WorkerId, spawn};
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use parking_lot::RwLock;

/// Worker pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Number of workers in the pool
    pub num_workers: usize,
    
    /// Configuration template for workers
    pub worker_config: WorkerConfig,
    
    /// Whether to enable CPU affinity pinning
    pub enable_cpu_affinity: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get(),
            worker_config: WorkerConfig::default(),
            enable_cpu_affinity: false,
        }
    }
}

impl PoolConfig {
    /// Create a new pool configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the number of workers
    pub fn with_num_workers(mut self, num: usize) -> Self {
        self.num_workers = num;
        self
    }
    
    /// Set the worker configuration template
    pub fn with_worker_config(mut self, config: WorkerConfig) -> Self {
        self.worker_config = config;
        self
    }
    
    /// Enable CPU affinity pinning
    pub fn with_cpu_affinity(mut self, enable: bool) -> Self {
        self.enable_cpu_affinity = enable;
        self
    }
}

/// A pool of workers that share no state
pub struct WorkerPool<S, M>
where
    S: Send + 'static,
    M: Send + 'static,
{
    /// Pool configuration
    config: PoolConfig,
    
    /// Worker handles indexed by worker ID
    workers: Arc<RwLock<HashMap<WorkerId, WorkerHandle<S, M>>>>,
    
    /// Partitioner for routing messages
    partitioner: Arc<dyn Partitioner>,
    
    /// Quick access to worker IDs in order
    worker_ids: Arc<RwLock<Vec<WorkerId>>>,
}

impl<S, M> WorkerPool<S, M>
where
    S: Send + 'static,
    M: Send + 'static,
{
    /// Create a new worker pool with a factory function
    pub fn new<W, F>(config: PoolConfig, factory: F) -> Result<Self>
    where
        W: Worker<State = S, Message = M>,
        F: FnMut(usize) -> W,
    {
        Self::with_partitioner(config, factory, Arc::new(HashPartitioner::new()))
    }
    
    /// Create a new worker pool with a custom partitioner
    pub fn with_partitioner<W, F>(
        config: PoolConfig,
        mut factory: F,
        partitioner: Arc<dyn Partitioner>,
    ) -> Result<Self>
    where
        W: Worker<State = S, Message = M>,
        F: FnMut(usize) -> W,
    {
        let mut workers = HashMap::new();
        let mut worker_ids = Vec::new();
        
        for i in 0..config.num_workers {
            let mut worker_config = config.worker_config.clone();
            
            // Set CPU affinity if enabled
            if config.enable_cpu_affinity {
                worker_config.cpu_affinity = Some(i % num_cpus::get());
            }
            
            // Set worker name
            if worker_config.name.is_none() {
                worker_config.name = Some(format!("pool-worker-{}", i));
            }
            
            let worker = factory(i);
            let handle = spawn(worker, worker_config)?;
            let worker_id = handle.id();
            
            worker_ids.push(worker_id);
            workers.insert(worker_id, handle);
        }
        
        Ok(Self {
            config,
            workers: Arc::new(RwLock::new(workers)),
            partitioner,
            worker_ids: Arc::new(RwLock::new(worker_ids)),
        })
    }
    
    /// Send a message to a specific worker by ID
    pub fn send_to_worker(&self, worker_id: WorkerId, message: M) -> Result<()> {
        let workers = self.workers.read();
        let worker = workers
            .get(&worker_id)
            .ok_or(Error::WorkerNotFound(worker_id))?;
        
        worker.send(message)
    }
    
    /// Send a message to a worker determined by partitioning the key
    pub fn send_partitioned<K: Hash>(&self, key: &K, message: M) -> Result<()> {
        let worker_ids = self.worker_ids.read();
        let num_workers = worker_ids.len();
        
        if num_workers == 0 {
            return Err(Error::Other("No workers in pool".to_string()));
        }
        
        let partition = self.partitioner.partition(key, num_workers);
        let worker_id = worker_ids[partition];
        
        self.send_to_worker(worker_id, message)
    }
    
    /// Broadcast a message to all workers
    pub fn broadcast(&self, message: M) -> Result<()>
    where
        M: Clone,
    {
        let workers = self.workers.read();
        let mut last_error = None;
        
        for (worker_id, worker) in workers.iter() {
            if let Err(e) = worker.send(message.clone()) {
                eprintln!("Failed to send to worker {}: {}", worker_id, e);
                last_error = Some(e);
            }
        }
        
        if let Some(err) = last_error {
            Err(err)
        } else {
            Ok(())
        }
    }
    
    /// Get the number of workers in the pool
    pub fn num_workers(&self) -> usize {
        self.worker_ids.read().len()
    }
    
    /// Get a list of all worker IDs
    pub fn worker_ids(&self) -> Vec<WorkerId> {
        self.worker_ids.read().clone()
    }
    
    /// Get a sender for a specific worker
    pub fn get_sender(&self, worker_id: WorkerId) -> Result<Sender<Envelope<M>>> {
        let workers = self.workers.read();
        let worker = workers
            .get(&worker_id)
            .ok_or(Error::WorkerNotFound(worker_id))?;
        
        Ok(worker.sender())
    }
    
    /// Stop all workers gracefully
    pub fn stop_all(&mut self) -> Result<()> {
        let mut workers = self.workers.write();
        let mut errors = Vec::new();
        
        for (worker_id, worker) in workers.iter_mut() {
            if let Err(e) = worker.stop() {
                eprintln!("Failed to stop worker {}: {}", worker_id, e);
                errors.push(e);
            }
        }
        
        workers.clear();
        self.worker_ids.write().clear();
        
        if !errors.is_empty() {
            Err(errors.into_iter().next().unwrap())
        } else {
            Ok(())
        }
    }
    
    /// Check if all workers are running
    pub fn all_running(&self) -> bool {
        let workers = self.workers.read();
        workers.values().all(|w| w.is_running())
    }
}

impl<S, M> Drop for WorkerPool<S, M>
where
    S: Send + 'static,
    M: Send + 'static,
{
    fn drop(&mut self) {
        // Best effort to stop all workers
        let _ = self.stop_all();
    }
}

/// Helper to get the number of CPUs
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::Worker;
    
    struct TestWorker {
        id: usize,
    }
    
    impl Worker for TestWorker {
        type State = Vec<u64>;
        type Message = u64;
        
        fn init(&mut self) -> Result<Self::State> {
            Ok(Vec::new())
        }
        
        fn handle_message(
            &mut self,
            state: &mut Self::State,
            message: Envelope<Self::Message>,
        ) -> Result<()> {
            state.push(message.payload);
            Ok(())
        }
    }
    
    #[test]
    fn test_worker_pool_creation() {
        let config = PoolConfig::new().with_num_workers(4);
        
        let pool = WorkerPool::new(config, |i| TestWorker { id: i });
        
        assert!(pool.is_ok());
        let mut pool = pool.unwrap();
        assert_eq!(pool.num_workers(), 4);
        assert!(pool.all_running());
        
        pool.stop_all().unwrap();
    }
    
    #[test]
    fn test_partitioned_send() {
        let config = PoolConfig::new().with_num_workers(4);
        
        let mut pool = WorkerPool::new(config, |i| TestWorker { id: i }).unwrap();
        
        // Send messages with different keys
        for i in 0..100 {
            pool.send_partitioned(&i, i as u64).unwrap();
        }
        
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        pool.stop_all().unwrap();
    }
}

