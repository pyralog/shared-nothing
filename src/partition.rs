//! Data partitioning strategies for distributing work across workers
//!
//! Partitioning is crucial in shared-nothing architectures to distribute
//! work evenly and minimize contention.

use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};
use ahash::AHasher;

/// Trait for partitioning strategies
pub trait Partitioner: Send + Sync {
    /// Determine which worker should handle this key
    /// Takes a hash value instead of a generic key for dyn compatibility
    fn partition_hash(&self, hash: u64, num_workers: usize) -> usize;
}

/// Helper to compute hash for a key
pub fn compute_hash<K: Hash>(key: &K) -> u64 {
    let mut hasher = AHasher::default();
    key.hash(&mut hasher);
    hasher.finish()
}

/// Extension trait that provides the generic partition method
pub trait PartitionerExt: Partitioner {
    /// Determine which worker should handle this key
    fn partition<K: Hash>(&self, key: &K, num_workers: usize) -> usize {
        let hash = compute_hash(key);
        self.partition_hash(hash, num_workers)
    }
}

// Blanket implementation for all Partitioner implementors
impl<T: Partitioner + ?Sized> PartitionerExt for T {}

/// Simple hash-based partitioner using modulo operation
#[derive(Debug, Clone, Default)]
pub struct HashPartitioner;

impl HashPartitioner {
    /// Create a new hash partitioner
    pub fn new() -> Self {
        Self
    }
}

impl Partitioner for HashPartitioner {
    fn partition_hash(&self, hash: u64, num_workers: usize) -> usize {
        if num_workers == 0 {
            return 0;
        }
        
        (hash % num_workers as u64) as usize
    }
}

/// Range-based partitioner for ordered data
#[derive(Debug, Clone)]
pub struct RangePartitioner<K> {
    /// Range boundaries
    ranges: Vec<K>,
}

impl<K: Ord + Clone> RangePartitioner<K> {
    /// Create a new range partitioner with the given boundaries
    /// 
    /// For n workers, you need n-1 boundaries.
    /// For example, with boundaries [10, 20] and 3 workers:
    /// - Worker 0: keys < 10
    /// - Worker 1: keys >= 10 and < 20
    /// - Worker 2: keys >= 20
    pub fn new(ranges: Vec<K>) -> Self {
        Self { ranges }
    }
}

impl<K: Ord + Clone + Send + Sync> Partitioner for RangePartitioner<K> {
    fn partition_hash(&self, hash: u64, num_workers: usize) -> usize {
        // For hash-based calls, use modulo
        if num_workers == 0 {
            return 0;
        }
        (hash % num_workers as u64) as usize
    }
}

impl<K: Ord + Clone> RangePartitioner<K> {
    /// Partition a key that matches the range type
    pub fn partition_range(&self, key: &K, num_workers: usize) -> usize {
        if num_workers == 0 {
            return 0;
        }
        
        for (idx, boundary) in self.ranges.iter().enumerate() {
            if key < boundary {
                return idx;
            }
        }
        
        // Key is greater than all boundaries
        self.ranges.len()
    }
}

/// Consistent hashing partitioner for minimal redistribution when workers change
/// 
/// This is useful when the number of workers can change dynamically,
/// as it minimizes the number of keys that need to be redistributed.
#[derive(Debug, Clone)]
pub struct ConsistentHashPartitioner {
    /// Number of virtual nodes per worker
    virtual_nodes: usize,
    
    /// Ring of virtual nodes (hash -> worker_id)
    ring: BTreeMap<u64, usize>,
}

impl ConsistentHashPartitioner {
    /// Create a new consistent hash partitioner
    /// 
    /// # Arguments
    /// * `num_workers` - Number of workers in the system
    /// * `virtual_nodes` - Number of virtual nodes per worker (higher = better distribution)
    pub fn new(num_workers: usize, virtual_nodes: usize) -> Self {
        let mut ring = BTreeMap::new();
        
        for worker_id in 0..num_workers {
            for vnode in 0..virtual_nodes {
                let mut hasher = AHasher::default();
                hasher.write_usize(worker_id);
                hasher.write_usize(vnode);
                let hash = hasher.finish();
                ring.insert(hash, worker_id);
            }
        }
        
        Self {
            virtual_nodes,
            ring,
        }
    }
    
    /// Add a new worker to the ring
    pub fn add_worker(&mut self, worker_id: usize) {
        for vnode in 0..self.virtual_nodes {
            let mut hasher = AHasher::default();
            hasher.write_usize(worker_id);
            hasher.write_usize(vnode);
            let hash = hasher.finish();
            self.ring.insert(hash, worker_id);
        }
    }
    
    /// Remove a worker from the ring
    pub fn remove_worker(&mut self, worker_id: usize) {
        let mut keys_to_remove = Vec::new();
        
        for (&hash, &id) in &self.ring {
            if id == worker_id {
                keys_to_remove.push(hash);
            }
        }
        
        for key in keys_to_remove {
            self.ring.remove(&key);
        }
    }
    
    /// Get the number of workers in the ring
    pub fn num_workers(&self) -> usize {
        if self.virtual_nodes == 0 {
            0
        } else {
            self.ring.len() / self.virtual_nodes
        }
    }
}

impl Partitioner for ConsistentHashPartitioner {
    fn partition_hash(&self, hash: u64, _num_workers: usize) -> usize {
        if self.ring.is_empty() {
            return 0;
        }
        
        // Find the first node in the ring with hash >= key hash
        for (_node_hash, &worker_id) in self.ring.range(hash..) {
            return worker_id;
        }
        
        // Wrap around to the first node
        *self.ring.values().next().unwrap_or(&0)
    }
}

/// Round-robin partitioner for simple load distribution
#[derive(Debug)]
pub struct RoundRobinPartitioner {
    counter: std::sync::atomic::AtomicUsize,
}

impl Default for RoundRobinPartitioner {
    fn default() -> Self {
        Self::new()
    }
}

impl RoundRobinPartitioner {
    /// Create a new round-robin partitioner
    pub fn new() -> Self {
        Self {
            counter: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl Partitioner for RoundRobinPartitioner {
    fn partition_hash(&self, _hash: u64, num_workers: usize) -> usize {
        if num_workers == 0 {
            return 0;
        }
        
        let count = self.counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        count % num_workers
    }
}

/// Custom partitioner that allows user-defined logic
pub struct CustomPartitioner<F> {
    func: F,
}

impl<F> CustomPartitioner<F>
where
    F: Fn(u64, usize) -> usize + Send + Sync,
{
    /// Create a new custom partitioner with the given function
    pub fn new(func: F) -> Self {
        Self { func }
    }
}

impl<F> Partitioner for CustomPartitioner<F>
where
    F: Fn(u64, usize) -> usize + Send + Sync,
{
    fn partition_hash(&self, hash: u64, num_workers: usize) -> usize {
        (self.func)(hash, num_workers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hash_partitioner() {
        let partitioner = HashPartitioner::new();
        let num_workers = 4;
        
        // Same key should always map to same worker
        let key = "test_key";
        let worker1 = partitioner.partition(&key, num_workers);
        let worker2 = partitioner.partition(&key, num_workers);
        assert_eq!(worker1, worker2);
        
        // Worker ID should be in valid range
        assert!(worker1 < num_workers);
    }
    
    #[test]
    fn test_consistent_hash_partitioner() {
        let mut partitioner = ConsistentHashPartitioner::new(4, 100);
        
        let key = "test_key";
        let worker1 = partitioner.partition(&key, 4);
        
        // Add a new worker
        partitioner.add_worker(4);
        let worker2 = partitioner.partition(&key, 5);
        
        // Most keys should stay on the same worker
        // (we can't guarantee this specific key stays, but the principle holds)
        assert!(worker1 < 4);
        assert!(worker2 < 5);
    }
    
    #[test]
    fn test_round_robin_partitioner() {
        let partitioner = RoundRobinPartitioner::new();
        let num_workers = 3;
        
        let key1 = "key1";
        let key2 = "key2";
        let key3 = "key3";
        let key4 = "key4";
        
        let w1 = partitioner.partition(&key1, num_workers);
        let w2 = partitioner.partition(&key2, num_workers);
        let w3 = partitioner.partition(&key3, num_workers);
        let w4 = partitioner.partition(&key4, num_workers);
        
        // Should cycle through workers
        assert_eq!(w1, 0);
        assert_eq!(w2, 1);
        assert_eq!(w3, 2);
        assert_eq!(w4, 0);
    }
    
    #[test]
    fn test_custom_partitioner() {
        // Always assign to worker 0
        let partitioner = CustomPartitioner::new(|_hash, _num_workers| 0);
        
        let key = "test";
        assert_eq!(partitioner.partition(&key, 4), 0);
    }
}

