//! Benchmarks for worker pool performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use shared_nothing::prelude::*;
use shared_nothing::partition::{HashPartitioner, RoundRobinPartitioner};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

struct BenchWorker;

impl Worker for BenchWorker {
    type State = u64;
    type Message = u64;
    
    fn init(&mut self) -> Result<Self::State> {
        Ok(0)
    }
    
    fn handle_message(&mut self, state: &mut Self::State, message: Envelope<Self::Message>) -> Result<()> {
        *state += message.payload;
        Ok(())
    }
}

fn bench_pool_partitioned_send(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_partitioned_send");
    
    for num_workers in [2, 4, 8].iter() {
        let size = 1000;
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_workers),
            num_workers,
            |b, &num_workers| {
                let pool_config = shared_nothing::pool::PoolConfig::new()
                    .with_num_workers(num_workers)
                    .with_worker_config(WorkerConfig::new().with_queue_capacity(size));
                
                let mut pool = shared_nothing::pool::WorkerPool::new(
                    pool_config,
                    |_| BenchWorker,
                ).unwrap();
                
                b.iter(|| {
                    for i in 0..size {
                        pool.send_partitioned(&black_box(i), black_box(i as u64)).unwrap();
                    }
                });
                
                thread::sleep(Duration::from_millis(100));
                pool.stop_all().unwrap();
            },
        );
    }
    
    group.finish();
}

fn bench_pool_broadcast(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_broadcast");
    
    for num_workers in [2, 4, 8].iter() {
        group.throughput(Throughput::Elements(*num_workers as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_workers),
            num_workers,
            |b, &num_workers| {
                let pool_config = shared_nothing::pool::PoolConfig::new()
                    .with_num_workers(num_workers)
                    .with_worker_config(WorkerConfig::new());
                
                let mut pool = shared_nothing::pool::WorkerPool::new(
                    pool_config,
                    |_| BenchWorker,
                ).unwrap();
                
                b.iter(|| {
                    pool.broadcast(black_box(42)).unwrap();
                });
                
                thread::sleep(Duration::from_millis(100));
                pool.stop_all().unwrap();
            },
        );
    }
    
    group.finish();
}

fn bench_partitioner_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("partitioner_types");
    let size = 1000;
    
    group.throughput(Throughput::Elements(size as u64));
    
    // Hash partitioner
    group.bench_function("hash", |b| {
        let pool_config = shared_nothing::pool::PoolConfig::new()
            .with_num_workers(4)
            .with_worker_config(WorkerConfig::new().with_queue_capacity(size));
        
        let mut pool = shared_nothing::pool::WorkerPool::with_partitioner(
            pool_config,
            |_| BenchWorker,
            Arc::new(HashPartitioner::new()),
        ).unwrap();
        
        b.iter(|| {
            for i in 0..size {
                pool.send_partitioned(&black_box(i), black_box(i as u64)).unwrap();
            }
        });
        
        thread::sleep(Duration::from_millis(100));
        pool.stop_all().unwrap();
    });
    
    // Round-robin partitioner
    group.bench_function("round_robin", |b| {
        let pool_config = shared_nothing::pool::PoolConfig::new()
            .with_num_workers(4)
            .with_worker_config(WorkerConfig::new().with_queue_capacity(size));
        
        let mut pool = shared_nothing::pool::WorkerPool::with_partitioner(
            pool_config,
            |_| BenchWorker,
            Arc::new(RoundRobinPartitioner::new()),
        ).unwrap();
        
        b.iter(|| {
            for i in 0..size {
                pool.send_partitioned(&black_box(i), black_box(i as u64)).unwrap();
            }
        });
        
        thread::sleep(Duration::from_millis(100));
        pool.stop_all().unwrap();
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_pool_partitioned_send,
    bench_pool_broadcast,
    bench_partitioner_types
);

criterion_main!(benches);

