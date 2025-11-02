//! Benchmarks for message passing performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use shared_nothing::channel::{Channel, ChannelType, ChannelConfig};
use std::thread;

fn bench_channel_send_recv(c: &mut Criterion) {
    let mut group = c.benchmark_group("channel_send_recv");
    
    for size in [10, 100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let (tx, rx) = Channel::mpsc::<i64>(size);
                
                thread::scope(|s| {
                    s.spawn(|| {
                        for i in 0..size {
                            tx.send(black_box(i as i64)).unwrap();
                        }
                    });
                    
                    s.spawn(|| {
                        for _ in 0..size {
                            black_box(rx.recv().unwrap());
                        }
                    });
                });
            });
        });
    }
    
    group.finish();
}

fn bench_channel_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("channel_types");
    let size = 1000;
    
    group.throughput(Throughput::Elements(size as u64));
    
    for channel_type in [ChannelType::SPSC, ChannelType::MPSC, ChannelType::MPMC].iter() {
        group.bench_with_input(
            BenchmarkId::new("type", format!("{:?}", channel_type)),
            channel_type,
            |b, &channel_type| {
                b.iter(|| {
                    let (tx, rx) = Channel::new::<i64>(
                        ChannelConfig::new().with_capacity(size),
                        channel_type,
                    );
                    
                    thread::scope(|s| {
                        s.spawn(|| {
                            for i in 0..size {
                                tx.send(black_box(i as i64)).unwrap();
                            }
                        });
                        
                        s.spawn(|| {
                            for _ in 0..size {
                                black_box(rx.recv().unwrap());
                            }
                        });
                    });
                });
            },
        );
    }
    
    group.finish();
}

fn bench_bounded_vs_unbounded(c: &mut Criterion) {
    let mut group = c.benchmark_group("bounded_vs_unbounded");
    let size = 1000;
    
    group.throughput(Throughput::Elements(size as u64));
    
    group.bench_function("bounded", |b| {
        b.iter(|| {
            let (tx, rx) = Channel::mpsc::<i64>(size);
            
            thread::scope(|s| {
                s.spawn(|| {
                    for i in 0..size {
                        tx.send(black_box(i as i64)).unwrap();
                    }
                });
                
                s.spawn(|| {
                    for _ in 0..size {
                        black_box(rx.recv().unwrap());
                    }
                });
            });
        });
    });
    
    group.bench_function("unbounded", |b| {
        b.iter(|| {
            let (tx, rx) = Channel::mpsc_unbounded::<i64>();
            
            thread::scope(|s| {
                s.spawn(|| {
                    for i in 0..size {
                        tx.send(black_box(i as i64)).unwrap();
                    }
                });
                
                s.spawn(|| {
                    for _ in 0..size {
                        black_box(rx.recv().unwrap());
                    }
                });
            });
        });
    });
    
    group.finish();
}

fn bench_multi_producer(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_producer");
    let size = 1000;
    
    for num_producers in [1, 2, 4, 8].iter() {
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::from_parameter(num_producers),
            num_producers,
            |b, &num_producers| {
                b.iter(|| {
                    let (tx, rx) = Channel::mpsc::<i64>(size * num_producers);
                    let messages_per_producer = size / num_producers;
                    
                    thread::scope(|s| {
                        for _ in 0..num_producers {
                            let tx_clone = tx.clone();
                            s.spawn(move || {
                                for i in 0..messages_per_producer {
                                    tx_clone.send(black_box(i as i64)).unwrap();
                                }
                            });
                        }
                        
                        s.spawn(move || {
                            for _ in 0..size {
                                black_box(rx.recv().unwrap());
                            }
                        });
                    });
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_channel_send_recv,
    bench_channel_types,
    bench_bounded_vs_unbounded,
    bench_multi_producer
);

criterion_main!(benches);

