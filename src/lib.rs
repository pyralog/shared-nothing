//! # Shared-Nothing Architecture Library
//!
//! A high-performance shared-nothing architecture library for Rust that provides
//! isolated workers, lock-free message passing, and efficient data partitioning.
//!
//! ## Key Features
//!
//! - **Zero-sharing by design**: Each worker has its own isolated state
//! - **Lock-free message passing**: High-throughput channels with minimal contention
//! - **Cache-optimized**: Proper alignment and padding to prevent false sharing
//! - **Multiple channel types**: SPSC, MPSC, and MPMC for different use cases
//! - **Data partitioning**: Built-in strategies for distributing work
//! - **CPU affinity**: Optional pinning of workers to specific cores
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────┐      Messages      ┌─────────────┐
//! │  Worker 1   │ ──────────────────> │  Worker 2   │
//! │  (Isolated) │                     │  (Isolated) │
//! │   Memory    │ <────────────────── │   Memory    │
//! └─────────────┘      Messages      └─────────────┘
//!       │                                    │
//!       │                                    │
//!       ▼                                    ▼
//! ┌─────────────┐                    ┌─────────────┐
//! │   Local     │                    │   Local     │
//! │    State    │                    │    State    │
//! └─────────────┘                    └─────────────┘
//! ```

#![warn(missing_docs, rust_2018_idioms)]
#![allow(dead_code)]

pub mod worker;
pub mod channel;
pub mod partition;
pub mod pool;
pub mod message;
pub mod error;

// Re-exports
pub use worker::{Worker, WorkerConfig, WorkerId};
pub use channel::{Sender, Receiver, Channel, ChannelConfig};
pub use partition::{Partitioner, PartitionerExt, HashPartitioner, RangePartitioner, ConsistentHashPartitioner};
pub use pool::{WorkerPool, PoolConfig};
pub use message::Message;
pub use error::{Error, Result};

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::worker::{Worker, WorkerConfig, WorkerId};
    pub use crate::channel::{Sender, Receiver, Channel};
    pub use crate::partition::{Partitioner, PartitionerExt};
    pub use crate::pool::WorkerPool;
    pub use crate::message::{Message, Envelope};
    pub use crate::error::{Error, Result};
}

