//! Error types for the shared-nothing library

use std::fmt;

/// Result type alias for shared-nothing operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types that can occur in shared-nothing operations
#[derive(Debug, Clone)]
pub enum Error {
    /// Worker is not running
    WorkerNotRunning,
    
    /// Worker is already running
    WorkerAlreadyRunning,
    
    /// Channel send error
    SendError(String),
    
    /// Channel receive error
    ReceiveError(String),
    
    /// Worker panicked
    WorkerPanicked(String),
    
    /// Invalid configuration
    InvalidConfig(String),
    
    /// Worker pool is full
    PoolFull,
    
    /// Worker not found
    WorkerNotFound(u64),
    
    /// Partitioning error
    PartitionError(String),
    
    /// Timeout
    Timeout,
    
    /// Other error
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::WorkerNotRunning => write!(f, "Worker is not running"),
            Error::WorkerAlreadyRunning => write!(f, "Worker is already running"),
            Error::SendError(msg) => write!(f, "Channel send error: {}", msg),
            Error::ReceiveError(msg) => write!(f, "Channel receive error: {}", msg),
            Error::WorkerPanicked(msg) => write!(f, "Worker panicked: {}", msg),
            Error::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            Error::PoolFull => write!(f, "Worker pool is full"),
            Error::WorkerNotFound(id) => write!(f, "Worker not found: {}", id),
            Error::PartitionError(msg) => write!(f, "Partitioning error: {}", msg),
            Error::Timeout => write!(f, "Operation timed out"),
            Error::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

impl std::error::Error for Error {}

impl<T> From<flume::SendError<T>> for Error {
    fn from(err: flume::SendError<T>) -> Self {
        Error::SendError(err.to_string())
    }
}

impl From<flume::RecvError> for Error {
    fn from(err: flume::RecvError) -> Self {
        Error::ReceiveError(err.to_string())
    }
}

impl<T> From<crossbeam::channel::SendError<T>> for Error {
    fn from(err: crossbeam::channel::SendError<T>) -> Self {
        Error::SendError(err.to_string())
    }
}

impl From<crossbeam::channel::RecvError> for Error {
    fn from(err: crossbeam::channel::RecvError) -> Self {
        Error::ReceiveError(err.to_string())
    }
}

