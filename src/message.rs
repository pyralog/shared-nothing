//! Message types and traits for inter-worker communication

use std::fmt::Debug;

#[cfg(feature = "serialization")]
use serde::{Serialize, Deserialize};

/// Trait for messages that can be sent between workers
/// 
/// Messages must be Send + 'static to ensure they can be safely
/// transferred between threads without sharing references.
pub trait Message: Send + 'static {}

// Blanket implementation for all types that meet the requirements
impl<T: Send + 'static> Message for T {}

/// A generic envelope for typed messages
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct Envelope<T> {
    /// The actual message payload
    pub payload: T,
    
    /// Optional message ID for tracking
    pub id: Option<u64>,
    
    /// Source worker ID
    pub source: Option<u64>,
    
    /// Target worker ID
    pub target: Option<u64>,
    
    /// Timestamp (in microseconds since epoch)
    pub timestamp: u64,
}

impl<T> Envelope<T> {
    /// Create a new envelope with the given payload
    pub fn new(payload: T) -> Self {
        Self {
            payload,
            id: None,
            source: None,
            target: None,
            timestamp: current_timestamp_micros(),
        }
    }
    
    /// Set the message ID
    pub fn with_id(mut self, id: u64) -> Self {
        self.id = Some(id);
        self
    }
    
    /// Set the source worker ID
    pub fn with_source(mut self, source: u64) -> Self {
        self.source = Some(source);
        self
    }
    
    /// Set the target worker ID
    pub fn with_target(mut self, target: u64) -> Self {
        self.target = Some(target);
        self
    }
}

/// Control messages for worker lifecycle management
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub enum ControlMessage {
    /// Stop the worker gracefully
    Stop,
    
    /// Pause the worker
    Pause,
    
    /// Resume a paused worker
    Resume,
    
    /// Ping for health check
    Ping,
    
    /// Pong response
    Pong,
}

/// Get current timestamp in microseconds
fn current_timestamp_micros() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_micros() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_envelope_creation() {
        let envelope = Envelope::new(42)
            .with_id(1)
            .with_source(100)
            .with_target(200);
        
        assert_eq!(envelope.payload, 42);
        assert_eq!(envelope.id, Some(1));
        assert_eq!(envelope.source, Some(100));
        assert_eq!(envelope.target, Some(200));
        assert!(envelope.timestamp > 0);
    }
}

