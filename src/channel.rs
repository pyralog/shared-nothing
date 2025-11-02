//! High-performance message channels with cache-line optimization
//!
//! This module provides multiple channel implementations optimized for
//! different use cases in a shared-nothing architecture.

use crate::error::{Error, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Cache line size for padding (typically 64 bytes on x86-64)
const CACHE_LINE_SIZE: usize = 64;

/// Channel configuration
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    /// Buffer capacity
    pub capacity: usize,
    
    /// Whether to use bounded or unbounded channels
    pub bounded: bool,
    
    /// Timeout for send/receive operations (None = no timeout)
    pub timeout: Option<Duration>,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        Self {
            capacity: 1024,
            bounded: true,
            timeout: None,
        }
    }
}

impl ChannelConfig {
    /// Create a new channel configuration with default values
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the capacity
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity;
        self
    }
    
    /// Set whether the channel is bounded
    pub fn with_bounded(mut self, bounded: bool) -> Self {
        self.bounded = bounded;
        self
    }
    
    /// Set the timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
}

/// Channel type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelType {
    /// Single Producer Single Consumer (fastest)
    SPSC,
    
    /// Multiple Producer Single Consumer
    MPSC,
    
    /// Multiple Producer Multiple Consumer
    MPMC,
}

/// Statistics for channel performance monitoring
#[repr(align(64))] // Align to cache line
#[derive(Debug)]
pub struct ChannelStats {
    /// Number of messages sent
    pub messages_sent: AtomicU64,
    
    /// Number of messages received
    pub messages_received: AtomicU64,
    
    /// Number of send errors
    pub send_errors: AtomicU64,
    
    /// Number of receive errors
    pub recv_errors: AtomicU64,
    
    _padding: [u8; CACHE_LINE_SIZE - 32], // Pad to cache line
}

impl Default for ChannelStats {
    fn default() -> Self {
        Self {
            messages_sent: AtomicU64::new(0),
            messages_received: AtomicU64::new(0),
            send_errors: AtomicU64::new(0),
            recv_errors: AtomicU64::new(0),
            _padding: [0; CACHE_LINE_SIZE - 32],
        }
    }
}

impl ChannelStats {
    /// Get the number of messages sent
    pub fn sent(&self) -> u64 {
        self.messages_sent.load(Ordering::Relaxed)
    }
    
    /// Get the number of messages received
    pub fn received(&self) -> u64 {
        self.messages_received.load(Ordering::Relaxed)
    }
    
    /// Get the number of send errors
    pub fn send_errors(&self) -> u64 {
        self.send_errors.load(Ordering::Relaxed)
    }
    
    /// Get the number of receive errors
    pub fn recv_errors(&self) -> u64 {
        self.recv_errors.load(Ordering::Relaxed)
    }
    
    /// Reset all statistics
    pub fn reset(&self) {
        self.messages_sent.store(0, Ordering::Relaxed);
        self.messages_received.store(0, Ordering::Relaxed);
        self.send_errors.store(0, Ordering::Relaxed);
        self.recv_errors.store(0, Ordering::Relaxed);
    }
}

/// Sender half of a channel
pub struct Sender<T> {
    inner: SenderInner<T>,
    stats: Arc<ChannelStats>,
    timeout: Option<Duration>,
}

enum SenderInner<T> {
    Flume(flume::Sender<T>),
    Crossbeam(crossbeam::channel::Sender<T>),
}

impl<T> Clone for Sender<T> {
    fn clone(&self) -> Self {
        Self {
            inner: match &self.inner {
                SenderInner::Flume(s) => SenderInner::Flume(s.clone()),
                SenderInner::Crossbeam(s) => SenderInner::Crossbeam(s.clone()),
            },
            stats: Arc::clone(&self.stats),
            timeout: self.timeout,
        }
    }
}

impl<T> Sender<T> {
    /// Send a message through the channel
    pub fn send(&self, msg: T) -> Result<()> {
        let result = match &self.inner {
            SenderInner::Flume(s) => {
                if let Some(timeout) = self.timeout {
                    s.send_timeout(msg, timeout).map_err(|e| match e {
                        flume::SendTimeoutError::Timeout(_) => Error::Timeout,
                        flume::SendTimeoutError::Disconnected(_) => {
                            Error::SendError("Channel disconnected".to_string())
                        }
                    })
                } else {
                    s.send(msg).map_err(|e| Error::SendError(e.to_string()))
                }
            }
            SenderInner::Crossbeam(s) => {
                if let Some(timeout) = self.timeout {
                    s.send_timeout(msg, timeout).map_err(|e| match e {
                        crossbeam::channel::SendTimeoutError::Timeout(_) => Error::Timeout,
                        crossbeam::channel::SendTimeoutError::Disconnected(_) => {
                            Error::SendError("Channel disconnected".to_string())
                        }
                    })
                } else {
                    s.send(msg).map_err(|e| Error::SendError(e.to_string()))
                }
            }
        };
        
        match result {
            Ok(_) => {
                self.stats.messages_sent.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                self.stats.send_errors.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }
    
    /// Try to send a message without blocking
    pub fn try_send(&self, msg: T) -> Result<()> {
        let result = match &self.inner {
            SenderInner::Flume(s) => {
                s.try_send(msg).map_err(|e| match e {
                    flume::TrySendError::Full(_) => Error::SendError("Channel full".to_string()),
                    flume::TrySendError::Disconnected(_) => {
                        Error::SendError("Channel disconnected".to_string())
                    }
                })
            }
            SenderInner::Crossbeam(s) => {
                s.try_send(msg).map_err(|e| match e {
                    crossbeam::channel::TrySendError::Full(_) => {
                        Error::SendError("Channel full".to_string())
                    }
                    crossbeam::channel::TrySendError::Disconnected(_) => {
                        Error::SendError("Channel disconnected".to_string())
                    }
                })
            }
        };
        
        match result {
            Ok(_) => {
                self.stats.messages_sent.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            Err(e) => {
                self.stats.send_errors.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }
    
    /// Get channel statistics
    pub fn stats(&self) -> Arc<ChannelStats> {
        Arc::clone(&self.stats)
    }
}

/// Receiver half of a channel
pub struct Receiver<T> {
    inner: ReceiverInner<T>,
    stats: Arc<ChannelStats>,
    timeout: Option<Duration>,
}

enum ReceiverInner<T> {
    Flume(flume::Receiver<T>),
    Crossbeam(crossbeam::channel::Receiver<T>),
}

impl<T> Clone for Receiver<T> {
    fn clone(&self) -> Self {
        Self {
            inner: match &self.inner {
                ReceiverInner::Flume(r) => ReceiverInner::Flume(r.clone()),
                ReceiverInner::Crossbeam(r) => ReceiverInner::Crossbeam(r.clone()),
            },
            stats: Arc::clone(&self.stats),
            timeout: self.timeout,
        }
    }
}

impl<T> Receiver<T> {
    /// Receive a message from the channel
    pub fn recv(&self) -> Result<T> {
        let result = match &self.inner {
            ReceiverInner::Flume(r) => {
                if let Some(timeout) = self.timeout {
                    r.recv_timeout(timeout).map_err(|e| match e {
                        flume::RecvTimeoutError::Timeout => Error::Timeout,
                        flume::RecvTimeoutError::Disconnected => {
                            Error::ReceiveError("Channel disconnected".to_string())
                        }
                    })
                } else {
                    r.recv().map_err(|e| Error::ReceiveError(e.to_string()))
                }
            }
            ReceiverInner::Crossbeam(r) => {
                if let Some(timeout) = self.timeout {
                    r.recv_timeout(timeout).map_err(|e| match e {
                        crossbeam::channel::RecvTimeoutError::Timeout => Error::Timeout,
                        crossbeam::channel::RecvTimeoutError::Disconnected => {
                            Error::ReceiveError("Channel disconnected".to_string())
                        }
                    })
                } else {
                    r.recv().map_err(|e| Error::ReceiveError(e.to_string()))
                }
            }
        };
        
        match result {
            Ok(msg) => {
                self.stats.messages_received.fetch_add(1, Ordering::Relaxed);
                Ok(msg)
            }
            Err(e) => {
                self.stats.recv_errors.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }
    
    /// Try to receive a message without blocking
    pub fn try_recv(&self) -> Result<T> {
        let result = match &self.inner {
            ReceiverInner::Flume(r) => {
                r.try_recv().map_err(|e| match e {
                    flume::TryRecvError::Empty => Error::ReceiveError("Channel empty".to_string()),
                    flume::TryRecvError::Disconnected => {
                        Error::ReceiveError("Channel disconnected".to_string())
                    }
                })
            }
            ReceiverInner::Crossbeam(r) => {
                r.try_recv().map_err(|e| match e {
                    crossbeam::channel::TryRecvError::Empty => {
                        Error::ReceiveError("Channel empty".to_string())
                    }
                    crossbeam::channel::TryRecvError::Disconnected => {
                        Error::ReceiveError("Channel disconnected".to_string())
                    }
                })
            }
        };
        
        match result {
            Ok(msg) => {
                self.stats.messages_received.fetch_add(1, Ordering::Relaxed);
                Ok(msg)
            }
            Err(e) => {
                self.stats.recv_errors.fetch_add(1, Ordering::Relaxed);
                Err(e)
            }
        }
    }
    
    /// Create an iterator over incoming messages
    pub fn iter(&self) -> ReceiverIterator<'_, T> {
        ReceiverIterator { receiver: self }
    }
    
    /// Get channel statistics
    pub fn stats(&self) -> Arc<ChannelStats> {
        Arc::clone(&self.stats)
    }
}

/// Iterator for receiving messages
pub struct ReceiverIterator<'a, T> {
    receiver: &'a Receiver<T>,
}

impl<'a, T> Iterator for ReceiverIterator<'a, T> {
    type Item = T;
    
    fn next(&mut self) -> Option<Self::Item> {
        self.receiver.recv().ok()
    }
}

/// Channel factory
pub struct Channel;

impl Channel {
    /// Create a new channel with the given configuration and type
    pub fn new<T>(config: ChannelConfig, channel_type: ChannelType) -> (Sender<T>, Receiver<T>) {
        let stats = Arc::new(ChannelStats::default());
        
        match channel_type {
            ChannelType::SPSC | ChannelType::MPSC | ChannelType::MPMC => {
                // Use flume for all types (it's highly optimized for all scenarios)
                let (tx, rx) = if config.bounded {
                    flume::bounded(config.capacity)
                } else {
                    flume::unbounded()
                };
                
                (
                    Sender {
                        inner: SenderInner::Flume(tx),
                        stats: Arc::clone(&stats),
                        timeout: config.timeout,
                    },
                    Receiver {
                        inner: ReceiverInner::Flume(rx),
                        stats,
                        timeout: config.timeout,
                    },
                )
            }
        }
    }
    
    /// Create a bounded MPSC channel (most common use case)
    pub fn mpsc<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
        Self::new(
            ChannelConfig::new().with_capacity(capacity),
            ChannelType::MPSC,
        )
    }
    
    /// Create an unbounded MPSC channel
    pub fn mpsc_unbounded<T>() -> (Sender<T>, Receiver<T>) {
        Self::new(
            ChannelConfig::new().with_bounded(false),
            ChannelType::MPSC,
        )
    }
    
    /// Create a bounded SPSC channel (for single producer/consumer)
    pub fn spsc<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
        Self::new(
            ChannelConfig::new().with_capacity(capacity),
            ChannelType::SPSC,
        )
    }
    
    /// Create a bounded MPMC channel
    pub fn mpmc<T>(capacity: usize) -> (Sender<T>, Receiver<T>) {
        Self::new(
            ChannelConfig::new().with_capacity(capacity),
            ChannelType::MPMC,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mpsc_channel() {
        let (tx, rx) = Channel::mpsc::<i32>(10);
        
        tx.send(42).unwrap();
        tx.send(43).unwrap();
        
        assert_eq!(rx.recv().unwrap(), 42);
        assert_eq!(rx.recv().unwrap(), 43);
        
        assert_eq!(tx.stats().sent(), 2);
        assert_eq!(rx.stats().received(), 2);
    }
    
    #[test]
    fn test_channel_stats() {
        let (tx, rx) = Channel::mpsc::<i32>(10);
        
        for i in 0..5 {
            tx.send(i).unwrap();
        }
        
        for _ in 0..5 {
            rx.recv().unwrap();
        }
        
        assert_eq!(tx.stats().sent(), 5);
        assert_eq!(rx.stats().received(), 5);
    }
}

