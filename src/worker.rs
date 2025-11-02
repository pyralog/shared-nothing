//! Worker abstraction for shared-nothing architecture
//!
//! Workers are isolated execution units with their own state and message queue.
//! They communicate only through message passing, never sharing memory.

use crate::channel::{Channel, Sender};
use crate::error::{Error, Result};
use crate::message::{ControlMessage, Envelope};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Unique identifier for a worker
pub type WorkerId = u64;

/// Global worker ID counter
static WORKER_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Worker configuration
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Worker name (for debugging/monitoring)
    pub name: Option<String>,
    
    /// CPU core to pin this worker to (None = no pinning)
    pub cpu_affinity: Option<usize>,
    
    /// Message queue capacity
    pub queue_capacity: usize,
    
    /// Whether to use thread priority
    pub high_priority: bool,
    
    /// Stack size for worker thread (None = default)
    pub stack_size: Option<usize>,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            name: None,
            cpu_affinity: None,
            queue_capacity: 1024,
            high_priority: false,
            stack_size: None,
        }
    }
}

impl WorkerConfig {
    /// Create a new worker configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set the worker name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    
    /// Set CPU affinity
    pub fn with_cpu_affinity(mut self, cpu: usize) -> Self {
        self.cpu_affinity = Some(cpu);
        self
    }
    
    /// Set queue capacity
    pub fn with_queue_capacity(mut self, capacity: usize) -> Self {
        self.queue_capacity = capacity;
        self
    }
    
    /// Set high priority
    pub fn with_high_priority(mut self, high: bool) -> Self {
        self.high_priority = high;
        self
    }
    
    /// Set stack size
    pub fn with_stack_size(mut self, size: usize) -> Self {
        self.stack_size = Some(size);
        self
    }
}

/// Worker state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    /// Worker is not started
    Idle,
    
    /// Worker is running
    Running,
    
    /// Worker is paused
    Paused,
    
    /// Worker is stopped
    Stopped,
}

/// Worker handle for managing a running worker
pub struct WorkerHandle<S, M> {
    /// Unique worker ID
    id: WorkerId,
    
    /// Worker configuration
    config: WorkerConfig,
    
    /// Sender for messages to the worker
    message_tx: Sender<Envelope<M>>,
    
    /// Sender for control messages
    control_tx: Sender<ControlMessage>,
    
    /// Thread handle
    thread_handle: Option<JoinHandle<Result<()>>>,
    
    /// Running state
    running: Arc<AtomicBool>,
    
    /// Phantom data for state type
    _phantom: std::marker::PhantomData<S>,
}

impl<S, M> WorkerHandle<S, M>
where
    S: Send + 'static,
    M: Send + 'static,
{
    /// Get the worker ID
    pub fn id(&self) -> WorkerId {
        self.id
    }
    
    /// Get the worker name
    pub fn name(&self) -> Option<&str> {
        self.config.name.as_deref()
    }
    
    /// Send a message to the worker
    pub fn send(&self, message: M) -> Result<()> {
        let envelope = Envelope::new(message)
            .with_target(self.id);
        self.message_tx.send(envelope)
    }
    
    /// Send a message envelope to the worker
    pub fn send_envelope(&self, envelope: Envelope<M>) -> Result<()> {
        self.message_tx.send(envelope)
    }
    
    /// Send a control message to the worker
    pub fn send_control(&self, control: ControlMessage) -> Result<()> {
        self.control_tx.send(control)
    }
    
    /// Check if the worker is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }
    
    /// Stop the worker gracefully
    pub fn stop(&mut self) -> Result<()> {
        if !self.is_running() {
            return Err(Error::WorkerNotRunning);
        }
        
        self.send_control(ControlMessage::Stop)?;
        
        // Wait for worker to finish
        if let Some(handle) = self.thread_handle.take() {
            handle.join()
                .map_err(|_| Error::WorkerPanicked("Worker thread panicked".to_string()))??;
        }
        
        Ok(())
    }
    
    /// Pause the worker
    pub fn pause(&self) -> Result<()> {
        if !self.is_running() {
            return Err(Error::WorkerNotRunning);
        }
        self.send_control(ControlMessage::Pause)
    }
    
    /// Resume the worker
    pub fn resume(&self) -> Result<()> {
        if !self.is_running() {
            return Err(Error::WorkerNotRunning);
        }
        self.send_control(ControlMessage::Resume)
    }
    
    /// Ping the worker for health check
    pub fn ping(&self) -> Result<()> {
        if !self.is_running() {
            return Err(Error::WorkerNotRunning);
        }
        self.send_control(ControlMessage::Ping)
    }
    
    /// Get a cloned sender for sending messages
    pub fn sender(&self) -> Sender<Envelope<M>> {
        self.message_tx.clone()
    }
}

/// Trait for worker message handlers
pub trait Worker: Send + Sized + 'static {
    /// The type of state this worker maintains
    type State: Send + 'static;
    
    /// The type of messages this worker processes
    type Message: Send + 'static;
    
    /// Initialize the worker state
    fn init(&mut self) -> Result<Self::State>;
    
    /// Handle an incoming message
    fn handle_message(&mut self, state: &mut Self::State, message: Envelope<Self::Message>) -> Result<()>;
    
    /// Called when the worker is about to shutdown (optional)
    fn shutdown(&mut self, _state: Self::State) -> Result<()> {
        Ok(())
    }
    
    /// Called on each iteration (optional)
    fn tick(&mut self, _state: &mut Self::State) -> Result<()> {
        Ok(())
    }
}

/// Spawn a worker with the given configuration
pub fn spawn<W>(mut worker: W, config: WorkerConfig) -> Result<WorkerHandle<W::State, W::Message>>
where
    W: Worker,
{
    let id = WORKER_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    
    // Create message channel
    let (message_tx, message_rx) = Channel::mpsc(config.queue_capacity);
    
    // Create control channel
    let (control_tx, control_rx) = Channel::mpsc(16);
    
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = Arc::clone(&running);
    
    let worker_config = config.clone();
    
    // Build thread
    let mut thread_builder = thread::Builder::new();
    
    if let Some(name) = &config.name {
        thread_builder = thread_builder.name(format!("worker-{}-{}", id, name));
    } else {
        thread_builder = thread_builder.name(format!("worker-{}", id));
    }
    
    if let Some(stack_size) = config.stack_size {
        thread_builder = thread_builder.stack_size(stack_size);
    }
    
    // Spawn worker thread
    let thread_handle = thread_builder.spawn(move || {
        // Set CPU affinity if specified
        if let Some(cpu) = worker_config.cpu_affinity {
            if let Some(core_ids) = core_affinity::get_core_ids() {
                if cpu < core_ids.len() {
                    core_affinity::set_for_current(core_ids[cpu]);
                }
            }
        }
        
        // Initialize worker state
        let mut state = worker.init()?;
        
        let mut paused = false;
        
        // Main worker loop
        while running_clone.load(Ordering::Acquire) {
            // Check for control messages
            match control_rx.try_recv() {
                Ok(ControlMessage::Stop) => {
                    running_clone.store(false, Ordering::Release);
                    break;
                }
                Ok(ControlMessage::Pause) => {
                    paused = true;
                }
                Ok(ControlMessage::Resume) => {
                    paused = false;
                }
                Ok(ControlMessage::Ping) => {
                    // Could send pong back if we had a response channel
                }
                Ok(ControlMessage::Pong) => {}
                Err(_) => {}
            }
            
            if paused {
                thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }
            
            // Process messages
            match message_rx.try_recv() {
                Ok(envelope) => {
                    if let Err(e) = worker.handle_message(&mut state, envelope) {
                        eprintln!("Worker {} error handling message: {}", id, e);
                    }
                }
                Err(_) => {
                    // No message available, call tick
                    if let Err(e) = worker.tick(&mut state) {
                        eprintln!("Worker {} error in tick: {}", id, e);
                    }
                    
                    // Small sleep to avoid busy waiting
                    thread::sleep(std::time::Duration::from_micros(100));
                }
            }
        }
        
        // Shutdown
        worker.shutdown(state)?;
        
        Ok(())
    }).map_err(|e| Error::Other(format!("Failed to spawn worker thread: {}", e)))?;
    
    Ok(WorkerHandle {
        id,
        config,
        message_tx,
        control_tx,
        thread_handle: Some(thread_handle),
        running,
        _phantom: std::marker::PhantomData,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    struct CounterWorker;
    
    impl Worker for CounterWorker {
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
    
    #[test]
    fn test_worker_spawn() {
        let worker = CounterWorker;
        let config = WorkerConfig::new().with_name("test-worker");
        
        let mut handle = spawn(worker, config).unwrap();
        
        assert!(handle.is_running());
        
        handle.send(1).unwrap();
        handle.send(2).unwrap();
        handle.send(3).unwrap();
        
        thread::sleep(std::time::Duration::from_millis(100));
        
        handle.stop().unwrap();
        
        assert!(!handle.is_running());
    }
}

