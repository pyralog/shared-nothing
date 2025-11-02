# Zero-Configuration Library Design

This document defines the zero-configuration approach combining **Smart Defaults with Auto-Detection** + **Builder Pattern** + **Profile-Based Presets** + **Runtime Adaptation** + **Feature Flag Defaults** + **Capability-Based Auto-Config**.

## Table of Contents

1. [Philosophy](#philosophy)
2. [Auto-Detection System](#auto-detection-system)
3. [Builder Pattern API](#builder-pattern-api)
4. [Profile-Based Presets](#profile-based-presets)
5. [Runtime Adaptation](#runtime-adaptation)
6. [Feature Flag Defaults](#feature-flag-defaults)
7. [Capability-Based Configuration](#capability-based-configuration)
8. [Usage Examples](#usage-examples)
9. [Migration Paths](#migration-paths)

---

## Philosophy

### Design Principles

1. **Zero Config by Default**: `WorkerPool::new()` just works
2. **Progressive Disclosure**: Expose complexity only when needed
3. **Smart Over Explicit**: Auto-detect rather than require configuration
4. **Safe Defaults**: Optimize for correctness, then performance
5. **Runtime Learning**: Adapt to actual workload characteristics
6. **Graceful Degradation**: Work even when optimal hardware unavailable

### The Gradient of Configuration

```rust
// Level 0: Zero config - auto-everything (80% of users)
let pool = WorkerPool::new();

// Level 1: Profile selection (15% of users)
let pool = WorkerPool::production();

// Level 2: Minimal config (4% of users)
let pool = WorkerPool::builder()
    .workers(8)
    .build();

// Level 3: Full control (1% of users)
let pool = WorkerPool::builder()
    .workers(16)
    .cpu_affinity(true)
    .numa_aware(true)
    .storage_backend(StorageBackend::Spdk { device: "0000:03:00.0" })
    .build();
```

---

## Auto-Detection System

### System Capability Discovery

```rust
/// Automatically detect all system capabilities
pub struct SystemCapabilities {
    cpu: CpuCapabilities,
    memory: MemoryCapabilities,
    storage: StorageCapabilities,
    network: NetworkCapabilities,
    accelerators: AcceleratorCapabilities,
}

impl SystemCapabilities {
    pub fn detect() -> Self {
        Self {
            cpu: CpuCapabilities::detect(),
            memory: MemoryCapabilities::detect(),
            storage: StorageCapabilities::detect(),
            network: NetworkCapabilities::detect(),
            accelerators: AcceleratorCapabilities::detect(),
        }
    }
}

/// CPU detection
pub struct CpuCapabilities {
    pub num_cores: usize,
    pub num_physical_cores: usize,
    pub num_sockets: usize,
    pub numa_nodes: Vec<NumaNode>,
    pub cache_line_size: usize,
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_aes_ni: bool,
}

impl CpuCapabilities {
    pub fn detect() -> Self {
        Self {
            num_cores: num_cpus::get(),
            num_physical_cores: num_cpus::get_physical(),
            num_sockets: Self::detect_sockets(),
            numa_nodes: Self::detect_numa(),
            cache_line_size: Self::detect_cache_line_size(),
            has_avx2: is_x86_feature_detected!("avx2"),
            has_avx512: is_x86_feature_detected!("avx512f"),
            has_aes_ni: is_x86_feature_detected!("aes"),
        }
    }
    
    fn detect_numa() -> Vec<NumaNode> {
        // Parse /sys/devices/system/node/node*/cpulist
        let mut nodes = Vec::new();
        
        for entry in std::fs::read_dir("/sys/devices/system/node")
            .ok()
            .into_iter()
            .flatten()
        {
            let path = entry.ok()?.path();
            if let Some(name) = path.file_name()?.to_str() {
                if name.starts_with("node") {
                    let node = NumaNode::from_sysfs(&path).ok()?;
                    nodes.push(node);
                }
            }
        }
        
        if nodes.is_empty() {
            // Single NUMA node fallback
            nodes.push(NumaNode::default());
        }
        
        nodes
    }
}

/// Memory detection
pub struct MemoryCapabilities {
    pub total_bytes: usize,
    pub available_bytes: usize,
    pub page_size: usize,
    pub huge_page_size: Option<usize>,
    pub has_dax: bool,
    pub has_pmem: bool,
}

impl MemoryCapabilities {
    pub fn detect() -> Self {
        Self {
            total_bytes: Self::get_total_memory(),
            available_bytes: Self::get_available_memory(),
            page_size: Self::get_page_size(),
            huge_page_size: Self::detect_huge_pages(),
            has_dax: Self::check_dax_support(),
            has_pmem: Self::check_pmem_devices(),
        }
    }
}

/// Storage detection
pub struct StorageCapabilities {
    pub has_spdk: bool,
    pub has_io_uring: bool,
    pub nvme_devices: Vec<NvmeDevice>,
    pub has_dax_filesystem: bool,
    pub has_pmem: bool,
}

impl StorageCapabilities {
    pub fn detect() -> Self {
        Self {
            has_spdk: Self::check_spdk_available(),
            has_io_uring: Self::check_io_uring_support(),
            nvme_devices: Self::enumerate_nvme_devices(),
            has_dax_filesystem: Self::check_dax_mounts(),
            has_pmem: Self::check_pmem_devices(),
        }
    }
    
    fn check_io_uring_support() -> bool {
        // Try to create io_uring instance
        io_uring::IoUring::new(2).is_ok()
    }
    
    fn enumerate_nvme_devices() -> Vec<NvmeDevice> {
        let mut devices = Vec::new();
        
        // Check /dev/nvme*
        if let Ok(entries) = std::fs::read_dir("/dev") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                if let Some(name_str) = name.to_str() {
                    if name_str.starts_with("nvme") && !name_str.contains('p') {
                        if let Ok(device) = NvmeDevice::open(&entry.path()) {
                            devices.push(device);
                        }
                    }
                }
            }
        }
        
        devices
    }
}

/// Network detection
pub struct NetworkCapabilities {
    pub has_dpdk: bool,
    pub has_af_xdp: bool,
    pub has_rdma: bool,
    pub interfaces: Vec<NetworkInterface>,
    pub max_speed_gbps: u64,
}

impl NetworkCapabilities {
    pub fn detect() -> Self {
        Self {
            has_dpdk: Self::check_dpdk_available(),
            has_af_xdp: Self::check_af_xdp_support(),
            has_rdma: Self::check_rdma_devices(),
            interfaces: Self::enumerate_interfaces(),
            max_speed_gbps: Self::get_max_interface_speed(),
        }
    }
}

/// Accelerator detection
pub struct AcceleratorCapabilities {
    pub gpus: Vec<GpuInfo>,
    pub has_qat: bool,
    pub has_dpu: bool,
    pub has_tpu: bool,
    pub has_fpga: bool,
}

impl AcceleratorCapabilities {
    pub fn detect() -> Self {
        Self {
            gpus: Self::detect_gpus(),
            has_qat: Self::check_qat_devices(),
            has_dpu: Self::check_dpu_devices(),
            has_tpu: Self::check_tpu_devices(),
            has_fpga: Self::check_fpga_devices(),
        }
    }
    
    fn detect_gpus() -> Vec<GpuInfo> {
        let mut gpus = Vec::new();
        
        // Try CUDA
        #[cfg(feature = "cuda")]
        if let Ok(cuda_gpus) = Self::detect_cuda_gpus() {
            gpus.extend(cuda_gpus);
        }
        
        // Try Metal (macOS)
        #[cfg(all(feature = "metal", target_os = "macos"))]
        if let Ok(metal_gpus) = Self::detect_metal_gpus() {
            gpus.extend(metal_gpus);
        }
        
        // Try Vulkan
        #[cfg(feature = "vulkan")]
        if gpus.is_empty() {
            if let Ok(vulkan_gpus) = Self::detect_vulkan_gpus() {
                gpus.extend(vulkan_gpus);
            }
        }
        
        // Try wgpu (always available)
        #[cfg(feature = "wgpu")]
        if gpus.is_empty() {
            if let Ok(wgpu_gpus) = Self::detect_wgpu_gpus() {
                gpus.extend(wgpu_gpus);
            }
        }
        
        gpus
    }
}
```

### Smart Default Calculation

```rust
/// Calculate optimal default configuration
pub struct DefaultConfig {
    capabilities: SystemCapabilities,
}

impl DefaultConfig {
    pub fn calculate() -> WorkerPoolConfig {
        let caps = SystemCapabilities::detect();
        
        WorkerPoolConfig {
            // Application workers: 50-70% of cores
            num_app_workers: Self::calculate_app_workers(&caps),
            
            // I/O workers: 10-20% of cores
            num_io_workers: Self::calculate_io_workers(&caps),
            
            // Accelerator workers: 5-10% of cores (if available)
            num_accelerator_workers: Self::calculate_accelerator_workers(&caps),
            
            // CPU affinity: enabled if >16 cores
            cpu_affinity: caps.cpu.num_cores > 16,
            
            // NUMA aware: enabled if multiple NUMA nodes
            numa_aware: caps.cpu.numa_nodes.len() > 1,
            
            // Channel size: based on memory
            channel_size: Self::calculate_channel_size(&caps),
            
            // Storage backend: best available
            storage_backend: Self::select_storage_backend(&caps),
            
            // Network backend: best available
            network_backend: Self::select_network_backend(&caps),
            
            // Accelerators: use what's available
            use_gpu: !caps.accelerators.gpus.is_empty(),
            use_qat: caps.accelerators.has_qat,
        }
    }
    
    fn calculate_app_workers(caps: &SystemCapabilities) -> usize {
        let total_cores = caps.cpu.num_physical_cores;
        
        // Reserve cores for I/O
        let reserved = (total_cores as f64 * 0.3).ceil() as usize;
        let app_cores = total_cores.saturating_sub(reserved);
        
        // At least 1, at most total-1
        app_cores.max(1).min(total_cores - 1)
    }
    
    fn calculate_io_workers(caps: &SystemCapabilities) -> usize {
        let total_cores = caps.cpu.num_physical_cores;
        
        // 2 workers per NUMA node, minimum 2
        let numa_based = caps.cpu.numa_nodes.len() * 2;
        
        // Or 20% of cores
        let percent_based = (total_cores as f64 * 0.2).ceil() as usize;
        
        numa_based.max(2).min(percent_based)
    }
    
    fn calculate_accelerator_workers(caps: &SystemCapabilities) -> usize {
        let num_gpus = caps.accelerators.gpus.len();
        
        if num_gpus > 0 {
            // 1-2 workers per GPU
            num_gpus.max(2)
        } else {
            0
        }
    }
    
    fn calculate_channel_size(caps: &SystemCapabilities) -> usize {
        let memory_gb = caps.memory.total_bytes / (1024 * 1024 * 1024);
        
        // Scale channel size with memory
        match memory_gb {
            0..=4 => 256,
            5..=16 => 1024,
            17..=64 => 4096,
            _ => 8192,
        }
    }
    
    fn select_storage_backend(caps: &StorageCapabilities) -> StorageBackend {
        // Priority: SPDK > io_uring > DAX > Standard
        
        if caps.has_spdk && !caps.nvme_devices.is_empty() {
            StorageBackend::Spdk {
                device: caps.nvme_devices[0].pci_address.clone(),
            }
        } else if caps.has_io_uring {
            StorageBackend::IoUring {
                path: "/tmp/shared-nothing.db".to_string(),
                queue_depth: 256,
            }
        } else if caps.has_dax_filesystem {
            StorageBackend::Dax {
                path: "/mnt/dax/shared-nothing.db".to_string(),
                size: 1024 * 1024 * 1024, // 1GB
            }
        } else {
            StorageBackend::Standard {
                path: "/tmp/shared-nothing.db".to_string(),
            }
        }
    }
    
    fn select_network_backend(caps: &NetworkCapabilities) -> NetworkBackend {
        // Priority: DPDK > io_uring > AF_XDP > Standard
        
        if caps.has_dpdk && caps.max_speed_gbps >= 10 {
            NetworkBackend::Dpdk
        } else if caps.has_af_xdp {
            NetworkBackend::AfXdp
        } else {
            NetworkBackend::IoUring
        }
    }
}
```

---

## Builder Pattern API

### Progressive Disclosure Builder

```rust
/// Main entry point - zero config
pub struct WorkerPool<W: Worker> {
    workers: Vec<WorkerHandle<W>>,
    config: WorkerPoolConfig,
}

impl<W: Worker> WorkerPool<W> {
    /// Zero-config constructor - auto-detect everything
    pub fn new() -> Result<Self> {
        let config = DefaultConfig::calculate();
        Self::with_config(config)
    }
    
    /// Start building with custom config
    pub fn builder() -> WorkerPoolBuilder<W, NeedsFactory> {
        WorkerPoolBuilder::new()
    }
}

/// Type-state builder pattern
pub struct WorkerPoolBuilder<W, State> {
    config: WorkerPoolConfig,
    factory: Option<Box<dyn Fn() -> W>>,
    _phantom: PhantomData<(W, State)>,
}

// Type states
pub struct NeedsFactory;
pub struct Ready;

impl<W: Worker> WorkerPoolBuilder<W, NeedsFactory> {
    pub fn new() -> Self {
        Self {
            config: DefaultConfig::calculate(),
            factory: None,
            _phantom: PhantomData,
        }
    }
    
    /// Provide worker factory (required)
    pub fn factory<F>(mut self, factory: F) -> WorkerPoolBuilder<W, Ready>
    where
        F: Fn() -> W + 'static,
    {
        WorkerPoolBuilder {
            config: self.config,
            factory: Some(Box::new(factory)),
            _phantom: PhantomData,
        }
    }
}

impl<W: Worker> WorkerPoolBuilder<W, Ready> {
    // Basic configuration
    
    pub fn workers(mut self, num: usize) -> Self {
        self.config.num_app_workers = num;
        self
    }
    
    pub fn io_workers(mut self, num: usize) -> Self {
        self.config.num_io_workers = num;
        self
    }
    
    pub fn channel_size(mut self, size: usize) -> Self {
        self.config.channel_size = size;
        self
    }
    
    // Advanced configuration
    
    pub fn cpu_affinity(mut self, enabled: bool) -> Self {
        self.config.cpu_affinity = enabled;
        self
    }
    
    pub fn numa_aware(mut self, enabled: bool) -> Self {
        self.config.numa_aware = enabled;
        self
    }
    
    pub fn pin_cores(mut self, cores: Vec<usize>) -> Self {
        self.config.pinned_cores = Some(cores);
        self
    }
    
    // Storage configuration
    
    pub fn storage(mut self, backend: StorageBackend) -> Self {
        self.config.storage_backend = Some(backend);
        self
    }
    
    pub fn storage_spdk(mut self, device: impl Into<String>) -> Self {
        self.config.storage_backend = Some(StorageBackend::Spdk {
            device: device.into(),
        });
        self
    }
    
    pub fn storage_io_uring(mut self, path: impl Into<String>) -> Self {
        self.config.storage_backend = Some(StorageBackend::IoUring {
            path: path.into(),
            queue_depth: 256,
        });
        self
    }
    
    // Network configuration
    
    pub fn network(mut self, backend: NetworkBackend) -> Self {
        self.config.network_backend = Some(backend);
        self
    }
    
    // Accelerator configuration
    
    pub fn gpu(mut self, enabled: bool) -> Self {
        self.config.use_gpu = enabled;
        self
    }
    
    pub fn gpu_device(mut self, device_id: usize) -> Self {
        self.config.gpu_device = Some(device_id);
        self
    }
    
    pub fn crypto_accelerator(mut self, enabled: bool) -> Self {
        self.config.use_qat = enabled;
        self
    }
    
    // Build
    
    pub fn build(self) -> Result<WorkerPool<W>> {
        WorkerPool::with_config_and_factory(
            self.config,
            self.factory.expect("Factory already set in type state"),
        )
    }
}

/// Convenient fluent API methods
impl<W: Worker> WorkerPoolBuilder<W, Ready> {
    /// Chain multiple configs
    pub fn with<F>(mut self, f: F) -> Self
    where
        F: FnOnce(WorkerPoolConfig) -> WorkerPoolConfig,
    {
        self.config = f(self.config);
        self
    }
    
    /// Conditional configuration
    pub fn when<F>(self, condition: bool, f: F) -> Self
    where
        F: FnOnce(Self) -> Self,
    {
        if condition {
            f(self)
        } else {
            self
        }
    }
}
```

---

## Profile-Based Presets

### Predefined Profiles

```rust
/// Predefined configuration profiles
pub enum Profile {
    /// Development: Single-threaded, debug-friendly
    Development,
    
    /// Testing: Deterministic, reproducible
    Testing,
    
    /// Production: Optimized for throughput and reliability
    Production,
    
    /// Performance: Maximum performance, all optimizations
    Performance,
    
    /// LowLatency: Optimized for low latency
    LowLatency,
    
    /// HighThroughput: Optimized for maximum throughput
    HighThroughput,
    
    /// Minimal: Minimum resource usage
    Minimal,
    
    /// Embedded: Resource-constrained environments
    Embedded,
}

impl Profile {
    pub fn config(&self) -> WorkerPoolConfig {
        let caps = SystemCapabilities::detect();
        
        match self {
            Profile::Development => Self::development_config(&caps),
            Profile::Testing => Self::testing_config(&caps),
            Profile::Production => Self::production_config(&caps),
            Profile::Performance => Self::performance_config(&caps),
            Profile::LowLatency => Self::low_latency_config(&caps),
            Profile::HighThroughput => Self::high_throughput_config(&caps),
            Profile::Minimal => Self::minimal_config(&caps),
            Profile::Embedded => Self::embedded_config(&caps),
        }
    }
    
    fn development_config(caps: &SystemCapabilities) -> WorkerPoolConfig {
        WorkerPoolConfig {
            // Single worker for debugging
            num_app_workers: 1,
            num_io_workers: 1,
            num_accelerator_workers: 0,
            
            // No CPU pinning for flexibility
            cpu_affinity: false,
            numa_aware: false,
            
            // Small channels
            channel_size: 16,
            
            // Standard backends
            storage_backend: Some(StorageBackend::Standard {
                path: "/tmp/dev.db".to_string(),
            }),
            network_backend: Some(NetworkBackend::Standard),
            
            // No accelerators
            use_gpu: false,
            use_qat: false,
            
            // Debug settings
            debug_mode: true,
            log_level: LogLevel::Debug,
        }
    }
    
    fn production_config(caps: &SystemCapabilities) -> WorkerPoolConfig {
        WorkerPoolConfig {
            // Use most cores for workers
            num_app_workers: (caps.cpu.num_physical_cores as f64 * 0.7) as usize,
            num_io_workers: (caps.cpu.num_physical_cores as f64 * 0.2) as usize,
            num_accelerator_workers: caps.accelerators.gpus.len().min(4),
            
            // Enable optimizations
            cpu_affinity: caps.cpu.num_cores > 8,
            numa_aware: caps.cpu.numa_nodes.len() > 1,
            
            // Reasonable channel sizes
            channel_size: 1024,
            
            // Best available backends
            storage_backend: DefaultConfig::select_storage_backend(&caps.storage),
            network_backend: Some(DefaultConfig::select_network_backend(&caps.network)),
            
            // Use accelerators if available
            use_gpu: !caps.accelerators.gpus.is_empty(),
            use_qat: caps.accelerators.has_qat,
            
            // Production settings
            debug_mode: false,
            log_level: LogLevel::Info,
        }
    }
    
    fn performance_config(caps: &SystemCapabilities) -> WorkerPoolConfig {
        WorkerPoolConfig {
            // Use ALL cores
            num_app_workers: caps.cpu.num_physical_cores.saturating_sub(4),
            num_io_workers: 2,
            num_accelerator_workers: caps.accelerators.gpus.len(),
            
            // All optimizations enabled
            cpu_affinity: true,
            numa_aware: true,
            huge_pages: true,
            zero_copy: true,
            
            // Large channels
            channel_size: 8192,
            
            // Highest performance backends
            storage_backend: Some(StorageBackend::Spdk {
                device: caps.storage.nvme_devices
                    .first()
                    .map(|d| d.pci_address.clone())
                    .unwrap_or_default(),
            }),
            network_backend: Some(NetworkBackend::Dpdk),
            
            // All accelerators
            use_gpu: true,
            use_qat: true,
            
            // Minimal logging
            debug_mode: false,
            log_level: LogLevel::Error,
        }
    }
    
    fn low_latency_config(caps: &SystemCapabilities) -> WorkerPoolConfig {
        WorkerPoolConfig {
            // Fewer workers for lower latency
            num_app_workers: caps.cpu.num_physical_cores / 2,
            num_io_workers: caps.cpu.numa_nodes.len() * 2,
            num_accelerator_workers: caps.accelerators.gpus.len(),
            
            // Strict CPU pinning
            cpu_affinity: true,
            numa_aware: true,
            isolated_cores: true,
            realtime_priority: true,
            
            // Small channels (lower latency)
            channel_size: 64,
            
            // Lowest latency backends
            storage_backend: Some(StorageBackend::Dax {
                path: "/mnt/pmem/latency.db".to_string(),
                size: 1024 * 1024 * 1024,
            }),
            network_backend: Some(NetworkBackend::Rdma),
            
            // Use accelerators
            use_gpu: !caps.accelerators.gpus.is_empty(),
            use_qat: caps.accelerators.has_qat,
            
            // Minimal overhead
            debug_mode: false,
            log_level: LogLevel::Error,
            metrics_enabled: false,
        }
    }
    
    fn minimal_config(caps: &SystemCapabilities) -> WorkerPoolConfig {
        WorkerPoolConfig {
            // Minimal workers
            num_app_workers: 2,
            num_io_workers: 1,
            num_accelerator_workers: 0,
            
            // No special features
            cpu_affinity: false,
            numa_aware: false,
            
            // Small channels
            channel_size: 32,
            
            // Standard backends
            storage_backend: Some(StorageBackend::Standard {
                path: "/tmp/minimal.db".to_string(),
            }),
            network_backend: Some(NetworkBackend::Standard),
            
            // No accelerators
            use_gpu: false,
            use_qat: false,
            
            debug_mode: false,
            log_level: LogLevel::Warn,
        }
    }
}

/// Convenient profile constructors
impl<W: Worker> WorkerPool<W> {
    pub fn development() -> Result<Self> {
        Self::with_config(Profile::Development.config())
    }
    
    pub fn production() -> Result<Self> {
        Self::with_config(Profile::Production.config())
    }
    
    pub fn performance() -> Result<Self> {
        Self::with_config(Profile::Performance.config())
    }
    
    pub fn low_latency() -> Result<Self> {
        Self::with_config(Profile::LowLatency.config())
    }
    
    pub fn minimal() -> Result<Self> {
        Self::with_config(Profile::Minimal.config())
    }
}
```

---

## Runtime Adaptation

### Adaptive Configuration Manager

```rust
/// Continuously monitors and adapts configuration
pub struct AdaptiveManager {
    metrics: Arc<MetricsCollector>,
    config: Arc<RwLock<WorkerPoolConfig>>,
    adaptation_interval: Duration,
}

impl AdaptiveManager {
    pub fn new(initial_config: WorkerPoolConfig) -> Self {
        Self {
            metrics: Arc::new(MetricsCollector::new()),
            config: Arc::new(RwLock::new(initial_config)),
            adaptation_interval: Duration::from_secs(10),
        }
    }
    
    pub async fn run(&self) {
        let mut interval = tokio::time::interval(self.adaptation_interval);
        
        loop {
            interval.tick().await;
            
            // Collect metrics
            let metrics = self.metrics.snapshot();
            
            // Analyze and adapt
            self.adapt(&metrics).await;
        }
    }
    
    async fn adapt(&self, metrics: &MetricsSnapshot) {
        let mut config = self.config.write();
        
        // Adapt worker count
        self.adapt_worker_count(&mut config, metrics);
        
        // Adapt batch sizes
        self.adapt_batch_sizes(&mut config, metrics);
        
        // Adapt channel sizes
        self.adapt_channel_sizes(&mut config, metrics);
        
        // Adapt accelerator usage
        self.adapt_accelerators(&mut config, metrics);
    }
    
    fn adapt_worker_count(&self, config: &mut WorkerPoolConfig, metrics: &MetricsSnapshot) {
        // If CPU utilization is high, add workers
        if metrics.avg_cpu_utilization > 0.9 {
            let new_count = (config.num_app_workers as f64 * 1.2) as usize;
            config.num_app_workers = new_count.min(metrics.available_cores);
        }
        
        // If CPU utilization is low, reduce workers
        if metrics.avg_cpu_utilization < 0.5 && config.num_app_workers > 2 {
            let new_count = (config.num_app_workers as f64 * 0.8) as usize;
            config.num_app_workers = new_count.max(2);
        }
        
        // If queue depth is high, add I/O workers
        if metrics.avg_queue_depth > 1000 {
            config.num_io_workers = (config.num_io_workers + 1).min(8);
        }
    }
    
    fn adapt_batch_sizes(&self, config: &mut WorkerPoolConfig, metrics: &MetricsSnapshot) {
        // Increase batch size if throughput is priority
        if metrics.throughput_trend == Trend::Increasing {
            config.batch_size = (config.batch_size as f64 * 1.2) as usize;
        }
        
        // Decrease batch size if latency is increasing
        if metrics.p99_latency > config.target_latency_ms {
            config.batch_size = (config.batch_size as f64 * 0.8) as usize;
        }
    }
    
    fn adapt_channel_sizes(&self, config: &mut WorkerPoolConfig, metrics: &MetricsSnapshot) {
        // If channels are frequently full, increase size
        if metrics.channel_full_rate > 0.1 {
            config.channel_size = (config.channel_size as f64 * 1.5) as usize;
        }
        
        // If channels are mostly empty, decrease size
        if metrics.avg_channel_utilization < 0.2 {
            config.channel_size = (config.channel_size / 2).max(16);
        }
    }
    
    fn adapt_accelerators(&self, config: &mut WorkerPoolConfig, metrics: &MetricsSnapshot) {
        // If GPU is underutilized, reduce GPU workers
        if config.use_gpu && metrics.gpu_utilization < 0.3 {
            // Consider disabling or reducing GPU workers
            if metrics.cpu_compute_faster_than_gpu {
                config.use_gpu = false;
            }
        }
        
        // If CPU is overloaded and GPU available, enable GPU
        if !config.use_gpu 
            && metrics.avg_cpu_utilization > 0.9 
            && metrics.has_available_gpu 
        {
            config.use_gpu = true;
        }
    }
}

/// Metrics collection
pub struct MetricsSnapshot {
    pub avg_cpu_utilization: f64,
    pub avg_queue_depth: usize,
    pub throughput_trend: Trend,
    pub p99_latency: u64,
    pub channel_full_rate: f64,
    pub avg_channel_utilization: f64,
    pub gpu_utilization: f64,
    pub cpu_compute_faster_than_gpu: bool,
    pub has_available_gpu: bool,
    pub available_cores: usize,
    pub target_latency_ms: u64,
}

pub enum Trend {
    Increasing,
    Stable,
    Decreasing,
}
```

### Learning from Workload

```rust
/// Machine learning-based configuration optimizer
pub struct WorkloadLearner {
    history: VecDeque<WorkloadSample>,
    optimal_configs: HashMap<WorkloadPattern, WorkerPoolConfig>,
}

impl WorkloadLearner {
    pub fn learn(&mut self, sample: WorkloadSample) {
        self.history.push_back(sample.clone());
        
        // Keep last 1000 samples
        if self.history.len() > 1000 {
            self.history.pop_front();
        }
        
        // Identify pattern
        let pattern = self.identify_pattern(&sample);
        
        // Update optimal config for this pattern
        if sample.performance_score > self.get_best_score(&pattern) {
            self.optimal_configs.insert(pattern, sample.config);
        }
    }
    
    fn identify_pattern(&self, sample: &WorkloadSample) -> WorkloadPattern {
        WorkloadPattern {
            request_rate: Self::bucket_rate(sample.requests_per_sec),
            compute_intensity: Self::bucket_compute(sample.avg_compute_us),
            io_intensity: Self::bucket_io(sample.avg_io_us),
            memory_usage: Self::bucket_memory(sample.memory_bytes),
        }
    }
    
    pub fn suggest_config(&self, current_workload: &WorkloadSample) -> Option<WorkerPoolConfig> {
        let pattern = self.identify_pattern(current_workload);
        self.optimal_configs.get(&pattern).cloned()
    }
}

#[derive(Hash, Eq, PartialEq)]
pub struct WorkloadPattern {
    request_rate: RateBucket,
    compute_intensity: ComputeBucket,
    io_intensity: IoBucket,
    memory_usage: MemoryBucket,
}

pub struct WorkloadSample {
    pub requests_per_sec: u64,
    pub avg_compute_us: u64,
    pub avg_io_us: u64,
    pub memory_bytes: usize,
    pub config: WorkerPoolConfig,
    pub performance_score: f64,
}
```

---

## Feature Flag Defaults

### Cargo Feature Configuration

```toml
# Cargo.toml

[features]
# Default feature set: Safe, portable, works everywhere
default = ["safe-defaults"]

# Safe defaults: Standard I/O, no special hardware requirements
safe-defaults = []

# Performance: Enable all performance optimizations
performance = [
    "io-uring",
    "numa-aware",
    "huge-pages",
    "cuda",
    "vulkan",
]

# Server: Optimized for server deployments
server = [
    "io-uring",
    "numa-aware",
    "spdk",
    "dpdk",
    "qat",
]

# Embedded: Minimal dependencies, small binary
embedded = [
    "minimal-deps",
    "no-std-support",
]

# Compatibility: Maximum compatibility, minimal features
compatibility = [
    "safe-defaults",
    "fallback-implementations",
]

# Development: Extra debugging, logging
development = [
    "debug-logging",
    "metrics",
    "tracing",
]

# Individual features
io-uring = ["dep:io-uring"]
spdk = ["dep:spdk-sys"]
dpdk = ["dep:dpdk-sys"]
rdma = ["dep:rdma-core"]
numa-aware = ["dep:hwloc"]
huge-pages = []
cuda = ["dep:cudarc"]
vulkan = ["dep:vulkano"]
metal = ["dep:metal-rs"]
wgpu = ["dep:wgpu"]
qat = ["dep:qat-sys"]

# Convenience feature sets
all-networking = ["io-uring", "dpdk", "rdma", "af-xdp"]
all-storage = ["spdk", "io-uring", "dax"]
all-accelerators = ["cuda", "vulkan", "metal", "wgpu", "qat"]
all = ["all-networking", "all-storage", "all-accelerators"]
```

### Feature-Based Configuration

```rust
/// Configuration based on enabled features
pub struct FeatureBasedConfig;

impl FeatureBasedConfig {
    pub fn default_for_features() -> WorkerPoolConfig {
        let mut config = WorkerPoolConfig::default();
        
        // Adjust based on enabled features
        #[cfg(feature = "performance")]
        {
            config.apply_performance_optimizations();
        }
        
        #[cfg(feature = "server")]
        {
            config.apply_server_optimizations();
        }
        
        #[cfg(feature = "embedded")]
        {
            config.apply_embedded_constraints();
        }
        
        #[cfg(feature = "development")]
        {
            config.enable_debugging();
        }
        
        config
    }
}

impl WorkerPoolConfig {
    #[cfg(feature = "performance")]
    fn apply_performance_optimizations(&mut self) {
        // Use all available optimizations
        self.cpu_affinity = true;
        self.numa_aware = true;
        self.huge_pages = true;
        self.zero_copy = true;
        
        #[cfg(feature = "io-uring")]
        {
            self.io_backend = IoBackend::IoUring;
        }
        
        #[cfg(feature = "cuda")]
        {
            self.use_gpu = true;
        }
    }
    
    #[cfg(feature = "server")]
    fn apply_server_optimizations(&mut self) {
        // Server-appropriate settings
        self.cpu_affinity = true;
        self.numa_aware = true;
        
        #[cfg(feature = "spdk")]
        {
            self.storage_backend = Some(StorageBackend::Spdk {
                device: Self::auto_detect_nvme(),
            });
        }
        
        #[cfg(feature = "dpdk")]
        {
            self.network_backend = Some(NetworkBackend::Dpdk);
        }
    }
    
    #[cfg(feature = "embedded")]
    fn apply_embedded_constraints(&mut self) {
        // Minimize resource usage
        self.num_app_workers = 2;
        self.num_io_workers = 1;
        self.channel_size = 16;
        self.cpu_affinity = false;
        self.numa_aware = false;
        self.use_gpu = false;
    }
    
    #[cfg(feature = "development")]
    fn enable_debugging(&mut self) {
        self.debug_mode = true;
        self.log_level = LogLevel::Debug;
        self.metrics_enabled = true;
        self.tracing_enabled = true;
    }
}
```

---

## Capability-Based Configuration

### Automatic Feature Selection

```rust
/// Select best available implementation at runtime
pub struct CapabilitySelector;

impl CapabilitySelector {
    pub fn select_best<T>(
        implementations: Vec<(Capability, T)>,
        caps: &SystemCapabilities,
    ) -> Option<T> {
        // Sort by priority (highest first)
        let mut viable: Vec<_> = implementations
            .into_iter()
            .filter(|(cap, _)| cap.is_available(caps))
            .collect();
        
        viable.sort_by_key(|(cap, _)| cap.priority());
        
        viable.into_iter().next().map(|(_, impl_)| impl_)
    }
}

pub enum Capability {
    Spdk,
    IoUring,
    Dpdk,
    Rdma,
    Cuda,
    Vulkan,
    Metal,
    Wgpu,
    Standard,
}

impl Capability {
    pub fn is_available(&self, caps: &SystemCapabilities) -> bool {
        match self {
            Capability::Spdk => caps.storage.has_spdk,
            Capability::IoUring => caps.storage.has_io_uring,
            Capability::Dpdk => caps.network.has_dpdk,
            Capability::Rdma => caps.network.has_rdma,
            Capability::Cuda => caps.accelerators.gpus.iter().any(|g| g.is_cuda()),
            Capability::Vulkan => caps.accelerators.gpus.iter().any(|g| g.supports_vulkan()),
            Capability::Metal => caps.accelerators.gpus.iter().any(|g| g.is_metal()),
            Capability::Wgpu => true, // Always available
            Capability::Standard => true, // Always available
        }
    }
    
    pub fn priority(&self) -> u32 {
        match self {
            // Lower number = higher priority
            Capability::Spdk => 1,
            Capability::IoUring => 2,
            Capability::Dpdk => 1,
            Capability::Rdma => 1,
            Capability::Cuda => 1,
            Capability::Vulkan => 2,
            Capability::Metal => 1,
            Capability::Wgpu => 3,
            Capability::Standard => 100,
        }
    }
}

/// Automatically select implementation
pub trait AutoSelect: Sized {
    fn auto_select(caps: &SystemCapabilities) -> Result<Self>;
}

impl AutoSelect for Box<dyn StorageProtocol> {
    fn auto_select(caps: &SystemCapabilities) -> Result<Self> {
        let impls = vec![
            #[cfg(feature = "spdk")]
            (Capability::Spdk, || -> Result<Box<dyn StorageProtocol>> {
                Ok(Box::new(SpdkStorage::new(&caps.storage.nvme_devices[0].pci_address)?))
            }),
            
            #[cfg(feature = "io-uring")]
            (Capability::IoUring, || -> Result<Box<dyn StorageProtocol>> {
                Ok(Box::new(IoUringStorage::new("/tmp/storage.db", 256)?))
            }),
            
            (Capability::Standard, || -> Result<Box<dyn StorageProtocol>> {
                Ok(Box::new(StandardStorage::new("/tmp/storage.db")?))
            }),
        ];
        
        CapabilitySelector::select_best(impls, caps)
            .ok_or("No storage implementation available")?()
    }
}
```

---

## Usage Examples

### Example 1: Absolute Zero Config

```rust
use shared_nothing::prelude::*;

fn main() -> Result<()> {
    // That's it! Auto-detects everything
    let pool = WorkerPool::new()?;
    
    pool.run()?;
    
    Ok(())
}

// Auto-detected:
// - 16 application workers (70% of 24 cores)
// - 4 I/O workers (2 per NUMA node)
// - 2 GPU workers (detected 2 GPUs)
// - io_uring for storage (detected Linux 5.1+)
// - NUMA-aware allocation (detected 2 NUMA nodes)
// - CPU affinity enabled (>16 cores detected)
```

### Example 2: Profile Selection

```rust
use shared_nothing::prelude::*;

fn main() -> Result<()> {
    // Use production profile
    let pool = WorkerPool::production()?;
    
    // Or for development
    let pool = WorkerPool::development()?;
    
    // Or maximum performance
    let pool = WorkerPool::performance()?;
    
    pool.run()?;
    
    Ok(())
}
```

### Example 3: Minimal Configuration

```rust
use shared_nothing::prelude::*;

fn main() -> Result<()> {
    // Just specify worker count
    let pool = WorkerPool::builder()
        .factory(|| MyWorker::new())
        .workers(8)
        .build()?;
    
    pool.run()?;
    
    Ok(())
}
```

### Example 4: Selective Configuration

```rust
use shared_nothing::prelude::*;

fn main() -> Result<()> {
    // Override specific settings
    let pool = WorkerPool::builder()
        .factory(|| MyWorker::new())
        .workers(16)
        .gpu(true)
        .storage_io_uring("/data/app.db")
        .build()?;
    
    pool.run()?;
    
    Ok(())
}
```

### Example 5: Full Control

```rust
use shared_nothing::prelude::*;

fn main() -> Result<()> {
    let pool = WorkerPool::builder()
        .factory(|| MyWorker::new())
        .workers(32)
        .io_workers(8)
        .channel_size(4096)
        .cpu_affinity(true)
        .numa_aware(true)
        .pin_cores(vec![0, 1, 2, 3, 4, 5, 6, 7])
        .storage(StorageBackend::Spdk {
            device: "0000:03:00.0".to_string(),
        })
        .network(NetworkBackend::Dpdk)
        .gpu_device(0)
        .crypto_accelerator(true)
        .with(|mut config| {
            config.custom_setting = true;
            config
        })
        .build()?;
    
    pool.run()?;
    
    Ok(())
}
```

### Example 6: Conditional Configuration

```rust
use shared_nothing::prelude::*;

fn main() -> Result<()> {
    let is_production = std::env::var("ENV")
        .map(|v| v == "production")
        .unwrap_or(false);
    
    let pool = WorkerPool::builder()
        .factory(|| MyWorker::new())
        .workers(8)
        .when(is_production, |builder| {
            builder
                .cpu_affinity(true)
                .numa_aware(true)
                .gpu(true)
        })
        .build()?;
    
    pool.run()?;
    
    Ok(())
}
```

### Example 7: Feature-Based Configuration

```rust
// Different behavior based on compiled features

#[cfg(feature = "performance")]
fn create_pool() -> Result<WorkerPool<MyWorker>> {
    // All optimizations enabled
    WorkerPool::performance()
}

#[cfg(feature = "embedded")]
fn create_pool() -> Result<WorkerPool<MyWorker>> {
    // Minimal resource usage
    WorkerPool::minimal()
}

#[cfg(not(any(feature = "performance", feature = "embedded")))]
fn create_pool() -> Result<WorkerPool<MyWorker>> {
    // Safe defaults
    WorkerPool::new()
}
```

---

## Migration Paths

### From Zero Config to Custom Config

```rust
// Stage 1: Zero config (just starting)
let pool = WorkerPool::new()?;

// Stage 2: Profile (production deployment)
let pool = WorkerPool::production()?;

// Stage 3: Tune one setting (need more workers)
let pool = WorkerPool::builder()
    .factory(|| MyWorker::new())
    .workers(32)
    .build()?;

// Stage 4: Fine-tune multiple settings
let pool = WorkerPool::builder()
    .factory(|| MyWorker::new())
    .workers(32)
    .io_workers(8)
    .gpu(true)
    .build()?;

// Stage 5: Full control (performance critical)
let pool = WorkerPool::builder()
    .factory(|| MyWorker::new())
    // ... all settings ...
    .build()?;
```

### Configuration Evolution Timeline

```
Week 1: Zero config
  ↓
Month 1: Profile selection
  ↓
Month 3: Worker count tuning
  ↓
Month 6: Storage/network backend selection
  ↓
Year 1: Full custom configuration
```

---

## Summary

### Configuration Layers

```
Layer 6: Full Custom Config (1% of users)
  ↓ fall through if not specified
Layer 5: Selective Override (4% of users)
  ↓ fall through
Layer 4: Profile Selection (15% of users)
  ↓ fall through
Layer 3: Feature Flags (compile-time)
  ↓ fall through
Layer 2: Runtime Adaptation (automatic)
  ↓ fall through
Layer 1: Auto-Detection (80% of users)
```

### Benefits

✅ **Zero Config**: Works out of the box with no configuration  
✅ **Smart Defaults**: Automatically detects and uses best hardware  
✅ **Progressive**: Start simple, add complexity only when needed  
✅ **Profile-Based**: One-line config for common scenarios  
✅ **Adaptive**: Learns and optimizes based on workload  
✅ **Feature Flags**: Compile-time optimization for binary size  
✅ **Type-Safe**: Builder pattern prevents invalid configurations  
✅ **Gradual Migration**: Easy path from simple to complex  

### Implementation Priority

**Phase 1** (Month 1):
- Auto-detection system
- Basic builder pattern
- Development/Production profiles

**Phase 2** (Month 2):
- All profiles
- Feature flag integration
- Capability-based selection

**Phase 3** (Month 3):
- Runtime adaptation
- Workload learning
- Advanced builder features

**Phase 4** (Month 4):
- Performance tuning
- Documentation
- Migration guides

---

**Document Version**: 1.0  
**Last Updated**: October 31, 2025  
**Status**: Design Complete - Ready for Implementation  
**Philosophy**: "It just works, but you can tweak everything"

