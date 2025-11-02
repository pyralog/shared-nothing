# Accelerator Integration: Unified Multi-Accelerator Architecture

This document defines the integration of **Dedicated Accelerator Workers** + **Hybrid CPU-GPU Pipeline** + **Opportunistic Acceleration** + **Specialized Accelerator Pools** for comprehensive hardware acceleration support.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Specialized Accelerator Pools](#specialized-accelerator-pools)
3. [Hybrid CPU-GPU Pipeline](#hybrid-cpu-gpu-pipeline)
4. [Opportunistic Acceleration](#opportunistic-acceleration)
5. [Accelerator Protocol Layer](#accelerator-protocol-layer)
6. [Worker Allocation Strategy](#worker-allocation-strategy)
7. [Task Scheduling & Batching](#task-scheduling--batching)
8. [Memory Management](#memory-management)
9. [Performance Optimization](#performance-optimization)
10. [Integration Examples](#integration-examples)

---

## Architecture Overview

### Unified Multi-Tier Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Application Workers                             │
│                  (Business Logic, Orchestration)                     │
│                         Cores 16-63                                  │
└─────────────────────────────────────────────────────────────────────┘
                                  ▲ │
                                  │ │ Task Messages
                                  │ ▼
┌─────────────────────────────────────────────────────────────────────┐
│              Opportunistic Acceleration Layer                        │
│         (Auto-detect, Load Balance, Fallback Manager)               │
└─────────────────────────────────────────────────────────────────────┘
                                  ▲ │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
                ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 Specialized Accelerator Pools                        │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   GPU Pool   │  │  Crypto Pool │  │ Network Pool │             │
│  │  (Cores 4-7) │  │  (Cores 8-9) │  │ (Cores 10-11)│             │
│  │              │  │              │  │              │             │
│  │ • CUDA       │  │ • QAT        │  │ • DPU        │             │
│  │ • Metal      │  │ • AES-NI     │  │ • XDP        │             │
│  │ • Vulkan     │  │ • Compress   │  │ • DPDK       │             │
│  │ • wgpu       │  │              │  │              │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │   ML Pool    │  │  Signal Pool │  │  Video Pool  │             │
│  │(Cores 12-13) │  │(Cores 14-15) │  │  (Cores 4-5) │             │
│  │              │  │              │  │              │             │
│  │ • TPU        │  │ • FPGA       │  │ • HW Decode  │             │
│  │ • NPU        │  │ • DSP        │  │ • HW Encode  │             │
│  │ • Inference  │  │ • FFT        │  │ • GPU Proc   │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
                                  ▲ │
                                  │ │ Hardware Operations
                                  │ ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Hardware Accelerators                             │
│                                                                      │
│  GPU0  GPU1  │  QAT  │  DPU  │  TPU  │  FPGA  │  Video Codec       │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Specialized Pools**: Separate worker pools per accelerator type
2. **Pipeline Architecture**: Overlap CPU and GPU execution
3. **Opportunistic**: Auto-detect hardware, graceful fallback
4. **Dynamic Routing**: Route tasks to best available accelerator
5. **Zero-Copy**: Minimize data movement between stages
6. **Composable**: Mix accelerator types in same application

---

## Specialized Accelerator Pools

### 1. GPU Compute Pool

**Purpose**: General-purpose GPU computing (CUDA, Metal, Vulkan, wgpu)

**Worker Configuration**:
```rust
pub struct GpuPoolConfig {
    /// Number of GPU workers per device
    pub workers_per_gpu: usize,
    
    /// GPU device selection
    pub devices: Vec<GpuDevice>,
    
    /// Backend preference
    pub backend: GpuBackend,
    
    /// Memory pool size per worker
    pub memory_pool_size: usize,
    
    /// Batch configuration
    pub max_batch_size: usize,
    pub batch_timeout: Duration,
}

pub enum GpuBackend {
    Cuda { device_id: usize },
    Metal { device_name: String },
    Vulkan { device_id: usize },
    Wgpu { auto_select: bool },
}

pub struct GpuWorkerPool {
    workers: Vec<GpuWorker>,
    device_map: HashMap<usize, Arc<GpuDevice>>,
    stream_pool: StreamPool,
    memory_manager: GpuMemoryManager,
}
```

**Operations Supported**:
- Matrix operations (BLAS)
- Convolutions
- FFT/IFFT
- Image processing
- General compute kernels
- Reduce operations
- Sort/scan operations

**CPU Core Assignment**:
- Cores 4-7 (4 workers for quad-GPU setup)
- Pin to cores on same NUMA node as GPU
- Avoid hyperthreads

---

### 2. Crypto Accelerator Pool

**Purpose**: Hardware-accelerated cryptography and compression (Intel QAT, AES-NI)

**Worker Configuration**:
```rust
pub struct CryptoPoolConfig {
    /// Number of crypto workers
    pub num_workers: usize,
    
    /// QAT device IDs
    pub qat_devices: Vec<usize>,
    
    /// Supported operations
    pub operations: CryptoOperations,
    
    /// Session pool size
    pub session_pool_size: usize,
}

pub struct CryptoOperations {
    pub symmetric: bool,      // AES-GCM, ChaCha20
    pub asymmetric: bool,     // RSA, ECDSA
    pub hash: bool,           // SHA-256, SHA-512
    pub compression: bool,    // DEFLATE, LZ4
}

pub struct CryptoWorkerPool {
    workers: Vec<CryptoWorker>,
    qat_instances: Vec<Arc<QatInstance>>,
    session_manager: SessionManager,
}
```

**Operations Supported**:
- AES-128/192/256 (GCM, CBC, CTR)
- ChaCha20-Poly1305
- RSA 2048/4096
- ECDSA P-256/P-384
- SHA-256/384/512
- DEFLATE/LZ4/ZSTD compression

**CPU Core Assignment**:
- Cores 8-9 (2 workers)
- Pin to cores with AES-NI instructions
- Near QAT PCIe device

---

### 3. Network Accelerator Pool

**Purpose**: Hardware-accelerated networking (DPU, SmartNIC, XDP)

**Worker Configuration**:
```rust
pub struct NetworkAccelPoolConfig {
    /// Number of network workers
    pub num_workers: usize,
    
    /// DPU/SmartNIC devices
    pub devices: Vec<DpuDevice>,
    
    /// Offload capabilities
    pub offloads: NetworkOffloads,
    
    /// Packet buffer pool
    pub packet_pool_size: usize,
}

pub struct NetworkOffloads {
    pub checksum: bool,
    pub tso: bool,            // TCP Segmentation Offload
    pub lro: bool,            // Large Receive Offload
    pub rss: bool,            // Receive Side Scaling
    pub packet_filter: bool,  // eBPF/XDP
    pub crypto: bool,         // IPsec offload
}

pub struct NetworkAccelWorkerPool {
    workers: Vec<NetworkAccelWorker>,
    dpu_handles: Vec<Arc<DpuHandle>>,
    packet_buffers: PacketBufferPool,
}
```

**Operations Supported**:
- Packet filtering (XDP/eBPF)
- TCP/UDP checksum offload
- IPsec encryption
- Load balancing (RSS)
- Packet reordering
- Deep packet inspection

**CPU Core Assignment**:
- Cores 10-11 (2 workers)
- Pin to cores on same NUMA node as NIC
- Dedicated cores for interrupt handling

---

### 4. ML Inference Pool

**Purpose**: Specialized ML accelerators (TPU, NPU, Edge TPU)

**Worker Configuration**:
```rust
pub struct MlPoolConfig {
    /// Number of ML workers
    pub num_workers: usize,
    
    /// Accelerator type
    pub accelerator: MlAccelerator,
    
    /// Model configurations
    pub models: Vec<ModelConfig>,
    
    /// Batch inference settings
    pub batch_size: usize,
    pub batch_timeout: Duration,
}

pub enum MlAccelerator {
    Tpu { version: TpuVersion },
    AppleNeuralEngine,
    IntelMovidius,
    GoogleEdgeTpu,
    AwsInferentia,
    Cuda { device_id: usize },
}

pub struct MlWorkerPool {
    workers: Vec<MlWorker>,
    model_cache: Arc<ModelCache>,
    accelerator_handle: Arc<dyn MlAcceleratorHandle>,
}
```

**Operations Supported**:
- Image classification
- Object detection
- Natural language processing
- Recommendation inference
- Embeddings generation
- Model ensemble

**CPU Core Assignment**:
- Cores 12-13 (2 workers)
- Co-located with accelerator
- Minimal CPU overhead

---

### 5. Signal Processing Pool

**Purpose**: FPGA and DSP acceleration for real-time signal processing

**Worker Configuration**:
```rust
pub struct SignalPoolConfig {
    /// Number of signal processing workers
    pub num_workers: usize,
    
    /// FPGA/DSP devices
    pub devices: Vec<SignalDevice>,
    
    /// Bitstream/firmware
    pub bitstream_path: Option<String>,
    
    /// Real-time constraints
    pub max_latency_us: u64,
}

pub enum SignalDevice {
    Fpga { vendor: FpgaVendor, device_id: usize },
    Dsp { model: DspModel },
}

pub struct SignalWorkerPool {
    workers: Vec<SignalWorker>,
    fpga_handles: Vec<Arc<FpgaHandle>>,
    dma_buffers: DmaBufferPool,
}
```

**Operations Supported**:
- FFT/IFFT (fixed latency)
- FIR/IIR filters
- Correlations
- Modulation/demodulation
- Error correction coding
- Custom algorithms

**CPU Core Assignment**:
- Cores 14-15 (2 workers)
- Hard real-time scheduling (SCHED_FIFO)
- CPU isolation (isolcpus)

---

### 6. Video Processing Pool

**Purpose**: Hardware video codec acceleration

**Worker Configuration**:
```rust
pub struct VideoPoolConfig {
    /// Number of video workers
    pub num_workers: usize,
    
    /// Hardware codec engines
    pub codecs: Vec<VideoCodec>,
    
    /// GPU for processing
    pub gpu_device: Option<usize>,
}

pub enum VideoCodec {
    NvEnc,      // NVIDIA NVENC
    NvDec,      // NVIDIA NVDEC
    QuickSync,  // Intel Quick Sync
    VideoToolbox, // Apple VideoToolbox
    VaApi,      // Linux VA-API
}

pub struct VideoWorkerPool {
    workers: Vec<VideoWorker>,
    codec_handles: Vec<Arc<CodecHandle>>,
    frame_buffers: FrameBufferPool,
}
```

**Operations Supported**:
- H.264/H.265 encode/decode
- VP9/AV1 encode/decode
- GPU scaling/color conversion
- Overlay/composition
- Motion estimation

**CPU Core Assignment**:
- Share with GPU pool (Cores 4-5)
- Can multiplex with compute workloads

---

## Hybrid CPU-GPU Pipeline

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Stages                           │
│                                                              │
│  ┌────────┐  ┌─────────┐  ┌────────┐  ┌─────────┐         │
│  │Ingest  │→ │Preproc  │→ │ GPU    │→ │Postproc │→ Output │
│  │(CPU)   │  │(CPU)    │  │Compute │  │(CPU)    │         │
│  └────────┘  └─────────┘  └────────┘  └─────────┘         │
│      ↓            ↓            ↓            ↓               │
│  ┌────────────────────────────────────────────────┐        │
│  │         Overlapped Execution Timeline          │        │
│  │                                                 │        │
│  │ CPU: [====][====][====][====][====]             │        │
│  │ GPU:   [========][========][========]           │        │
│  │                                                 │        │
│  │ Time: 0ms  5ms  10ms  15ms  20ms  25ms         │        │
│  └────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

```rust
pub struct AcceleratorPipeline {
    stages: Vec<PipelineStage>,
    buffers: Vec<Arc<RingBuffer>>,
}

pub enum PipelineStage {
    Cpu {
        worker: CpuWorker,
        function: Box<dyn Fn(&[u8]) -> Result<Vec<u8>>>,
    },
    Gpu {
        pool: Arc<GpuWorkerPool>,
        kernel: GpuKernel,
    },
    Crypto {
        pool: Arc<CryptoWorkerPool>,
        operation: CryptoOperation,
    },
    Network {
        pool: Arc<NetworkAccelWorkerPool>,
        filter: PacketFilter,
    },
}

impl AcceleratorPipeline {
    pub fn new(stages: Vec<PipelineStage>) -> Self {
        // Create ring buffers between stages
        let buffers = (0..stages.len() - 1)
            .map(|_| Arc::new(RingBuffer::new(1024)))
            .collect();
        
        Self { stages, buffers }
    }
    
    pub async fn execute(&self, input: Vec<u8>) -> Result<Vec<u8>> {
        let mut data = input;
        
        for (stage, buffer) in self.stages.iter().zip(&self.buffers) {
            // Execute stage
            let result = match stage {
                PipelineStage::Cpu { function, .. } => {
                    function(&data)?
                }
                PipelineStage::Gpu { pool, kernel } => {
                    pool.execute_kernel(kernel, &data).await?
                }
                PipelineStage::Crypto { pool, operation } => {
                    pool.execute_operation(operation, &data).await?
                }
                PipelineStage::Network { pool, filter } => {
                    pool.apply_filter(filter, &data).await?
                }
            };
            
            // Write to next stage's buffer
            buffer.write(result.clone()).await?;
            data = result;
        }
        
        Ok(data)
    }
    
    pub async fn execute_streaming(&self, input_stream: impl Stream<Item = Vec<u8>>) {
        // Pipeline stages run concurrently
        // Each stage pulls from previous buffer, pushes to next
        
        let mut stage_handles = Vec::new();
        
        for (i, stage) in self.stages.iter().enumerate() {
            let input_buffer = if i == 0 {
                None // First stage reads from input stream
            } else {
                Some(self.buffers[i - 1].clone())
            };
            
            let output_buffer = self.buffers.get(i).cloned();
            
            let handle = tokio::spawn(async move {
                loop {
                    // Read from input
                    let data = if let Some(buf) = &input_buffer {
                        buf.read().await?
                    } else {
                        // Read from input stream
                        continue;
                    };
                    
                    // Process
                    let result = match stage {
                        // ... stage execution ...
                    };
                    
                    // Write to output
                    if let Some(buf) = &output_buffer {
                        buf.write(result).await?;
                    }
                }
            });
            
            stage_handles.push(handle);
        }
        
        // Wait for all stages
        futures::future::join_all(stage_handles).await;
    }
}
```

### Pipeline Examples

#### Example 1: Image Processing Pipeline

```rust
pub fn create_image_pipeline() -> AcceleratorPipeline {
    AcceleratorPipeline::new(vec![
        // Stage 1: Decode (CPU)
        PipelineStage::Cpu {
            worker: CpuWorker::new(),
            function: Box::new(|data| {
                // Decode JPEG/PNG
                let image = image::load_from_memory(data)?;
                Ok(image.into_bytes())
            }),
        },
        
        // Stage 2: Resize (GPU)
        PipelineStage::Gpu {
            pool: gpu_pool.clone(),
            kernel: GpuKernel::Resize {
                width: 224,
                height: 224,
            },
        },
        
        // Stage 3: Normalize (GPU)
        PipelineStage::Gpu {
            pool: gpu_pool.clone(),
            kernel: GpuKernel::Normalize {
                mean: vec![0.485, 0.456, 0.406],
                std: vec![0.229, 0.224, 0.225],
            },
        },
        
        // Stage 4: ML Inference (TPU/GPU)
        PipelineStage::Gpu {
            pool: gpu_pool.clone(),
            kernel: GpuKernel::MlInference {
                model: "resnet50".to_string(),
            },
        },
        
        // Stage 5: Post-process (CPU)
        PipelineStage::Cpu {
            worker: CpuWorker::new(),
            function: Box::new(|data| {
                // Parse results, apply NMS
                let predictions = parse_predictions(data)?;
                Ok(predictions)
            }),
        },
    ])
}
```

#### Example 2: Video Transcoding Pipeline

```rust
pub fn create_transcode_pipeline() -> AcceleratorPipeline {
    AcceleratorPipeline::new(vec![
        // Stage 1: Decode (HW Decoder)
        PipelineStage::Video {
            pool: video_pool.clone(),
            operation: VideoOperation::Decode {
                codec: VideoCodec::NvDec,
            },
        },
        
        // Stage 2: Scale/Filter (GPU)
        PipelineStage::Gpu {
            pool: gpu_pool.clone(),
            kernel: GpuKernel::VideoScale {
                width: 1920,
                height: 1080,
            },
        },
        
        // Stage 3: Encode (HW Encoder)
        PipelineStage::Video {
            pool: video_pool.clone(),
            operation: VideoOperation::Encode {
                codec: VideoCodec::NvEnc,
                preset: "fast".to_string(),
            },
        },
    ])
}
```

#### Example 3: Crypto Pipeline

```rust
pub fn create_crypto_pipeline() -> AcceleratorPipeline {
    AcceleratorPipeline::new(vec![
        // Stage 1: Compress (QAT)
        PipelineStage::Crypto {
            pool: crypto_pool.clone(),
            operation: CryptoOperation::Compress {
                algorithm: CompressionAlgorithm::Deflate,
            },
        },
        
        // Stage 2: Encrypt (QAT)
        PipelineStage::Crypto {
            pool: crypto_pool.clone(),
            operation: CryptoOperation::Encrypt {
                algorithm: EncryptionAlgorithm::AesGcm256,
                key: encryption_key.clone(),
            },
        },
        
        // Stage 3: Sign (QAT)
        PipelineStage::Crypto {
            pool: crypto_pool.clone(),
            operation: CryptoOperation::Sign {
                algorithm: SignatureAlgorithm::EcdsaP256,
                key: signing_key.clone(),
            },
        },
    ])
}
```

---

## Opportunistic Acceleration

### Auto-Detection & Capability Discovery

```rust
pub struct AcceleratorDiscovery {
    detected: HashMap<AcceleratorType, Vec<AcceleratorInfo>>,
}

impl AcceleratorDiscovery {
    pub fn discover() -> Self {
        let mut detected = HashMap::new();
        
        // Discover GPUs
        if let Ok(gpus) = Self::discover_gpus() {
            detected.insert(AcceleratorType::Gpu, gpus);
        }
        
        // Discover QAT devices
        if let Ok(qat) = Self::discover_qat() {
            detected.insert(AcceleratorType::Crypto, qat);
        }
        
        // Discover DPUs/SmartNICs
        if let Ok(dpus) = Self::discover_dpus() {
            detected.insert(AcceleratorType::Network, dpus);
        }
        
        // Discover TPUs
        if let Ok(tpus) = Self::discover_tpus() {
            detected.insert(AcceleratorType::Ml, tpus);
        }
        
        // Discover FPGAs
        if let Ok(fpgas) = Self::discover_fpgas() {
            detected.insert(AcceleratorType::Signal, fpgas);
        }
        
        Self { detected }
    }
    
    fn discover_gpus() -> Result<Vec<AcceleratorInfo>> {
        let mut gpus = Vec::new();
        
        // Try CUDA
        if let Ok(cuda_gpus) = Self::detect_cuda_gpus() {
            gpus.extend(cuda_gpus);
        }
        
        // Try Metal (macOS)
        #[cfg(target_os = "macos")]
        if let Ok(metal_gpus) = Self::detect_metal_gpus() {
            gpus.extend(metal_gpus);
        }
        
        // Try Vulkan
        if let Ok(vulkan_gpus) = Self::detect_vulkan_gpus() {
            gpus.extend(vulkan_gpus);
        }
        
        // Try wgpu (fallback)
        if gpus.is_empty() {
            if let Ok(wgpu_gpus) = Self::detect_wgpu_gpus() {
                gpus.extend(wgpu_gpus);
            }
        }
        
        Ok(gpus)
    }
    
    fn detect_cuda_gpus() -> Result<Vec<AcceleratorInfo>> {
        #[cfg(feature = "cuda")]
        {
            use cudarc::driver::*;
            
            let device_count = CudaDevice::count()?;
            let mut gpus = Vec::new();
            
            for i in 0..device_count {
                let device = CudaDevice::new(i)?;
                let props = device.properties()?;
                
                gpus.push(AcceleratorInfo {
                    type_: AcceleratorType::Gpu,
                    backend: AcceleratorBackend::Cuda,
                    device_id: i,
                    name: props.name.clone(),
                    memory_bytes: props.total_memory,
                    compute_capability: Some(props.compute_capability),
                    capabilities: vec![
                        "compute",
                        "ml_training",
                        "ml_inference",
                        "video",
                    ].iter().map(|s| s.to_string()).collect(),
                });
            }
            
            Ok(gpus)
        }
        #[cfg(not(feature = "cuda"))]
        Err("CUDA not available".into())
    }
    
    fn detect_qat_devices() -> Result<Vec<AcceleratorInfo>> {
        // Check for QAT devices in /sys/class
        let qat_path = "/sys/class/qat";
        if !Path::new(qat_path).exists() {
            return Err("QAT not available".into());
        }
        
        // Enumerate QAT devices
        // ...
        
        Ok(vec![])
    }
}

pub struct AcceleratorInfo {
    pub type_: AcceleratorType,
    pub backend: AcceleratorBackend,
    pub device_id: usize,
    pub name: String,
    pub memory_bytes: usize,
    pub compute_capability: Option<(u32, u32)>,
    pub capabilities: Vec<String>,
}

pub enum AcceleratorBackend {
    Cuda,
    Metal,
    Vulkan,
    Wgpu,
    Qat,
    Dpu,
    Tpu,
    Fpga,
}
```

### Dynamic Task Routing

```rust
pub struct OpportunisticRouter {
    discovery: AcceleratorDiscovery,
    pools: HashMap<AcceleratorType, Arc<dyn AcceleratorPool>>,
    fallback_strategy: FallbackStrategy,
    performance_tracker: PerformanceTracker,
}

impl OpportunisticRouter {
    pub fn new(config: RouterConfig) -> Self {
        let discovery = AcceleratorDiscovery::discover();
        let pools = Self::create_pools(&discovery, &config);
        
        Self {
            discovery,
            pools,
            fallback_strategy: config.fallback_strategy,
            performance_tracker: PerformanceTracker::new(),
        }
    }
    
    pub async fn execute_task(&self, task: AcceleratorTask) -> Result<Vec<u8>> {
        // Select best accelerator for task
        let accelerator = self.select_accelerator(&task)?;
        
        // Try execution
        match self.try_execute(&accelerator, &task).await {
            Ok(result) => {
                // Track performance
                self.performance_tracker.record_success(
                    &accelerator,
                    task.elapsed(),
                );
                Ok(result)
            }
            Err(e) => {
                // Try fallback
                self.execute_with_fallback(&task, e).await
            }
        }
    }
    
    fn select_accelerator(&self, task: &AcceleratorTask) -> Result<AcceleratorSelection> {
        let candidates = self.get_candidates_for_task(task);
        
        if candidates.is_empty() {
            return Err("No suitable accelerator found".into());
        }
        
        // Score each candidate
        let mut scored: Vec<_> = candidates
            .iter()
            .map(|candidate| {
                let score = self.score_candidate(candidate, task);
                (candidate, score)
            })
            .collect();
        
        // Sort by score (higher is better)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        Ok(scored[0].0.clone())
    }
    
    fn score_candidate(&self, candidate: &AcceleratorSelection, task: &AcceleratorTask) -> f64 {
        let mut score = 0.0;
        
        // Factor 1: Performance (from history)
        let perf_score = self.performance_tracker
            .get_avg_throughput(candidate)
            .unwrap_or(1.0);
        score += perf_score * 0.4;
        
        // Factor 2: Queue depth (lower is better)
        let queue_score = 1.0 / (1.0 + candidate.queue_depth() as f64);
        score += queue_score * 0.3;
        
        // Factor 3: Capability match
        let capability_score = if candidate.has_capability(&task.required_capability) {
            1.0
        } else {
            0.0
        };
        score += capability_score * 0.2;
        
        // Factor 4: Data locality (GPU memory residency)
        let locality_score = if candidate.has_data_resident(&task.data_id) {
            1.0
        } else {
            0.0
        };
        score += locality_score * 0.1;
        
        score
    }
    
    async fn execute_with_fallback(
        &self,
        task: &AcceleratorTask,
        error: Error,
    ) -> Result<Vec<u8>> {
        match self.fallback_strategy {
            FallbackStrategy::None => Err(error),
            
            FallbackStrategy::CpuFallback => {
                // Execute on CPU
                self.execute_on_cpu(task).await
            }
            
            FallbackStrategy::AlternateAccelerator => {
                // Try next best accelerator
                let alternates = self.get_alternate_accelerators(task);
                for alternate in alternates {
                    if let Ok(result) = self.try_execute(&alternate, task).await {
                        return Ok(result);
                    }
                }
                // Final fallback to CPU
                self.execute_on_cpu(task).await
            }
            
            FallbackStrategy::Retry => {
                // Retry on same accelerator
                self.retry_execution(task, 3).await
            }
        }
    }
}

pub enum FallbackStrategy {
    None,                    // Fail immediately
    CpuFallback,            // Fall back to CPU
    AlternateAccelerator,   // Try other accelerators
    Retry,                  // Retry same accelerator
}

pub struct AcceleratorSelection {
    pub type_: AcceleratorType,
    pub device_id: usize,
    pub backend: AcceleratorBackend,
}
```

### Graceful Degradation

```rust
pub struct DegradationManager {
    health_monitor: Arc<HealthMonitor>,
    load_balancer: Arc<LoadBalancer>,
}

impl DegradationManager {
    pub fn check_health(&self) -> SystemHealth {
        let mut health = SystemHealth::default();
        
        // Check GPU health
        for (gpu_id, gpu_health) in self.health_monitor.gpu_health() {
            if gpu_health.temperature > 85.0 {
                // GPU overheating - reduce load
                health.gpu_throttle.insert(gpu_id, 0.5);
            }
            
            if gpu_health.memory_usage > 0.95 {
                // GPU OOM risk - reject new tasks
                health.gpu_available.insert(gpu_id, false);
            }
            
            if gpu_health.error_rate > 0.1 {
                // GPU errors - disable
                health.gpu_available.insert(gpu_id, false);
            }
        }
        
        // Check QAT health
        for (qat_id, qat_health) in self.health_monitor.qat_health() {
            if qat_health.queue_depth > 10000 {
                // QAT overloaded
                health.qat_throttle.insert(qat_id, 0.7);
            }
        }
        
        health
    }
    
    pub fn apply_degradation(&self, health: &SystemHealth) {
        // Reduce GPU batch sizes if throttled
        for (gpu_id, throttle) in &health.gpu_throttle {
            self.load_balancer.set_gpu_capacity(*gpu_id, *throttle);
        }
        
        // Disable unhealthy accelerators
        for (gpu_id, available) in &health.gpu_available {
            if !available {
                self.load_balancer.disable_gpu(*gpu_id);
            }
        }
        
        // Increase CPU fallback rate
        if health.overall_capacity() < 0.5 {
            self.load_balancer.increase_cpu_fallback(0.3);
        }
    }
}

pub struct SystemHealth {
    pub gpu_throttle: HashMap<usize, f64>,      // 0.0-1.0
    pub gpu_available: HashMap<usize, bool>,
    pub qat_throttle: HashMap<usize, f64>,
    pub qat_available: HashMap<usize, bool>,
}
```

---

## Accelerator Protocol Layer

### Unified Accelerator Protocol

```rust
/// Base trait for all accelerator operations
pub trait AcceleratorProtocol: Send + Sync {
    /// Execute operation on accelerator
    fn execute(&self, operation: Operation) -> impl Future<Output = Result<Vec<u8>>>;
    
    /// Execute batch of operations
    fn execute_batch(&self, operations: Vec<Operation>) -> impl Future<Output = Result<Vec<Vec<u8>>>>;
    
    /// Get accelerator capabilities
    fn capabilities(&self) -> AcceleratorCapabilities;
    
    /// Get current load/utilization
    fn utilization(&self) -> AcceleratorUtilization;
    
    /// Allocate memory on accelerator
    fn allocate_memory(&self, size: usize) -> Result<AcceleratorMemory>;
    
    /// Free memory on accelerator
    fn free_memory(&self, memory: AcceleratorMemory) -> Result<()>;
}

pub struct AcceleratorCapabilities {
    pub compute: bool,
    pub ml_inference: bool,
    pub ml_training: bool,
    pub video_encode: bool,
    pub video_decode: bool,
    pub crypto_symmetric: bool,
    pub crypto_asymmetric: bool,
    pub compression: bool,
    pub packet_processing: bool,
    pub signal_processing: bool,
}

pub struct AcceleratorUtilization {
    pub compute_percent: f64,
    pub memory_percent: f64,
    pub queue_depth: usize,
    pub temperature_celsius: f64,
    pub power_watts: f64,
}

pub enum Operation {
    Compute(ComputeOperation),
    MlInference(MlOperation),
    Video(VideoOperation),
    Crypto(CryptoOperation),
    Network(NetworkOperation),
    Signal(SignalOperation),
}
```

### GPU Protocol Implementation

```rust
pub struct GpuProtocol {
    device: Arc<GpuDevice>,
    stream: CudaStream,
    memory_pool: Arc<MemoryPool>,
}

impl AcceleratorProtocol for GpuProtocol {
    async fn execute(&self, operation: Operation) -> Result<Vec<u8>> {
        match operation {
            Operation::Compute(op) => self.execute_compute(op).await,
            Operation::MlInference(op) => self.execute_ml(op).await,
            Operation::Video(op) => self.execute_video(op).await,
            _ => Err("Unsupported operation".into()),
        }
    }
    
    async fn execute_batch(&self, operations: Vec<Operation>) -> Result<Vec<Vec<u8>>> {
        // Group by operation type
        let mut compute_ops = Vec::new();
        let mut ml_ops = Vec::new();
        
        for op in operations {
            match op {
                Operation::Compute(c) => compute_ops.push(c),
                Operation::MlInference(m) => ml_ops.push(m),
                _ => {}
            }
        }
        
        // Execute in parallel on GPU
        let (compute_results, ml_results) = tokio::join!(
            self.execute_compute_batch(compute_ops),
            self.execute_ml_batch(ml_ops)
        );
        
        // Merge results
        let mut results = Vec::new();
        results.extend(compute_results?);
        results.extend(ml_results?);
        
        Ok(results)
    }
    
    fn capabilities(&self) -> AcceleratorCapabilities {
        AcceleratorCapabilities {
            compute: true,
            ml_inference: true,
            ml_training: self.device.supports_training(),
            video_encode: self.device.has_nvenc(),
            video_decode: self.device.has_nvdec(),
            crypto_symmetric: false,
            crypto_asymmetric: false,
            compression: false,
            packet_processing: false,
            signal_processing: false,
        }
    }
    
    fn utilization(&self) -> AcceleratorUtilization {
        AcceleratorUtilization {
            compute_percent: self.device.get_utilization().compute,
            memory_percent: self.device.get_memory_usage().percent,
            queue_depth: self.stream.get_queue_depth(),
            temperature_celsius: self.device.get_temperature(),
            power_watts: self.device.get_power_usage(),
        }
    }
}
```

### Crypto Protocol Implementation

```rust
pub struct CryptoProtocol {
    qat_instance: Arc<QatInstance>,
    session_pool: Arc<SessionPool>,
}

impl AcceleratorProtocol for CryptoProtocol {
    async fn execute(&self, operation: Operation) -> Result<Vec<u8>> {
        match operation {
            Operation::Crypto(op) => self.execute_crypto(op).await,
            _ => Err("Unsupported operation".into()),
        }
    }
    
    async fn execute_crypto(&self, op: CryptoOperation) -> Result<Vec<u8>> {
        // Get session from pool
        let session = self.session_pool.acquire().await?;
        
        match op {
            CryptoOperation::Encrypt { algorithm, key, data } => {
                self.qat_instance.encrypt(&session, algorithm, &key, &data).await
            }
            CryptoOperation::Decrypt { algorithm, key, data } => {
                self.qat_instance.decrypt(&session, algorithm, &key, &data).await
            }
            CryptoOperation::Compress { algorithm, data } => {
                self.qat_instance.compress(&session, algorithm, &data).await
            }
            CryptoOperation::Sign { algorithm, key, data } => {
                self.qat_instance.sign(&session, algorithm, &key, &data).await
            }
        }
    }
    
    fn capabilities(&self) -> AcceleratorCapabilities {
        AcceleratorCapabilities {
            compute: false,
            ml_inference: false,
            ml_training: false,
            video_encode: false,
            video_decode: false,
            crypto_symmetric: true,
            crypto_asymmetric: true,
            compression: true,
            packet_processing: false,
            signal_processing: false,
        }
    }
}
```

---

## Worker Allocation Strategy

### NUMA-Aware Allocation

```rust
pub struct NumaAwareAllocator {
    topology: NumaTopology,
}

impl NumaAwareAllocator {
    pub fn allocate_workers(&self, config: &WorkerAllocationConfig) -> WorkerAllocation {
        let mut allocation = WorkerAllocation::default();
        
        // For each NUMA node
        for node in &self.topology.nodes {
            // Allocate GPU workers near GPUs
            let gpu_cores = node.get_cores_near_devices(&["gpu"]);
            for (i, core) in gpu_cores.iter().take(config.gpu_workers_per_node).enumerate() {
                allocation.gpu_workers.push(WorkerAssignment {
                    core_id: *core,
                    numa_node: node.id,
                    device_id: i,
                    affinity: CpuAffinity::Exclusive,
                });
            }
            
            // Allocate crypto workers near QAT
            let qat_cores = node.get_cores_near_devices(&["qat"]);
            for (i, core) in qat_cores.iter().take(config.crypto_workers_per_node).enumerate() {
                allocation.crypto_workers.push(WorkerAssignment {
                    core_id: *core,
                    numa_node: node.id,
                    device_id: i,
                    affinity: CpuAffinity::Exclusive,
                });
            }
            
            // Allocate network workers near NICs
            let nic_cores = node.get_cores_near_devices(&["network"]);
            for (i, core) in nic_cores.iter().take(config.network_workers_per_node).enumerate() {
                allocation.network_workers.push(WorkerAssignment {
                    core_id: *core,
                    numa_node: node.id,
                    device_id: i,
                    affinity: CpuAffinity::Exclusive,
                });
            }
            
            // Allocate application workers on remaining cores
            let app_cores: Vec<_> = node.cores.iter()
                .filter(|c| !gpu_cores.contains(c) 
                         && !qat_cores.contains(c)
                         && !nic_cores.contains(c))
                .collect();
            
            for core in app_cores {
                allocation.app_workers.push(WorkerAssignment {
                    core_id: *core,
                    numa_node: node.id,
                    device_id: 0,
                    affinity: CpuAffinity::Shared,
                });
            }
        }
        
        allocation
    }
}

pub struct WorkerAllocation {
    pub gpu_workers: Vec<WorkerAssignment>,
    pub crypto_workers: Vec<WorkerAssignment>,
    pub network_workers: Vec<WorkerAssignment>,
    pub ml_workers: Vec<WorkerAssignment>,
    pub signal_workers: Vec<WorkerAssignment>,
    pub video_workers: Vec<WorkerAssignment>,
    pub app_workers: Vec<WorkerAssignment>,
}

pub struct WorkerAssignment {
    pub core_id: usize,
    pub numa_node: usize,
    pub device_id: usize,
    pub affinity: CpuAffinity,
}

pub enum CpuAffinity {
    Exclusive,  // Dedicated core
    Shared,     // Can be shared
    Pinned,     // Pinned but allows interrupts
}
```

### Example Allocation (2-Socket, 64-Core Server)

```
NUMA Node 0 (Cores 0-31):
  GPU Workers (4):  Cores 0, 1, 2, 3    → GPU 0, 1
  Crypto Workers (2): Cores 4, 5        → QAT 0
  Network Workers (2): Cores 6, 7       → NIC 0
  ML Workers (2):   Cores 8, 9          → TPU 0 (if present)
  App Workers (22): Cores 10-31

NUMA Node 1 (Cores 32-63):
  GPU Workers (4):  Cores 32, 33, 34, 35 → GPU 2, 3
  Crypto Workers (2): Cores 36, 37       → QAT 1
  Network Workers (2): Cores 38, 39      → NIC 1
  ML Workers (2):   Cores 40, 41         → TPU 1 (if present)
  App Workers (22): Cores 42-63

Total:
  - GPU Workers: 8
  - Crypto Workers: 4
  - Network Workers: 4
  - ML Workers: 4
  - App Workers: 44
```

---

## Task Scheduling & Batching

### Batching Strategy

```rust
pub struct BatchScheduler {
    queues: HashMap<AcceleratorType, BatchQueue>,
    config: BatchConfig,
}

pub struct BatchConfig {
    pub max_batch_size: usize,
    pub max_wait_time: Duration,
    pub min_batch_size: usize,
    pub batch_strategy: BatchStrategy,
}

pub enum BatchStrategy {
    /// Wait until batch full or timeout
    TimeOrSize,
    
    /// Adaptive based on queue depth
    Adaptive,
    
    /// Prioritize latency (smaller batches)
    LowLatency,
    
    /// Prioritize throughput (larger batches)
    HighThroughput,
}

impl BatchScheduler {
    pub async fn schedule_task(&self, task: AcceleratorTask) -> Result<()> {
        let queue = self.queues.get(&task.accelerator_type)
            .ok_or("No queue for accelerator type")?;
        
        // Add to batch queue
        queue.push(task).await?;
        
        // Check if should flush
        if self.should_flush(queue)? {
            self.flush_batch(queue).await?;
        }
        
        Ok(())
    }
    
    fn should_flush(&self, queue: &BatchQueue) -> Result<bool> {
        match self.config.batch_strategy {
            BatchStrategy::TimeOrSize => {
                // Flush if full or timeout
                Ok(queue.len() >= self.config.max_batch_size
                   || queue.oldest_age() >= self.config.max_wait_time)
            }
            
            BatchStrategy::Adaptive => {
                // Adaptive based on system load
                let load = queue.get_accelerator_load();
                let batch_size = if load > 0.8 {
                    // High load - larger batches
                    self.config.max_batch_size
                } else {
                    // Low load - smaller batches for latency
                    self.config.min_batch_size
                };
                
                Ok(queue.len() >= batch_size
                   || queue.oldest_age() >= self.config.max_wait_time)
            }
            
            BatchStrategy::LowLatency => {
                // Flush quickly
                Ok(queue.len() >= self.config.min_batch_size
                   || queue.oldest_age() >= Duration::from_micros(100))
            }
            
            BatchStrategy::HighThroughput => {
                // Wait for full batches
                Ok(queue.len() >= self.config.max_batch_size)
            }
        }
    }
    
    async fn flush_batch(&self, queue: &BatchQueue) -> Result<()> {
        let tasks = queue.drain().await?;
        
        if tasks.is_empty() {
            return Ok(());
        }
        
        // Group by operation type
        let grouped = self.group_tasks(tasks);
        
        // Execute each group
        for (op_type, group_tasks) in grouped {
            self.execute_batch(op_type, group_tasks).await?;
        }
        
        Ok(())
    }
}
```

---

## Memory Management

### Zero-Copy Memory Management

```rust
pub struct AcceleratorMemoryManager {
    pinned_pools: HashMap<usize, PinnedMemoryPool>,
    device_pools: HashMap<usize, DeviceMemoryPool>,
    unified_memory: bool,
}

impl AcceleratorMemoryManager {
    pub fn allocate_pinned(&self, size: usize) -> Result<PinnedBuffer> {
        // Allocate page-locked host memory
        let pool = self.pinned_pools.get(&0)
            .ok_or("No pinned pool")?;
        
        pool.allocate(size)
    }
    
    pub fn allocate_device(&self, device_id: usize, size: usize) -> Result<DeviceBuffer> {
        let pool = self.device_pools.get(&device_id)
            .ok_or("No device pool")?;
        
        pool.allocate(size)
    }
    
    pub async fn transfer_h2d(
        &self,
        host: &PinnedBuffer,
        device: &mut DeviceBuffer,
    ) -> Result<()> {
        // Zero-copy transfer from pinned host to device
        device.copy_from_pinned(host).await
    }
    
    pub async fn transfer_d2h(
        &self,
        device: &DeviceBuffer,
        host: &mut PinnedBuffer,
    ) -> Result<()> {
        // Zero-copy transfer from device to pinned host
        host.copy_from_device(device).await
    }
    
    pub fn use_unified_memory(&self) -> bool {
        // CUDA Unified Memory or Metal Shared Memory
        self.unified_memory
    }
}

pub struct PinnedBuffer {
    ptr: *mut u8,
    size: usize,
    alignment: usize,
}

pub struct DeviceBuffer {
    device_id: usize,
    ptr: *mut u8,
    size: usize,
}
```

### Memory Pooling

```rust
pub struct MemoryPool {
    free_blocks: Vec<MemoryBlock>,
    allocated_blocks: HashMap<usize, MemoryBlock>,
    total_size: usize,
}

impl MemoryPool {
    pub fn allocate(&mut self, size: usize) -> Result<MemoryBlock> {
        // Find best-fit free block
        let idx = self.free_blocks.iter()
            .position(|b| b.size >= size && b.size < size * 2)
            .or_else(|| {
                // No perfect fit, find any block large enough
                self.free_blocks.iter()
                    .position(|b| b.size >= size)
            })
            .ok_or("Out of memory")?;
        
        let mut block = self.free_blocks.remove(idx);
        
        // Split if block is much larger
        if block.size > size * 2 {
            let remainder = MemoryBlock {
                ptr: unsafe { block.ptr.add(size) },
                size: block.size - size,
                alignment: block.alignment,
            };
            self.free_blocks.push(remainder);
            block.size = size;
        }
        
        let block_id = block.ptr as usize;
        self.allocated_blocks.insert(block_id, block.clone());
        
        Ok(block)
    }
    
    pub fn free(&mut self, block: MemoryBlock) {
        let block_id = block.ptr as usize;
        self.allocated_blocks.remove(&block_id);
        
        // Coalesce with adjacent free blocks
        self.free_blocks.push(block);
        self.coalesce();
    }
    
    fn coalesce(&mut self) {
        // Merge adjacent free blocks
        self.free_blocks.sort_by_key(|b| b.ptr as usize);
        
        let mut i = 0;
        while i < self.free_blocks.len() - 1 {
            let current_end = unsafe {
                self.free_blocks[i].ptr.add(self.free_blocks[i].size)
            };
            
            if current_end == self.free_blocks[i + 1].ptr {
                // Merge
                self.free_blocks[i].size += self.free_blocks[i + 1].size;
                self.free_blocks.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }
}
```

---

## Performance Optimization

### Kernel Fusion

```rust
pub struct KernelFusion {
    optimizer: KernelOptimizer,
}

impl KernelFusion {
    pub fn fuse_operations(&self, ops: Vec<GpuOperation>) -> Vec<FusedKernel> {
        // Analyze operation dependencies
        let dag = self.build_dependency_graph(&ops);
        
        // Find fusible chains
        let chains = self.find_fusible_chains(&dag);
        
        // Create fused kernels
        chains.iter()
            .map(|chain| self.create_fused_kernel(chain))
            .collect()
    }
    
    fn create_fused_kernel(&self, ops: &[GpuOperation]) -> FusedKernel {
        // Example: Fuse [MatMul, Add, ReLU] into single kernel
        FusedKernel {
            operations: ops.to_vec(),
            generated_code: self.generate_fused_code(ops),
            registers_needed: self.estimate_registers(ops),
            shared_memory_needed: self.estimate_shared_memory(ops),
        }
    }
}

// Example: Fused MatMul + Bias + ReLU
// Instead of 3 kernel launches:
//   result1 = matmul(A, B)
//   result2 = add(result1, bias)
//   result3 = relu(result2)
// 
// Single fused kernel:
//   result = relu(matmul(A, B) + bias)
//
// Benefits:
// - 3x fewer kernel launches
// - No intermediate memory allocation
// - Better instruction cache utilization
```

### Async Execution Overlap

```rust
pub struct AsyncExecutor {
    streams: Vec<CudaStream>,
    current_stream: AtomicUsize,
}

impl AsyncExecutor {
    pub async fn execute_with_overlap(&self, tasks: Vec<GpuTask>) {
        // Use multiple streams for concurrent execution
        let mut futures = Vec::new();
        
        for (i, task) in tasks.iter().enumerate() {
            let stream_id = i % self.streams.len();
            let stream = &self.streams[stream_id];
            
            let future = async move {
                // Transfer input (async)
                stream.copy_h2d_async(&task.input).await?;
                
                // Execute kernel (async)
                stream.launch_kernel(&task.kernel).await?;
                
                // Transfer output (async)
                stream.copy_d2h_async(&task.output).await?;
                
                Ok::<_, Error>(())
            };
            
            futures.push(future);
        }
        
        // All streams execute concurrently
        futures::future::join_all(futures).await;
    }
}

// Timeline with 2 streams:
//
// Stream 0: [H2D][Kernel][D2H]      [H2D][Kernel][D2H]
// Stream 1:           [H2D][Kernel][D2H]      [H2D][Kernel][D2H]
//
// Overlap achieves ~1.8x throughput vs sequential
```

---

## Integration Examples

### Example 1: ML Inference Service

```rust
pub struct MlInferenceService {
    app_workers: WorkerPool<AppWorker>,
    gpu_pool: Arc<GpuWorkerPool>,
    ml_pool: Arc<MlWorkerPool>,
    router: Arc<OpportunisticRouter>,
}

impl MlInferenceService {
    pub async fn infer(&self, request: InferenceRequest) -> Result<InferenceResponse> {
        // Application worker receives request
        // Preprocesses input
        let preprocessed = self.preprocess(request.image).await?;
        
        // Route to best accelerator (GPU or TPU)
        let task = AcceleratorTask {
            type_: TaskType::MlInference,
            data: preprocessed,
            model: request.model,
            batch_size: 1,
        };
        
        let result = self.router.execute_task(task).await?;
        
        // Postprocess on CPU
        let response = self.postprocess(result).await?;
        
        Ok(response)
    }
}

// Performance:
// - p50 latency: 2ms (GPU) / 5ms (TPU)
// - p99 latency: 5ms (GPU) / 8ms (TPU)
// - Throughput: 10,000 inferences/sec (4x GPU)
```

### Example 2: Video Transcoding Service

```rust
pub struct VideoTranscodingService {
    video_pool: Arc<VideoWorkerPool>,
    gpu_pool: Arc<GpuWorkerPool>,
}

impl VideoTranscodingService {
    pub async fn transcode(&self, input: VideoStream) -> Result<VideoStream> {
        // Create pipeline
        let pipeline = AcceleratorPipeline::new(vec![
            // Decode with hardware decoder
            PipelineStage::Video {
                pool: self.video_pool.clone(),
                operation: VideoOperation::Decode {
                    codec: VideoCodec::NvDec,
                },
            },
            
            // Process on GPU (denoise, sharpen, etc.)
            PipelineStage::Gpu {
                pool: self.gpu_pool.clone(),
                kernel: GpuKernel::VideoFilter {
                    filters: vec!["denoise", "sharpen"],
                },
            },
            
            // Encode with hardware encoder
            PipelineStage::Video {
                pool: self.video_pool.clone(),
                operation: VideoOperation::Encode {
                    codec: VideoCodec::NvEnc,
                    preset: "fast".to_string(),
                },
            },
        ]);
        
        // Stream through pipeline
        let output = pipeline.execute_streaming(input).await?;
        
        Ok(output)
    }
}

// Performance:
// - 4K60 transcoding: Real-time (single GPU)
// - 8K30 transcoding: Real-time (2x GPU)
// - Latency: <100ms glass-to-glass
```

### Example 3: Encrypted Storage Service

```rust
pub struct EncryptedStorageService {
    crypto_pool: Arc<CryptoWorkerPool>,
    storage_pool: Arc<StorageWorkerPool>,
}

impl EncryptedStorageService {
    pub async fn write(&self, data: Vec<u8>) -> Result<()> {
        // Create crypto pipeline
        let pipeline = AcceleratorPipeline::new(vec![
            // Compress with QAT
            PipelineStage::Crypto {
                pool: self.crypto_pool.clone(),
                operation: CryptoOperation::Compress {
                    algorithm: CompressionAlgorithm::Zstd,
                },
            },
            
            // Encrypt with QAT
            PipelineStage::Crypto {
                pool: self.crypto_pool.clone(),
                operation: CryptoOperation::Encrypt {
                    algorithm: EncryptionAlgorithm::AesGcm256,
                    key: self.get_encryption_key(),
                },
            },
        ]);
        
        // Process through pipeline
        let encrypted = pipeline.execute(data).await?;
        
        // Write to storage via storage workers
        self.storage_pool.write(encrypted).await?;
        
        Ok(())
    }
}

// Performance with QAT:
// - Compression: 20 GB/s (10x faster than CPU)
// - Encryption: 100 GB/s (50x faster than CPU)
// - Total: 15 GB/s end-to-end
```

---

## Summary

### Architecture Benefits

✅ **Specialized Pools**: Optimal resource allocation per accelerator type  
✅ **Pipeline Architecture**: Overlap CPU and GPU execution for maximum throughput  
✅ **Opportunistic**: Auto-detect hardware, graceful fallback to CPU  
✅ **Flexible**: Mix GPU, crypto, network, ML accelerators in one application  
✅ **NUMA-Aware**: Pin workers near their accelerators for low latency  
✅ **Zero-Copy**: Minimize data movement with pinned memory  
✅ **Batching**: Automatic batching for throughput optimization  
✅ **Composable**: Build complex pipelines from simple stages  

### Implementation Priority

**Phase 1** (Month 1-2):
- Accelerator discovery and capability detection
- GPU pool with CUDA/wgpu support
- Basic pipeline framework
- Opportunistic routing with CPU fallback

**Phase 2** (Month 3-4):
- Crypto pool with QAT integration
- Network pool with DPU/XDP
- Advanced batching strategies
- Memory pooling and zero-copy

**Phase 3** (Month 5-6):
- ML pool with TPU support
- Video pool with hardware codecs
- Kernel fusion optimization
- Multi-stream async execution

**Phase 4** (Month 7-8):
- Signal processing pool with FPGA
- Advanced pipeline patterns
- Performance profiling and tuning
- Production hardening

### Performance Targets

| Workload | Configuration | Latency | Throughput |
|----------|--------------|---------|------------|
| **ML Inference** | 4x GPU + TPU | <5ms p99 | 50K inf/sec |
| **Video Transcode** | 2x GPU + HW codec | <100ms | 8K30 real-time |
| **Crypto** | 2x QAT | <10μs | 100 GB/s |
| **Packet Processing** | 2x DPU + XDP | <1μs | 200 Gbps |
| **Signal Processing** | FPGA | <1μs | Deterministic |

---

**Document Version**: 1.0  
**Last Updated**: October 31, 2025  
**Status**: Design Complete - Ready for Implementation  
**Integration**: Extends PROTOCOL_LAYER.md with accelerator support

