"""GPU profiling and performance monitoring utilities."""

import torch
import time
import psutil
from collections import defaultdict
from contextlib import contextmanager


class GPUProfiler:

    def __init__(self, device='cuda'):
        self.device = torch.device(device)
        self.stats = defaultdict(list)
        self.is_cuda = torch.cuda.is_available() and 'cuda' in str(device)

    def get_gpu_memory_stats(self):
        if not self.is_cuda:
            return {'allocated': 0, 'reserved': 0, 'free': 0}

        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        
        if hasattr(torch.cuda, 'mem_get_info'):
            free, total = torch.cuda.mem_get_info(self.device)
            free = free / 1e9
        else:
            free = 0
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'free': free
        }
    
    def get_cpu_memory_stats(self):
        process = psutil.Process()
        return {
            'rss': process.memory_info().rss / 1e9,
            'percent': process.memory_percent()
        }

    def log_memory_stats(self, logger=None, step=None):
        gpu_stats = self.get_gpu_memory_stats()
        cpu_stats = self.get_cpu_memory_stats()
        
        msg = f"GPU Memory: {gpu_stats['allocated']:.2f}GB allocated, {gpu_stats['reserved']:.2f}GB reserved"
        if gpu_stats['free'] > 0:
            msg += f", {gpu_stats['free']:.2f}GB free"
        msg += f" | CPU Memory: {cpu_stats['rss']:.2f}GB ({cpu_stats['percent']:.1f}%)"
        
        if step is not None:
            msg = f"Step {step} - " + msg
        
        if logger:
            logger.info(msg)
        else:
            print(msg)

        if step is not None:
            self.stats['gpu_allocated'].append((step, gpu_stats['allocated']))
            self.stats['gpu_reserved'].append((step, gpu_stats['reserved']))
            self.stats['cpu_rss'].append((step, cpu_stats['rss']))
    
    @contextmanager
    def profile_section(self, name, logger=None):
        if self.is_cuda:
            torch.cuda.synchronize()
        
        start_time = time.time()
        start_mem = self.get_gpu_memory_stats()
        
        yield
        
        if self.is_cuda:
            torch.cuda.synchronize()
        
        end_time = time.time()
        end_mem = self.get_gpu_memory_stats()
        
        elapsed = end_time - start_time
        mem_diff = end_mem['allocated'] - start_mem['allocated']
        
        msg = f"{name}: {elapsed:.3f}s, Memory delta: {mem_diff:+.2f}GB"
        
        if logger:
            logger.info(msg)
        else:
            print(msg)
        
        self.stats[name].append({
            'time': elapsed,
            'memory_delta': mem_diff
        })
    
    def get_summary(self):
        summary = {}
        
        for name, entries in self.stats.items():
            if isinstance(entries[0], dict):
                times = [e['time'] for e in entries]
                mem_deltas = [e['memory_delta'] for e in entries]
                
                summary[name] = {
                    'avg_time': sum(times) / len(times),
                    'total_time': sum(times),
                    'avg_memory_delta': sum(mem_deltas) / len(mem_deltas),
                    'calls': len(entries)
                }
        
        return summary
    
    def print_summary(self):
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("Profiling Summary")
        print("="*80)
        
        for name, stats in summary.items():
            print(f"\n{name}:")
            print(f"  Calls: {stats['calls']}")
            print(f"  Avg Time: {stats['avg_time']:.3f}s")
            print(f"  Total Time: {stats['total_time']:.3f}s")
            print(f"  Avg Memory Delta: {stats['avg_memory_delta']:+.2f}GB")
    
    def reset(self):
        self.stats.clear()


class PerformanceTimer:

    def __init__(self, name="Timer", use_cuda=True):
        self.name = name
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.start_time = None
        self.elapsed_time = 0

    def start(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        self.start_time = time.time()

    def stop(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        self.elapsed_time = time.time() - self.start_time
        return self.elapsed_time

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        elapsed = self.stop()
        print(f"{self.name}: {elapsed:.3f}s")


def get_model_size(model):
    param_size = 0
    param_count = 0
    buffer_size = 0
    buffer_count = 0
    
    for param in model.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size()
    
    for buffer in model.buffers():
        buffer_count += buffer.numel()
        buffer_size += buffer.numel() * buffer.element_size()
    
    total_size = param_size + buffer_size
    
    return {
        'total_params': param_count,
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_buffers': buffer_count,
        'param_size_mb': param_size / 1e6,
        'buffer_size_mb': buffer_size / 1e6,
        'total_size_mb': total_size / 1e6
    }


def print_model_summary(model, logger=None):
    size_info = get_model_size(model)
    
    msg = f"""
Model Summary:
{'='*60}
Total Parameters:     {size_info['total_params']:,}
Trainable Parameters: {size_info['trainable_params']:,}
Model Size:           {size_info['total_size_mb']:.2f} MB
{'='*60}
"""
    
    if logger:
        logger.info(msg)
    else:
        print(msg)


def estimate_batch_size(model, input_shape, device='cuda', max_memory_gb=8):
    if not torch.cuda.is_available() or device == 'cpu':
        return None
    
    model = model.to(device)
    model.eval()
    
    batch_size = 1
    max_batch_size = 1

    try:
        while True:
            x = torch.randn(batch_size, *input_shape, device=device)

            with torch.no_grad():
                _ = model(x)

            memory_used = torch.cuda.memory_allocated(device) / 1e9
            
            if memory_used > max_memory_gb:
                break
            
            max_batch_size = batch_size
            batch_size *= 2

            del x
            torch.cuda.empty_cache()
    
    except RuntimeError as e:
        if 'out of memory' in str(e):
            pass
        else:
            raise e
    
    finally:
        torch.cuda.empty_cache()
    
    return max_batch_size


def benchmark_model(model, input_shape, batch_size=32, num_iterations=100,
                   device='cuda', use_amp=False):
    model = model.to(device)
    model.eval()

    x = torch.randn(batch_size, *input_shape, device=device)
    for _ in range(10):
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model(x)
            else:
                _ = model(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            if use_amp:
                with torch.cuda.amp.autocast():
                    _ = model(x)
            else:
                _ = model(x)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    throughput = batch_size / avg_time

    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1e9
        memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    else:
        memory_allocated = 0
        memory_reserved = 0
    
    return {
        'avg_time_ms': avg_time * 1000,
        'throughput': throughput,
        'memory_allocated_gb': memory_allocated,
        'memory_reserved_gb': memory_reserved
    }

