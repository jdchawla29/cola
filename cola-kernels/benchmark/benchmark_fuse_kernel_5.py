import time
import json
import numpy as np
import torch
import cola
from cola.utils.utils_for_tests import generate_pd_from_diag, generate_beta_spectrum
from cola.ops import Dense
from cola_kernels.ops import fuse_kernel_5
from pathlib import Path
import torch.cuda as cuda

MATRIX_TYPE_NAMES = {
    'pd_exp': 'PD Exponential Decay',
    'pd_clustered': 'PD Clustered Spectrum',
    'random': 'Random Dense',
    'sparse': 'Sparse (1% density)'
}

def create_test_matrices(size, dtype=torch.float32, device='cuda'):
    """Creates a suite of test matrices with different properties."""
    matrices = {}
    
    # Dense PD matrix with exponentially decaying spectrum
    spectrum = torch.exp(-torch.linspace(0, 5, size)).to(dtype=dtype)
    matrices['pd_exp'] = torch.tensor(
        generate_pd_from_diag(spectrum.cpu().numpy(), dtype=np.float32), 
        dtype=dtype
    ).to(device)
    
    # Dense PD matrix with clustered spectrum
    spectrum = torch.tensor(
        generate_beta_spectrum(2, 1.0, size, dtype=np.float32), 
        dtype=dtype
    )
    matrices['pd_clustered'] = torch.tensor(
        generate_pd_from_diag(spectrum.cpu().numpy(), dtype=np.float32),
        dtype=dtype
    ).to(device)
    
    # Dense random matrix
    matrices['random'] = torch.randn(size, size, dtype=dtype, device=device)
    
    # Sparse matrix (~1% density)
    sparse = torch.zeros(size, size, dtype=dtype, device=device)
    nnz = int(0.01 * size * size)
    rows = torch.randint(0, size, (nnz,), device=device)
    cols = torch.randint(0, size, (nnz,), device=device)
    values = torch.randn(nnz, dtype=dtype, device=device)
    sparse[rows, cols] = values
    matrices['sparse'] = sparse
    
    return matrices

def find_optimal_batch_size(matrix_size):
    """Determine optimal batch size based on matrix size."""
    if matrix_size <= 1000:
        return 100
    elif matrix_size <= 2000:
        return 64
    elif matrix_size <= 5000:
        return 32
    else:
        return 16

def benchmark_diagonal_estimation(matrix, methods, device, n_runs=5):
    """Benchmarks different diagonal estimation methods."""
    results = {}
    true_diag = torch.diag(matrix).clone()
    
    for method_name, method_config in methods.items():
        times = []
        errors = []
        peak_memory = []
        diagonals = []
        
        # Pre-allocate output tensor for CUDA kernel
        if method_name.startswith('cuda_kernel'):
            est_diag = torch.zeros_like(true_diag)
        
        # Empty CUDA cache before testing each method
        if device == 'cuda':
            torch.cuda.empty_cache()
            
        for run in range(n_runs):
            try:
                if device == 'cuda':
                    torch.cuda.synchronize()
                    start_mem = torch.cuda.memory_allocated()
                
                if method_name.startswith('cuda_kernel'):
                    # Zero out est_diag before timing
                    est_diag.zero_()
                    torch.cuda.synchronize()
                    start_time = time.time()
                    diag = fuse_kernel_5(
                        matrix, est_diag,
                        batch_size=method_config['bs'],
                        tolerance=method_config['tol'],
                        max_iter=method_config.get('max_iter', 50),
                        use_rademacher=method_config.get('use_rademacher', False)
                    )
                else:  # Cola implementation
                    torch.cuda.synchronize()
                    start_time = time.time()
                    diag = cola.diag(Dense(matrix), alg=method_config['alg'])
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                    peak_mem = torch.cuda.max_memory_allocated() - start_mem
                    peak_memory.append(float(peak_mem))
                
                end_time = time.time()
                
                # Convert to CPU for calculations if needed
                if device == 'cuda':
                    diag = diag.cpu()
                    true_diag_cpu = true_diag.cpu()
                else:
                    true_diag_cpu = true_diag
                
                times.append(float(end_time - start_time))
                rel_error = float(torch.norm(diag - true_diag_cpu) / torch.norm(true_diag_cpu))
                errors.append(rel_error)
                diagonals.append(diag.tolist())
                
            except RuntimeError as e:
                print(f"Error in {method_name} (run {run}): {str(e)}")
                if run == 0:  # If first run fails, skip this method
                    print(f"Skipping method {method_name} due to error")
                    break
                continue
        
        if len(times) > 0:  # Only save results if at least one run succeeded
            results[method_name] = {
                'times': times,
                'errors': errors,
                'diagonals': diagonals,
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'mean_error': float(np.mean(errors)),
                'std_error': float(np.std(errors)),
                'successful_runs': len(times)
            }
            
            if device == 'cuda':
                results[method_name]['peak_memory'] = peak_memory
                results[method_name]['mean_peak_memory'] = float(np.mean(peak_memory))
                results[method_name]['std_peak_memory'] = float(np.std(peak_memory))
    
    return results

def run_hutch_benchmark(sizes, output_dir='hutch_benchmark_results'):
    """Run comprehensive benchmark of Hutchinson implementations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32
    
    # Results dictionary to store timing information for methods that fail
    timing_info = {}
    
    # Run benchmarks
    all_results = {}
    
    for size in sizes:
        print(f"\nTesting size {size}...")
        all_results[size] = {}
        
        # Determine batch size for this matrix size
        bs = find_optimal_batch_size(size)
        print(f"Using batch size: {bs}")
        
        # Configure methods to test
        methods = {
            'cola_hutch_01': {
                'alg': cola.linalg.Hutch(tol=0.1, bs=bs)
            },
            'cola_hutch_001': {
                'alg': cola.linalg.Hutch(tol=0.01, bs=bs)
            },
            'cuda_kernel_01': {
                'tol': 0.1,
                'bs': bs,
                'max_iter': 1000
            },
            'cuda_kernel_001': {
                'tol': 0.01,
                'bs': bs,
                'max_iter': 1000
            },
            'cuda_kernel_01_rad': {
                'tol': 0.1,
                'bs': bs,
                'max_iter': 1000,
                'use_rademacher': True
            },
            'cuda_kernel_001_rad': {
                'tol': 0.01,
                'bs': bs,
                'max_iter': 1000,
                'use_rademacher': True
            }
        }
        
        matrices = create_test_matrices(size, dtype=dtype, device=device)
        for matrix_type, matrix in matrices.items():
            print(f"  Testing {matrix_type}...")
            results = benchmark_diagonal_estimation(matrix, methods, device)
            all_results[size][matrix_type] = results
            
            # Store timing information
            timing_info[f"{size}_{matrix_type}"] = {
                method: result.get('mean_time', float('nan')) 
                for method, result in results.items()
            }
    
    # Save configuration
    config = {
        'sizes': sizes,
        'device': device,
        'dtype': str(dtype),
        'matrix_types': list(MATRIX_TYPE_NAMES.keys()),
        'matrix_type_names': MATRIX_TYPE_NAMES,
        'timing_info': timing_info,
        'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S'),
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save results
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nResults saved in {output_dir}")
    return all_results, config

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.generic, np.ndarray)):
            return obj.tolist()
        return super().default(obj)

if __name__ == '__main__':
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test matrices of different sizes
    sizes = [1000, 2000, 5000, 10000]
    
    # Run benchmark
    run_hutch_benchmark(sizes)