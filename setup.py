import io
import os
import re
import glob
from setuptools import find_packages, setup

def get_extensions():
    try:
        import torch
        from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension, CUDA_HOME
        
        debug_mode = os.getenv("DEBUG", "0") == "1"
        use_cuda = os.getenv("USE_CUDA", "1") == "1" and torch.cuda.is_available() and CUDA_HOME is not None
        extension = CUDAExtension if use_cuda else CppExtension
        
        extra_compile_args = {
            "cxx": ["-O3" if not debug_mode else "-O0", "-fdiagnostics-color=always"],
            "nvcc": ["-O3" if not debug_mode else "-O0", "--extended-lambda"]
        }
        
        if debug_mode:
            extra_compile_args["cxx"].append("-g")
            extra_compile_args["nvcc"].append("-g")
        
        this_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        extensions_dir = os.path.join(this_dir, "cola_kernels", "csrc")
        sources = glob.glob(os.path.join(extensions_dir, "*.cpp"))
        cuda_sources = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
        
        if use_cuda:
            sources.extend(cuda_sources)
        
        ext_modules = [extension(
            "cola_kernels._C",
            sources,
            include_dirs=[extensions_dir],
            extra_compile_args=extra_compile_args,
            libraries=["cusolver", "cublas"] if use_cuda else [],
        )]
        return ext_modules, {"build_ext": BuildExtension}
    except ImportError as e:
        print(f"Warning: Failed to build extensions: {e}")
        return [], {}

ext_modules, cmdclass = get_extensions()

setup(
    name="cola-ml",
    packages=find_packages(include=['cola*', 'cola_kernels*']),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires='>=3.9',
    install_requires=['scipy', 'tqdm>=4.38', 'cola-plum-dispatch==0.1.4', 'optree', 'torch'],
    extras_require={'dev': ['pytest', 'pytest-cov', 'setuptools_scm', 'pre-commit']},
    description="",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Marc Finzi and Andres Potapczynski",
    author_email="maf820@nyu.edu",
    url='https://github.com/wilson-labs/cola',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=['linear algebra', 'linear ops', 'sparse', 'PDE', 'AI', 'ML'],
)