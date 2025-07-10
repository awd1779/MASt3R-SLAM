from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the directory containing this setup.py
this_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name='semantic_fusion_cuda',
    ext_modules=[
        CUDAExtension(
            'semantic_fusion_cuda',
            sources=[
                os.path.join(this_dir, 'src/semantic_fusion_kernels.cu'),
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-gencode', 'arch=compute_70,code=sm_70',  # V100
                    '-gencode', 'arch=compute_75,code=sm_75',  # T4, RTX 2080
                    '-gencode', 'arch=compute_80,code=sm_80',  # A100
                    '-gencode', 'arch=compute_86,code=sm_86',  # RTX 3090
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)