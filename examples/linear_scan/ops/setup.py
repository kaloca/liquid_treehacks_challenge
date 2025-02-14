from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='fused_linear_scan',
    ext_modules=[CppExtension(
        'fused_linear_scan',
        ['linear_scan.cpp'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
        }
    )

