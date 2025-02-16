from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='generic_winograd_conv',
    ext_modules=[
        CppExtension(
            name='generic_winograd_conv',
            sources=['winograd.cpp']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
