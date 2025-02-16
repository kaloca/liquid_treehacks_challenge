from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='fused_fftconv',
    ext_modules=[
        CppExtension(
            name='fused_fftconv',
            sources=['fftconv.cpp']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
