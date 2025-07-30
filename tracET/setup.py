from setuptools import setup, Extension, find_packages
import os
import numpy

# Define the C extension
module1 = Extension(
    'supression',
    include_dirs=[
        numpy.get_include(),
        os.path.join(os.getcwd(), 'include'),
    ],
    sources=['cmodules/supressionmodule.c'],
)

# Main setup call
setup(
    name='tracET',
    version='2.2.2',
    description='A package that could transform a scalar map in a point cloud, trace a graph for filaments and compute the dice metric for two segmentations.',
    author='Pelayo Alvarez-Brecht, Antonio Martinez-Sanchez',
    packages=find_packages(),
    package_dir={'tracET': './tracET'},
    ext_modules=[module1],
    entry_points={
        'console_scripts': [
            'get_saliency = tracET.scripts.get_saliency:main',
            'apply_nonmaxsup = tracET.scripts.apply_nonmaxsup:main',
            'trace_graph = tracET.scripts.trace_graph:main',
            'get_cluster = tracET.scripts.Get_cluster:main',
            'membrane_poly = tracET.scripts.membrane_poly:main',
            'seg_skel_dice = tracET.scripts.seg_skel_dice:main',
        ],
    },
)
