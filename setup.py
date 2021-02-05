#!/usr/bin/env python

import os
import re
import sys
import sysconfig
import platform
import subprocess

from distutils.version import LooseVersion

try:
    from setuptools import setup, Extension, Command, find_packages
    from setuptools.command.install import install
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
except:
    from distutils import setup, Extension, Command, find_packages
    from distutils.command.install import install
    from Cython.Build import cythonize
    from Cython.Distutils import build_ext
    
try:
    from setuptools.command.build_clib import build_clib as _build_clib
except:
    from distutils.command.build_clib import build_clib as _build_clib

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class build_clib(_build_clib):
    def run(self):
        buildflag = True
        if buildflag:
            ext = CMakeExtension('molfile', sourcedir='pymolfile/molfile/'),
            self.my_build_extension(ext)

    def my_build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname('pymolfile/molfile'))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                cfg.upper(),
                extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j1']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', '../pymolfile/molfile'] + cmake_args,
                                cwd=self.build_temp + '/../', env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args,
                                cwd=self.build_temp + '/../')

        print()  # Add an empty line for cleaner output

def get_ext_filename_without_platform_suffix(filename):
    name, ext = os.path.splitext(filename)
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')

    if ext_suffix == ext:
        return filename

    ext_suffix = ext_suffix.replace(ext, '')
    idx = name.find(ext_suffix)

    if idx == -1:
        return filename
    else:
        return name[:idx] + ext

class BuildExtWithoutPlatformSuffix(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        return get_ext_filename_without_platform_suffix(filename)

def check_tcl_version():
    try:
        output = subprocess.check_output("echo \"puts \$tcl_version;exit 0\" | tclsh", shell=True)
        val = float(output)
    except subprocess.CalledProcessError:
        val = None
    return val

VERSION = "0.0.5"
CLASSIFIERS = [
    "Development Status :: 1 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: University of Illinois Open Source License (UIUC)",
    "Programming Language :: C",
    "Programming Language :: C++",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Physics",
    "Topic :: Scientific/Engineering :: Chemistry",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Material Science",
    "Operating System :: MacOS",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Operating System :: Microsoft :: Windows",
]

# from MDAnalysis setup.py (http://www.mdanalysis.org/)
class NumpyExtension(Extension, object):
    """Derived class to cleanly handle setup-time (numpy) dependencies.
    """
    # The only setup-time numpy dependency comes when setting up its
    #  include dir.
    # The actual numpy import and call can be delayed until after pip
    #  has figured it must install numpy.
    # This is accomplished by passing the get_numpy_include function
    #  as one of the include_dirs. This derived Extension class takes
    #  care of calling it when needed.
    def __init__(self, *args, **kwargs):
        self._np_include_dirs = []
        super(NumpyExtension, self).__init__(*args, **kwargs)

    @property
    def include_dirs(self):
        if not self._np_include_dirs:
            for item in self._np_include_dir_args:
                try:
                    self._np_include_dirs.append(item())  # The numpy callable
                except TypeError:
                    self._np_include_dirs.append(item)
        return self._np_include_dirs

    @include_dirs.setter
    def include_dirs(self, val):
        self._np_include_dir_args = val


# from MDAnalysis setup.py (http://www.mdanalysis.org/)
def get_numpy_include():
    try:
        # Obtain the numpy include directory. This logic works across numpy
        # versions.
        # setuptools forgets to unset numpy's setup flag and we get a crippled
        # version of it unless we do it ourselves.
        try:
            import __builtin__  # py2
            __builtin__.__NUMPY_SETUP__ = False
        except:
            import builtins  # py3
            builtins.__NUMPY_SETUP__ = False
        import numpy as np
    except ImportError as e:
        print(e)
        print('*** package "numpy" not found ***')
        print('pymolfile requires a version of NumPy, even for setup.')
        print('Please get it from http://numpy.scipy.org/ or install it through '
              'your package manager.')
        sys.exit(-1)
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include

# from SimpleTraj setup.py (https://github.com/arose/simpletraj)
if __name__ == '__main__':

    tcl_version = None
    tcl_version = check_tcl_version()
    if tcl_version is None:
        library_defs=['netcdf', 'expat']
    else:
        library_defs=['netcdf', 'expat', 'tcl']

#    if sys.version_info < (3, 0):
#        swig_opt_defs=['-Wall', '-c++']
#    else:
#        swig_opt_defs=['-py3', '-Wall', '-c++']
    swig_opt_defs=['-py3', '-Wall', '-c++']

    libpymolfile_module = Extension(
            'pymolfile/molfile/_libpymolfile', 
            sources=[
                'pymolfile/molfile/libpymolfile.i' , 
                'pymolfile/molfile/pymolfile.cxx',
                ],
            swig_opts=swig_opt_defs,
            library_dirs=[
                'build/external/tng/lib',
                'build/molfile_plugins/compile/lib/'
                ],
            libraries=library_defs,
            include_dirs = [
                get_numpy_include(),
                'pymolfile/molfile',
                'pymolfile/molfile/molfile_plugins/include',
                'pymolfile/molfile/molfile_plugins/molfile_plugin/include',
                'build/molfile_plugins/compile/lib/',
                'build/external/tng/include',
                ],
            extra_compile_args = [
                '-fPIC', '-shared', '-O0', '-g', '-w' 
                ],
            extra_link_args = [
                'build/molfile_plugins/compile/lib/libmolfile_plugin.a',
                'build/external/tng/lib/libtng_io.a',
                ],
            )

    setup(
        name = "pymolfile",
        author = "Berk Onat",
        author_email = "b.onat@warwick.ac.uk",
        description = "Not just a Python interface for VMD molfile plugins.",
        version = VERSION,
        classifiers = CLASSIFIERS,
        license = "UIUC",
        url = "https://gitlab.mpcdf.mpg.de/nomad-lab/pymolfile",
        zip_safe = False,
        packages = find_packages(),
        libraries = [('molfile_plugin', { 'sources' : ['build/molfile_plugins/compile/lib/libmolfile_plugin.a']})],
        cmdclass= {
            'build_clib' : build_clib,
            'build_ext'  : build_ext,
        },
        ext_modules = [
            libpymolfile_module,
        ],
        package_data={'pymolfile': ['LICENSE', 'pymolfile/molfile/libpymolfile.py']},
        py_modules=["pymolfile"],
        requires = [ "numpy" ],
        setup_requires = [ "numpy" ],
        install_requires = [ "numpy" ],
        extras_require = {}
    )
 

