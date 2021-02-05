
from __future__ import absolute_import
from __future__ import print_function

import os
import re
import sys
import numpy as np
from contextlib import contextmanager
import warnings

if sys.version_info > (3,):
    long = int

try:
    from .molfile import libpymolfile
except ImportError:
    warnings.warn("libpymolfile package not available, pymolfile does not work without its library!")

from .plugin_list import plugins, byte_str_decode, MOLFILE_PLUGINS, C_MOLFILE_PLUGINS

def list_plugins():
    global MOLFILE_PLUGINS
    return MOLFILE_PLUGINS[:]

def decode_array(inarray):
    return np.array([
            [byte_str_decode(element, 
                dectype="unicode_escape") for element in item 
                ] for item in inarray
            ])

# The function stdout_redirected is taken from 
# https://stackoverflow.com/questions/5081657/how-do-i-prevent-a-c-shared-library-to-print-on-stdout-in-python
# to supress the plugins from printing to standard output (stdout)

@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

def get_dir_base_extension(file_name):
    """ Splits directory, file base and file extensions

        Returns: directory without leading '/', 
                 file base name, and file extension without '.'
    """
    file_base, file_extension_with_dot = os.path.splitext(os.path.basename(file_name))
    file_extension = file_extension_with_dot.split(".")[-1]
    file_dir = os.path.dirname(file_name)
    return file_dir, file_base, file_extension

def get_listof_parts( filename, filepath, fileformats ):
    pattern = re.escape(filename[1:-4]) + "\.part[0-9]{4,4}\.(xtc|trr)$"
    parts = []
    for f in os.listdir(directory):
        m = re.match(pattern, f)
        if m and os.path.isfile(os.path.join(directory, f)):
            parts.append(os.path.join(directory, f))
    return sorted(parts)

def get_extension(file_name):
    """ Gets file extension of a file

        Returns: file extension without '.'
    """
    file_extension_with_dot  = os.path.splitext(os.path.basename(file_name))[1]
    return file_extension_with_dot.split(".")[-1]

def get_plugin_with_ext(file_ext):
    """ Search molfile plugins list and returns the plugin info 
        for the first matching extension.

        Returns: Plugin no in the list and the list item (the plugin info tuple)
    """
    global MOLFILE_PLUGINS

    if not MOLFILE_PLUGINS:
        MOLFILE_PLUGINS = plugins()

    if MOLFILE_PLUGINS:
        plugin_no = -1
        for plugin_info in MOLFILE_PLUGINS:
            plugin_no += 1
            plugin_ext_info = str(plugin_info[0]).replace('*|', 
                    '').replace('*', '').replace('|', ',').replace(' ', '')
            for ext_item in str(plugin_ext_info).split(','):
                if str(file_ext) == str(ext_item):
                    return (plugin_no, plugin_info)

    return None

def plugins_same(plugin_one, plugin_two):
    if((plugin_one[0] == plugin_two[0]) and 
       (plugin_one[1][1] == plugin_two[1][1])):
        return True
    else:
        return False

def get_plugin_with_name(plugin_name):
    """ Search molfile plugins list and returns the plugin info 
        for the first matching name in plugin name field.

        Returns: Plugin no in the list and the list item (the plugin info tuple)
    """
    global MOLFILE_PLUGINS

    if not MOLFILE_PLUGINS:
        MOLFILE_PLUGINS = plugins()

    if MOLFILE_PLUGINS:
        plugin_no = -1
        for plugin_info in MOLFILE_PLUGINS:
            plugin_no += 1
            if(str(plugin_name) in str(plugin_info[1]) or 
               str(plugin_name) in str(plugin_info[10])):
                return (plugin_no, plugin_info)

    return None

class Topology(object):

    def __init__(self, fplugin, fname, ftype, natoms, silent):
        self.natoms = None
        self.structure = None
        self.bonds = None
        self.angles = None
        self._structure = None
        self._bonds = None
        self._angles = None
        self.pluginhandle = None
        self.silent = silent
        try:
            if self.silent:
                with stdout_redirected():
                    self.pluginhandle = libpymolfile.open_file_read(fplugin, 
                        fname, ftype, natoms)
            else:
                self.pluginhandle = libpymolfile.open_file_read(fplugin, 
                    fname, ftype, natoms)
            self.natoms = self.pluginhandle.natoms
        except (IOError, OSError, AttributeError):
            pass

    def read_structure(self, prototype):
        try:
            if self.silent:
                with stdout_redirected():
                    self._structure = libpymolfile.read_fill_structure(
                            self.pluginhandle, prototype)
            else:
                self._structure = libpymolfile.read_fill_structure(
                        self.pluginhandle, prototype)
            return self._structure
        except (OSError, IOError, AttributeError, SystemError):
            return None

    def read_bonds(self):
        try:
            if self.silent:
                with stdout_redirected():
                    self._bonds = libpymolfile.read_fill_bonds(self.pluginhandle)
            else:
                self._bonds = libpymolfile.read_fill_bonds(self.pluginhandle)
            return self._bonds
        except (OSError, IOError, AttributeError, SystemError):
            return None

    def read_angles(self):
        try:
            if self.silent:
                with stdout_redirected():
                    self._angles = libpymolfile.read_fill_angles(self.pluginhandle)
            else:
                self._angles = libpymolfile.read_fill_angles(self.pluginhandle)
            return self._angles
        except (OSError, IOError, AttributeError, SystemError):
            return None

    def __del__( self ):
        pass
        #if self.pluginhandle is not None:
        #    libpymolfile.close_file_read(self.pluginhandle)

class Trajectory(object):
    
    def __init__(self, file_name, file_format, plugin, natoms, silent):
        global MOLFILE_PLUGINS
        global C_MOLFILE_PLUGINS
        self.natoms = None
        self.atoms = None
        self.plugin = None
        self.handle = None
        self.fname = None
        self.ftype = None
        self.fname = file_name
        self.ftype = file_format
        self.plugin = plugin
        self.natoms = 0
        self.silent = silent

        if natoms is not None:
            self.natoms = natoms

        numlist = libpymolfile.molfile_init()
        self.fplugin = libpymolfile.get_plugin(C_MOLFILE_PLUGINS, 
                                          self.plugin[0])
        self.pluginhandle = None

        if self.natoms > 0:
            try:
                if self.silent:
                    with stdout_redirected():
                        self.pluginhandle = libpymolfile.open_file_read(self.fplugin, 
                            self.fname, self.ftype, self.natoms)
                else:
                    self.pluginhandle = libpymolfile.open_file_read(self.fplugin, 
                        self.fname, self.ftype, self.natoms)
            except (IOError, OSError, AttributeError):
                pass
                #if self.pluginhandle is not None:
                #    libpymolfile.close_file_read(self.pluginhandle)

    def read_next(self):
        return libpymolfile.read_fill_next_timestep(self.pluginhandle)

    def iter_on_traj(self):
        try:
            empty = False
            self.step = 0
            while not empty:
                x = self.read_next()
                if x is None:
                    empty = True
                else:
                    self.step += 1
                    yield x
        except (IOError, AttributeError, OSError):
            return

    def iread(self):
        iter_obj = iter(self.iter_on_traj())
        try:
            while True:
                self.atoms = next(iter_obj)
                return self.atoms
        except StopIteration:
            pass
        finally:
            del iter_obj

#    def read_frame(self, index_no):
#        numlist = libpymolfile.molfile_init()
#        tplugin = libpymolfile.get_plugin(C_MOLFILE_PLUGINS, 
#                                          self.plugin[1])
#        pluginhandle = libpymolfile.open_file_read(tplugin, 
#            file_name, file_format, natoms)
#        libpymolfile.molfile_finish()

#    def read(self, index_list, chunk=None):
#        numlist = libpymolfile.molfile_init()
#        tplugin = libpymolfile.get_plugin(C_MOLFILE_PLUGINS, 
#                                          self.plugin[1])
#        pluginhandle = libpymolfile.open_file_read(tplugin, 
#            file_name, file_format, natoms)
#        libpymolfile.molfile_finish()

    def __del__(self):
        pass
        #if self.pluginhandle is not None:
        #    libpymolfile.close_file_read(self.pluginhandle)

def read_topology(file_name, file_format, plugin, silent):
    """ Reads structure, bonds, angles, dihedrals, impropers and 
        additional informations through molfile_plugin if the 
        data is available
     
        The topology data in pymolfile is a list of tuples that include
        the following fields:
            (name, type, resname, resid, segid, chain, altloc, insertion, 
             occupancy, bfactor, mass, charge, radius, atomicnumber)
            
        The data types are as follows in the tuple:
            s = string, i = integer, f = float
            (s, s, s, i, s, s, s, s, f, f, f, f, f, i)

        Returns: Topology class if at least one of the topology
                 data is available. If non of the data is accessible,
                 function returns None
    """
    topo = None
    structure = None
    bonds = None
    angles = None
    if(file_name is not None and
       file_format is not None and 
       plugin is not None):
        natoms=0
        topo = Topology(plugin, file_name, file_format, natoms, silent)
        #if 0 && vmdplugin_ABIVERSION > 17
        #  /* The new PDB file formats allows for much larger structures, */
        #  /* which can therefore require longer chain ID strings.  The   */
        #  /* new PDBx/mmCIF file formats do not have length limits on    */
        #  /* fields, so PDB chains could be arbitrarily long strings     */
        #  /* in such files.  At present, we know we need at least 3-char */
        #  /* chains for existing PDBx/mmCIF files.                       */
        #  char chain[4];      /**< required chain name, or ""            */
        #else
        #  char chain[2];      /**< required chain name, or ""            */
        #endif
        #
        # Change 'chain', S2 to S4
        #
        #if('pdb' in file_format or
        #   'psf' in file_format):
        chain_size = 'S4'
        #else:
        #    chain_size = 'S2'
        if('dtr' in file_format or 
           'stk' in file_format or 
           'atr' in file_format):
            prototype = np.array([
                ('','','',0,'','','','',1.0,1.0,1.0,1.0,1.0,6,0), 
                ('','','',0,'','','','',1.0,1.0,1.0,1.0,1.0,6,0)
                ],
                dtype=[
                    ('name', 'S16'), ('type', 'S16'), ('resname', 'S8'),
                    ('resid', 'i4'), ('segid', 'S8'), ('chain', chain_size),
                    ('altloc', 'S2'), ('insertion', 'S2'), ('occupancy', 'f4'),
                    ('bfactor', 'f4'), ('mass', 'f4'), ('charge', 'f4'),
                    ('radius', 'f4'), ('atomicnumber', 'i4'), ('ctnumber', 'i4')
                    ]
                )
        else:
            prototype = np.array([
                ('','','',0,'','','','',1.0,1.0,1.0,1.0,1.0,6),
                ('','','',0,'','','','',1.0,1.0,1.0,1.0,1.0,6)
                ],
                dtype=[
                    ('name', 'S16'), ('type', 'S16'), ('resname', 'S8'),
                    ('resid', 'i4'), ('segid', 'S8'), ('chain', chain_size),
                    ('altloc', 'S2'), ('insertion', 'S2'), ('occupancy', 'f4'),
                    ('bfactor', 'f4'), ('mass', 'f4'), ('charge', 'f4'),
                    ('radius', 'f4'), ('atomicnumber', 'i4')
                    ]
                )

        if topo.read_structure(prototype) is not None:
            topo.structure = decode_array(topo._structure)
        if topo.read_bonds() is not None:
            topo.bonds = topo._bonds
        if topo.read_angles() is not None:
            topo.angles = topo._angles

    if(topo.structure is not None or 
       topo.bonds is not None or
       topo.angles is not None):
        return topo
    else:
        del topo
        return None

class OpenMolfile(object):
    """ The main class/function to read topology and 
        trajectory files

        Returns: Depends on the file format and arguments:
                 If structure file is supplied, it returns topology class.
                 If only trajectory file is supplied, it returns trajectory class 
                     without topology information. (the number of atoms must be known)
                 If both files are supplied, it returns trajectory class with 
                     topology information.
                 If file has both topology and trajectory info, both info will be 
                     available in the class
                 If none of the above is available, the topology and 
                     trajectory will be None in the return object
    """

    def __init__(self, *args, **kwargs):
        self.initialize_settings()
        #super(OpenMolfile, self).__init__(*args, **kwargs)
        global MOLFILE_PLUGINS
        global C_MOLFILE_PLUGINS
        file_name = None

        if args:
            for arg in args:
                if isinstance(arg, dict):
                    self.reset_kwords()
                    for k, v in arg.items():
                        if k in self.kwords:
                            self.kwords[k] = v
                elif isinstance(arg, (list, tuple)):
                    self.kwords["file_name"] = arg[0]
                elif isinstance(arg, str):
                    self.kwords["file_name"] = arg
                else:
                    self.kwords["file_name"] = arg

        if kwargs:
            for k,v in kwargs.items():
                if k in self.kwords:
                    self.kwords[k] = v

        if self.kwords["file_name"] is not None:
            file_name = self.kwords["file_name"]

        if file_name:
            if self.kwords["file_format"] is None:
               file_dir, file_base, file_ext = get_dir_base_extension(file_name)
               if file_ext:
                   self.kwords["file_format"] = file_ext
               else:
                   self.kwords["file_format"] = file_base

            if self.kwords["file_plugin"] is None:
                self.kwords["file_plugin"] = "auto"

            if "auto" in self.kwords["file_plugin"]:
                plugin_item = get_plugin_with_ext(self.kwords["file_format"])
                if plugin_item:
                    self.fplugin = plugin_item
                    self.kwords["file_plugin"] = plugin_item[1][1]
                else:
                    self.kwords["file_plugin"] = None
             
            if self.kwords["file_plugin"]:
                # Check if file_plugin reads structure info 
                # for the given file format.
                if self.fplugin[1][2] == 1:
                    # Topology may be read with the plugin.
                    # This will override topology information 
                    # if a 'topology' is supplied in keywords.
                    if self.kwords["file_format"] is not None:
                        numlist = libpymolfile.molfile_init()
                        self.smolplugin = libpymolfile.get_plugin(C_MOLFILE_PLUGINS, 
                                                                  self.fplugin[0])
                        self.topology = read_topology(file_name, 
                                                      self.kwords["file_format"], 
                                                      self.smolplugin,
                                                      self.kwords["silent"])
                        #libpymolfile.molfile_finish()

                if(self.fplugin[1][2] == 0 or 
                   self.topology is None):
                    # Topology can not be read with plugin from 
                    # the file. If 'topology' is not set but 'natoms' 
                    # is set, only the trajectory will be available.
                    if self.kwords["topology"] is not None:
                        if isinstance(self.kwords["topology"], OpenMolfile):
                            self.topology = self.kwords["topology"].topology
                        elif isinstance(self.kwords["topology"], Topology):
                            self.topology = self.kwords["topology"]
                        else:
                            topo_file_name = self.kwords["topology"]
                            if self.kwords["topology_format"] is None:
                               topo_file_dir, topo_file_base, topo_file_ext = get_dir_base_extension(
                                       topo_file_name)
                               if topo_file_ext:
                                   self.kwords["topology_format"] = topo_file_ext
                               else:
                                   self.kwords["topology_format"] = topo_file_base

                            if self.kwords["topology_plugin"] is None:
                                self.kwords["topology_plugin"] = "auto"

                            if "auto" in self.kwords["topology_plugin"]:
                                topo_plugin_item = get_plugin_with_ext(self.kwords["topology_format"])
                                if topo_plugin_item:
                                    self.tplugin = topo_plugin_item
                                    self.kwords["topology_plugin"] = topo_plugin_item[1][1]
                                else:
                                    self.kwords["topology_plugin"] = None

                            if self.kwords["topology_plugin"]:
                            # Check if topology_plugin reads structure info 
                            # for the given file format.
                                if self.tplugin[1][2] == 1:
                                # Topology may be read with the plugin.
                                # This will override the topology information. 
                                    if self.kwords["topology_format"] is not None:
                                        numlist = libpymolfile.molfile_init()
                                        self.smolplugin = libpymolfile.get_plugin(C_MOLFILE_PLUGINS, 
                                                                                  self.fplugin[0])
                                        self.topology = read_topology(topo_file_name, 
                                                                      self.kwords["topology_format"], 
                                                                      self.smolplugin,
                                                                      self.kwords["silent"])
                                        #libpymolfile.molfile_finish()
                            else:
                                warnings.warn("Pymolfile can not find a plugin to open the '" + 
                                      self.kwords["topology_format"] + "' file format of the file " + 
                                      self.kwords["topology"])

                if self.fplugin[1][5] == 1:
                    num_atoms = 0
                    if self.topology:
                        num_atoms = self.topology.natoms
                    elif self.kwords["natoms"] is not None:
                        num_atoms = int(self.kwords["natoms"])
                    # Trajectory can be read if num_atoms is set
                    if num_atoms>0:
                        self.trajectory = Trajectory(
                                self.kwords["file_name"],
                                self.kwords["file_format"],
                                self.fplugin,
                                num_atoms,
                                self.kwords["silent"])
            else:
                warnings.warn("Pymolfile can not find a plugin to open the '" + self.kwords["file_format"] + 
                      "' file format of the file " + self.kwords["file_name"]) 

    def initialize_settings(self):
        #global MOLFILE_PLUGINS
        #global C_MOLFILE_PLUGINS
        #self.plugin_list = MOLFILE_PLUGINS
        #self.c_plugin_list = C_MOLFILE_PLUGINS
        self.trajectory = None
        self.topology = None
        self.fplugin = None
        self.tplugin = None
        self.smolplugin = None
        self.cmolplugin = None
        self.kwords = { 
            "file_name" : None,
            "file_format" : None,
            "file_plugin" : None,
            "topology" : None,
            "topology_format" : None,
            "topology_plugin" : None,
            "natoms" : None,
            "silent" : False
            }

        #if not C_MOLFILE_PLUGINS:
        #    C_MOLFILE_PLUGINS = libpymolfile.molfile_plugin_list(MAX_NUM_PLUGINS)
        #if not MOLFILE_PLUGINS:
        #    MOLFILE_PLUGINS = plugins()
        #if(MOLFILE_PLUGINS and self.plugin_list is None):
        #    self.plugin_list = MOLFILE_PLUGINS
        #if(C_MOLFILE_PLUGINS and self.c_plugin_list is None):
        #    self.c_plugin_list = C_MOLFILE_PLUGINS

    def reset_kwords(self):
        self.kwords.update({ 
            "file_name" : None,
            "file_format" : None,
            "file_plugin" : None,
            "topology" : None,
            "topology_format" : None,
            "topology_plugin" : None,
            "natoms" : None,
            "silent" : False
            })

    def __del__(self):
        #del self.topology
        #del self.trajectory
        libpymolfile.molfile_finish()

