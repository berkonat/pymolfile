import warnings
import sys

try:
    from .molfile import libpymolfile
except ImportError:
    warnings.warn("libpymolfile package not available, pymolfile does not work without its library!")

MAX_NUM_PLUGINS = 200
C_MOLFILE_PLUGINS = libpymolfile.molfile_plugin_list(MAX_NUM_PLUGINS)

def byte_str_decode(data, dectype=None):
    try:
        return data.decode(dectype).replace('\x00', '')
    except AttributeError:
        return data

def plugins():
    """ Information on the available molfile plugins

        Example tuple: ('psf', 'psf', 1, 1, 1, 0, 1, 1, 1, 0, 
            'CHARMM,NAMD,XPLOR PSF', 'mol file reader', 
            'Justin Gullingsrud, John Stone', 1, 9, 17, 1)

        The fields in the tuple represent info in ordered as follows:
            1: format extension
            2: format name
            3: read_structure is avaliable if 1
            4: read_bonds is avaliable if 1
            5: read_angles is avaliable if 1
            6: read_next_timestep is avaliable if 1
            7: write_structure is avaliable if 1
            8: write_bonds is avaliable if 1
            9: write_angles is avaliable if 1
           10: write_timestep is avaliable if 1
           11: long name of the plugin
           12: type of plugin
           13: authors of the plugin
           14: major version of the plugin
           15: minor version of the plugin
           16: ABI version of the plugin
           17: 1 if is reentrant (returns is_reentrant) 

    Returns: A list of tuples that includes the information and 
             capabilities of each molfile plugin. The information is 
             extracted from molfile_plugin_t. 
    """
    global C_MOLFILE_PLUGINS
    numlist = libpymolfile.molfile_init()
    if sys.version_info > (3,):
        basestring = str
    plugins_list = [
        [byte_str_decode(item, 
            dectype="unicode_escape") for item in libpymolfile.molfile_plugin_info(
            C_MOLFILE_PLUGINS, i)
            ] for i in range(numlist)
        ]
    libpymolfile.molfile_finish()
    return plugins_list

MOLFILE_PLUGINS = plugins()

