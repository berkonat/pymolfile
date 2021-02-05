/* Hey emacs this is -*- C -*- and this is my editor vim.
 * 
 * pymolfile.c : C and Python interfaces for molfile_plugins
 * Copyright (c) Berk Onat <b.onat@warwick.ac.uk> 2017
 *
 * This program is under UIUC Open Source License please see LICENSE file
 */

/*
 * The code is written following the plugin test 
 * context of main.c in molfile_plugin/src/ 
 */

/* Get HAVE_CONFIG_H */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <ctype.h>

/* get fixed-width types if we are using ANSI C99 */
#ifdef HAVE_STDINT_H
#  include <stdint.h>
#elif (defined HAVE_INTTYPES_H)
#  include <inttypes.h>
#endif

#include "pymolfile.h"

int numplugins=0;
molfile_plugin_t** plugin_list;

#if PY_VERSION_HEX >= 0x03000000
#define PyInt_AsLong PyLong_AsLong
#define PyString_FromString PyBytes_FromString
#define PyInt_FromLong PyLong_FromLong

void del_molfile_plugin_list(PyObject* molcapsule)
{
    plugin_list = (molfile_plugin_t**) PyMolfileCapsule_AsVoidPtr(molcapsule);   
    free(plugin_list); 
    Py_XDECREF(molcapsule);
}
void del_molfile_file_handle(PyObject* molcapsule)
{
    void *file_handle = (void*) PyMolfileCapsule_AsVoidPtr(molcapsule);   
    free(file_handle); 
    Py_XDECREF(molcapsule);
}
#else
void del_molfile_plugin_list(void* molcapsule)
{
    plugin_list = (molfile_plugin_t**) PyMolfileCapsule_AsVoidPtr((PyObject*)molcapsule);   
    free(plugin_list); 
    Py_XDECREF(molcapsule);
}
void del_molfile_file_handle(void* molcapsule)
{
    void *file_handle = PyMolfileCapsule_AsVoidPtr((PyObject*)molcapsule);   
    free(file_handle); 
    Py_XDECREF(molcapsule);
}
#endif

/* * * * * * * * * * * * * * * * * * * * * * *
 * Helper functions to set and store plugins *
 * * * * * * * * * * * * * * * * * * * * * * */

PyObject* get_plugin(PyObject* molcapsule, int plug_no)
{
    molfile_plugin_t* plugin;
    molfile_plugin_t** plug_list = (molfile_plugin_t**) PyMolfileCapsule_AsVoidPtr(molcapsule);   
    if(plug_no < 0){
	plugin = NULL;
    } else {
	if(plug_list != NULL){
	    plugin = plug_list[plug_no];
	} else {
	    plugin = NULL;
	}
    }
    return (PyObject*) PyMolfileCapsule_FromVoidPtr((void *)plugin, NULL);
}

/* * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Interface functions to initialize molfile plugins *
 * * * * * * * * * * * * * * * * * * * * * * * * * * */

/* check validity of plugins and register them. */
static int molfile_register(void*, vmdplugin_t *plugin) {
    if (!plugin->type || !plugin->name || !plugin->author) {
        // skipping plugin with incomplete header
        return VMDPLUGIN_ERROR;
    }
    else if (plugin->abiversion != vmdplugin_ABIVERSION) {
        // skipping plugin with incompatible ABI
        return VMDPLUGIN_ERROR;
    }
    else if (0 != strncmp(plugin->type, "mol file", 8)) {
        // skipping plugin of incompatible type
        return VMDPLUGIN_ERROR;
    } 
    else if (numplugins >= MAXPLUGINS) {
        // too many plugins: increase MAXPLUGINS
        return VMDPLUGIN_ERROR;
    }

        plugin_list[numplugins] = (molfile_plugin_t *) plugin;
        ++numplugins;
        return VMDPLUGIN_SUCCESS;
}

PyObject* molfile_plugin_list(int maxsize)
{
    if(maxsize < MAXPLUGINS){
        maxsize = MAXPLUGINS;
    }
    plugin_list = (molfile_plugin_t**) malloc(sizeof(molfile_plugin_t*)*maxsize);
    return PyMolfileCapsule_FromVoidPtr((void *)plugin_list, del_molfile_plugin_list);
}

/* register all available plugins and clear handles. */
int molfile_init(void) 
{
    MOLFILE_INIT_ALL;
    MOLFILE_REGISTER_ALL(NULL,molfile_register);
    return numplugins;
}

/* unregister all available plugins */
int molfile_finish(void) 
{
    MOLFILE_FINI_ALL;
    return 0;
}

/* * * * * * * * * * * * * * * * * * * * * * *
 * Wrappers to directly access molfile plugin*
 *         functions and settings            *
 * * * * * * * * * * * * * * * * * * * * * * */

/* molfile_plugin_t access */

/* Functions in molfile_plugin_t */

PyObject * molfile_plugin_info(PyObject* molcapsule, int plugin_no) {
    molfile_plugin_t *plugin;
    molfile_plugin_t** plugin_list = (molfile_plugin_t**) PyMolfileCapsule_AsVoidPtr(molcapsule);   
    int *plugno = &plugin_no;
    int has_readstructure = 0;
    int has_readbonds = 0;
    int has_readangles = 0;
    int has_writestructure = 0;
    int has_writebonds = 0;
    int has_writeangles = 0;
    int has_readnexttimestep = 0;
    int has_writetimestep = 0;
    int plugin_list_size = sizeof(plugin_list) / sizeof(molfile_plugin_t**);
    if (plugno==NULL || plugin_no<0){
      PyErr_Format(PyExc_IOError, "Error: molfile plugin handle no should be given, be positive value and should not exceed the list length'%d'. You set '%d'", plugin_list_size, plugin_no);
      return 0;
    }
    plugin = plugin_list[plugin_no];
    if(plugin==NULL || !plugin->open_file_read){
      PyErr_Format(PyExc_IOError, "Error: molfile plugin '%d' is not initialized.", plugin_no);
      return 0;
    }
    if (plugin->read_structure) has_readstructure = 1;
    if (plugin->read_bonds) has_readbonds = 1;
    if (plugin->read_angles) has_readangles = 1;
    if (plugin->read_next_timestep) has_readnexttimestep = 1;
    if (plugin->write_structure) has_writestructure = 1;
    if (plugin->write_bonds) has_writebonds = 1;
    if (plugin->write_angles) has_writeangles = 1;
    if (plugin->write_timestep) has_writetimestep = 1;
    PyObject *tuple = PyTuple_New(17);
    PyTuple_SET_ITEM(tuple, 0, PyString_FromString(plugin->filename_extension));
    PyTuple_SET_ITEM(tuple, 1, PyString_FromString(plugin->name));
    PyTuple_SET_ITEM(tuple, 2, PyInt_FromLong((long)has_readstructure));
    PyTuple_SET_ITEM(tuple, 3, PyInt_FromLong((long)has_readbonds));
    PyTuple_SET_ITEM(tuple, 4, PyInt_FromLong((long)has_readangles));
    PyTuple_SET_ITEM(tuple, 5, PyInt_FromLong((long)has_readnexttimestep));
    PyTuple_SET_ITEM(tuple, 6, PyInt_FromLong((long)has_writestructure));
    PyTuple_SET_ITEM(tuple, 7, PyInt_FromLong((long)has_writebonds));
    PyTuple_SET_ITEM(tuple, 8, PyInt_FromLong((long)has_writeangles));
    PyTuple_SET_ITEM(tuple, 9, PyInt_FromLong((long)has_writetimestep));
    PyTuple_SET_ITEM(tuple, 10, PyString_FromString(plugin->prettyname));
    PyTuple_SET_ITEM(tuple, 11, PyString_FromString(plugin->type));
    PyTuple_SET_ITEM(tuple, 12, PyString_FromString(plugin->author));
    PyTuple_SET_ITEM(tuple, 13, PyInt_FromLong((long)plugin->majorv));
    PyTuple_SET_ITEM(tuple, 14, PyInt_FromLong((long)plugin->minorv));
    PyTuple_SET_ITEM(tuple, 15, PyInt_FromLong((long)plugin->abiversion));
    PyTuple_SET_ITEM(tuple, 16, PyInt_FromLong((long)plugin->is_reentrant));
    return tuple;
}

PyObject* read_fill_structure(PyObject* molpack, PyObject* prototype)
{
    //Py_Initialize();
    import_array();
    int options = 0;
    molfile_plugin_t* plugin;
    void* file_handle;
    molfile_atom_t* data;
    int numatoms, status;
    int nd;
    PyObject *ret = NULL;
    // Access plugin_handle values
    MolObject* plugin_handle = (MolObject*) molpack;
    if (plugin_handle->plugin) {
        plugin = (molfile_plugin_t*) PyMolfileCapsule_AsVoidPtr(plugin_handle->plugin);
        //plugin = plugin_handle->plugin;   
    } else {
        PyErr_Format(PyExc_IOError, "molfile plugin is not active.");
	return NULL;
    } 
    if (plugin_handle->file_handle) {
        file_handle = (void*) PyMolfileCapsule_AsVoidPtr(plugin_handle->file_handle);
        //file_handle = plugin_handle->file_handle;
    } else {
        PyErr_Format(PyExc_IOError, "no file handle in molfile plugin handle.");
	return NULL; 
    } 
    if (plugin_handle->natoms) {
        numatoms = plugin_handle->natoms;
    } else { 
        PyErr_Format(PyExc_IOError, "no assigned number of atoms in molfile plugin handle.");
	return NULL;
    } 
    // Allocate memory for array of molfile_atom_t struct
    data = (molfile_atom_t *)calloc(numatoms,sizeof(molfile_atom_t));
    // Get array values in molfile_atom_t
    if (plugin->read_structure) {
        status = plugin->read_structure(file_handle, &options, data);
        // Check if the plugin returns the results
        if (status!=0){
            PyErr_Format(PyExc_IOError, "Error accessing molfile_atom_t in read_structure function of plugin.");
            return NULL;
        }
	if(numatoms>0){
            nd = 1;
            npy_intp dims[1] = { numatoms };
            npy_intp strides[1] = { sizeof(molfile_atom_t) };
            Py_INCREF(prototype);
            ret = PyArray_NewFromDescr(Py_TYPE(prototype), PyArray_DESCR((PyArrayObject*)prototype), 
	       	                       nd, dims,
		                       strides, data, 
			               PyArray_FLAGS((PyArrayObject*)prototype), prototype);
            Py_DECREF(prototype);
            return (PyObject*) ret;
	} else {
            PyErr_Format(PyExc_AttributeError, "plugin read_structure does not have atoms information.");
	    Py_RETURN_NONE;
	}
    } else {
        PyErr_Format(PyExc_AttributeError, "molfile plugin does not have read_structure function.");
	Py_RETURN_NONE;
    }
}

PyObject* read_fill_bonds(PyObject* molpack)
{
    import_array();
    int options = 0;
    molfile_plugin_t* plugin;
    void* file_handle;
    molfile_atom_t* data;
    int numatoms, status;
    int nd;
    PyObject *ret = NULL;
    // Access plugin_handle values
    MolObject* plugin_handle = (MolObject*) molpack;
    if (plugin_handle->plugin) {
        plugin = (molfile_plugin_t*) PyMolfileCapsule_AsVoidPtr(plugin_handle->plugin);
        //plugin = plugin_handle->plugin;   
    } else {
        PyErr_Format(PyExc_IOError, "molfile plugin is not active.");
	return NULL;
    } 
    if (plugin_handle->file_handle) {
        file_handle = (void*) PyMolfileCapsule_AsVoidPtr(plugin_handle->file_handle);
        //file_handle = plugin_handle->file_handle;
    } else {
        PyErr_Format(PyExc_IOError, "no file handle in molfile plugin handle.");
	return NULL; 
    } 
    numatoms = plugin_handle->natoms;
    if (plugin->read_bonds) {
        int nbonds, *from, *to, *bondtype, nbondtypes;
        float *bondorder;
        char **bondtypename;
        if ((status = plugin->read_bonds(file_handle, &nbonds, &from, &to, 
				         &bondorder, &bondtype, &nbondtypes, &bondtypename))) {
            PyErr_Format(PyExc_IOError, "Error accessing read_bonds function of plugin.");
            return NULL;
        }
        PyArrayInterface *inter = NULL;
        inter = (PyArrayInterface*)malloc(sizeof(PyArrayInterface));
        if (inter==NULL)
            return PyErr_NoMemory();
        inter->flags = NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE;
	ret = PyDict_New();
        nd = 1;
        npy_intp istrides[1] = { NPY_SIZEOF_INT };
        npy_intp fstrides[1] = { NPY_SIZEOF_FLOAT };
        npy_intp cstrides[1] = { sizeof(NPY_STRING) };
	if (nbonds>0) {
            PyObject *from_arr = NULL;
            PyObject *to_arr = NULL;
            npy_intp dims[1] = { nbonds };
            from_arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_INT), 
                                            nd, dims,
                                            istrides, from, 
                                            inter->flags, NULL);
	    PyDict_SetItemString(ret, "from", from_arr);
            to_arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_INT), 
                                          nd, dims,
                                          istrides, to, 
                                          inter->flags, NULL);
	    PyDict_SetItemString(ret, "to", to_arr);
	    if (bondorder!=NULL) {
                PyObject *bondorder_arr = NULL;
                bondorder_arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_FLOAT), 
                                                     nd, dims,
                                                     fstrides, bondorder, 
			                             inter->flags, NULL);
	        PyDict_SetItemString(ret, "bondorder", bondorder_arr);
	    }
	    if (bondtype!=NULL && nbondtypes>0) {
                PyObject *bondtype_arr = NULL;
                bondtype_arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_INT), 
                                                    nd, dims,
                                                    istrides, bondtype, 
			                            inter->flags, NULL);
	        PyDict_SetItemString(ret, "bondtype", bondtype_arr);
	    }
	    if (bondtypename!=NULL && nbondtypes>0) {
                PyObject *bondtypename_arr = NULL;
                npy_intp cdims[1] = { nbondtypes };
                bondtypename_arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_STRING), 
                                                        nd, cdims,
                                                        cstrides, bondtypename, 
	                                                inter->flags, NULL);
	        PyDict_SetItemString(ret, "bondtypename", bondtypename_arr);
	    }
            return (PyObject*) ret;
	} else {
            Py_RETURN_NONE;
	}
    } else {
        PyErr_Format(PyExc_AttributeError, "molfile plugin does not have read_bonds function.");
        Py_RETURN_NONE;
    }
}

PyObject* read_fill_angles(PyObject* molpack)
{
    import_array();
    int options = 0;
    molfile_plugin_t* plugin;
    void* file_handle;
    molfile_atom_t* data;
    int numatoms, status;
    int nd;
    int nodata = 0;
    PyObject *ret = NULL;
    // Access plugin_handle values
    MolObject* plugin_handle = (MolObject*) molpack;
    if (plugin_handle->plugin) {
        plugin = (molfile_plugin_t*) PyMolfileCapsule_AsVoidPtr(plugin_handle->plugin);
        //plugin = plugin_handle->plugin;   
    } else {
        PyErr_Format(PyExc_IOError, "molfile plugin is not active.");
	return NULL;
    } 
    if (plugin_handle->file_handle) {
        file_handle = (void*) PyMolfileCapsule_AsVoidPtr(plugin_handle->file_handle);
        //file_handle = plugin_handle->file_handle;
    } else {
        PyErr_Format(PyExc_IOError, "no file handle in molfile plugin handle.");
	return NULL; 
    } 
    numatoms = plugin_handle->natoms;
    // Check if there is read_angles support in this plugin
    if (plugin->read_angles) {
	// Angles
        int numangles;
        int *angles = NULL;
	int *angletypes = NULL;
        int numangletypes;
	char **angletypenames = NULL; 
	// Dihedrals
	int numdihedrals; 
	int *dihedrals = NULL;
	int *dihedraltypes = NULL;
	int numdihedraltypes;
        char **dihedraltypenames = NULL; 
	// Impropers
	int numimpropers;
        int *impropers = NULL;
        int *impropertypes = NULL;
	int numimpropertypes;
	char **impropertypenames = NULL;
	// Cterms
        int numcterms, ctermcols, ctermrows;
	int *cterms = NULL; 
	// Initilize zeros to number of angles, dihedrals, so on ...
	numangles = 0;
	numangletypes = 0;
	numdihedrals = 0;
	numdihedraltypes = 0;
	numimpropers = 0;
	numimpropertypes = 0;
	numcterms = 0;
	// Calling read_angles to gather the information
        if ((status = plugin->read_angles(file_handle, &numangles, &angles, &angletypes,
                                          &numangletypes, &angletypenames, &numdihedrals,
                                          &dihedrals, &dihedraltypes, &numdihedraltypes,
                                          &dihedraltypenames, &numimpropers, &impropers,        
                                          &impropertypes, &numimpropertypes, &impropertypenames,
                                          &numcterms, &cterms, &ctermcols, &ctermrows))) {
            PyErr_Format(PyExc_IOError, "Error accessing read_angles function of plugin.");
            return NULL;
        }
        PyArrayInterface *inter = NULL;
        inter = (PyArrayInterface*)malloc(sizeof(PyArrayInterface));
        if (inter==NULL)
            return PyErr_NoMemory();
        inter->flags = NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE;
	ret = PyDict_New();
        nd = 1;
        npy_intp istrides[1] = { NPY_SIZEOF_INT };
        npy_intp sstrides[1] = { sizeof(NPY_STRING) };
	if (numangles>0 && angles!=NULL) {
	    nodata = 1;
            PyObject *angles_arr = NULL;
            npy_intp adims[1] = { numangles };
            angles_arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_INT), 
                                              nd, adims,
                                              istrides, angles, 
                                              inter->flags, NULL);
	    PyDict_SetItemString(ret, "angles", angles_arr);
	    if (angletypes!=NULL) {
                PyObject *angletypes_arr = NULL;
                angletypes_arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_INT), 
                                                      nd, adims,
                                                      istrides, angletypes, 
                                                      inter->flags, NULL);
	        PyDict_SetItemString(ret, "angletypes", angletypes_arr);
	    }
	}
	if (numangletypes>0 && angletypenames!=NULL) {
	    nodata = 1;
            PyObject *angletypenames_arr = NULL;
            npy_intp atdims[1] = { numangletypes };
            angletypenames_arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_STRING), 
                                                      nd, atdims,
                                                      sstrides, angletypenames, 
			                              inter->flags, NULL);
	    PyDict_SetItemString(ret, "angletypenames", angletypenames_arr);
	}
	if (numdihedrals>0 && dihedrals!=NULL) {
	    nodata = 1;
            PyObject *dihedrals_arr = NULL;
            npy_intp ddims[1] = { numdihedrals };
            dihedrals_arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_INT), 
                                                 nd, ddims,
                                                 istrides, dihedrals, 
			                         inter->flags, NULL);
	    PyDict_SetItemString(ret, "dihedrals", dihedrals_arr);
	    if (dihedraltypes!=NULL) {
                PyObject *dihedraltypes_arr = NULL;
                dihedraltypes_arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_INT), 
                                                    nd, ddims,
                                                    istrides, dihedraltypes, 
                                                    inter->flags, NULL);
	        PyDict_SetItemString(ret, "dihedraltypes", dihedraltypes_arr);
	    }
	}
	if (numdihedraltypes>0 && dihedraltypenames!=NULL) {
            PyObject *dihedraltypenames_arr = NULL;
            npy_intp dtdims[1] = { numdihedraltypes };
            dihedraltypenames_arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_STRING), 
                                                         nd, dtdims,
                                                         sstrides, dihedraltypenames, 
	                                                 inter->flags, NULL);
	    PyDict_SetItemString(ret, "dihedraltypenames", dihedraltypenames_arr);
	}
	if (numimpropers>0 && impropers!=NULL) {
	    nodata = 1;
            PyObject *impropers_arr = NULL;
            npy_intp idims[1] = { numimpropers };
            impropers_arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_INT), 
                                                 nd, idims,
                                                 istrides, impropers, 
			                         inter->flags, NULL);
	    PyDict_SetItemString(ret, "impropers", impropers_arr);
	    if (impropertypes!=NULL) {
                PyObject *impropertypes_arr = NULL;
                impropertypes_arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_INT), 
                                                         nd, idims,
                                                         istrides, impropertypes, 
                                                         inter->flags, NULL);
	        PyDict_SetItemString(ret, "impropertypes", impropertypes_arr);
	    }
	}
	if (numimpropertypes>0 && impropertypenames!=NULL) {
            PyObject *impropertypenames_arr = NULL;
            npy_intp itdims[1] = { numimpropertypes };
            impropertypenames_arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_STRING), 
                                                         nd, itdims,
                                                         sstrides, impropertypenames, 
	                                                 inter->flags, NULL);
	    PyDict_SetItemString(ret, "impropertypenames", impropertypenames_arr);
	}
	if (numcterms>0 && cterms!=NULL) {
	    nodata = 1;
	    int ctermnd;
            npy_intp *ctermdims;
	    npy_intp *ctermstrides;
            PyObject *cterms_arr = NULL;
	    if (ctermrows>0 || ctermcols>0) {
		ctermnd = 2;
                ctermdims = (npy_intp*)calloc(ctermnd,sizeof(int));
                ctermstrides = (npy_intp*)calloc(ctermnd,sizeof(int));
	        ctermdims[0] = ctermrows;
	        ctermdims[1] = ctermcols;
	        ctermstrides[0] = NPY_SIZEOF_INT;
	        ctermstrides[1] = ctermcols*NPY_SIZEOF_INT;
	    } else {
		ctermnd = 1;
                ctermdims = (npy_intp*)calloc(ctermnd,sizeof(int));
                ctermstrides = (npy_intp*)calloc(ctermnd,sizeof(int));
	        ctermdims[0] = 8*numcterms;
	        ctermstrides[0] = NPY_SIZEOF_INT;
	    }
            cterms_arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_INT), 
                                              ctermnd, ctermdims,
                                              ctermstrides, cterms, 
			                      inter->flags, NULL);
	    PyDict_SetItemString(ret, "cterms", cterms_arr);
	}
	if (nodata>0) {
            return (PyObject*) ret;
	} else {
            Py_RETURN_NONE;
	}
    } else {
        PyErr_Format(PyExc_AttributeError, "molfile plugin does not have read_angles function.");
        Py_RETURN_NONE;
    }
}


PyObject* read_fill_next_timestep(PyObject* molpack)
{
    import_array();
    molfile_plugin_t* plugin;
    void* file_handle;
    int numatoms, status;
    int nd;
    int i, d;
    PyObject *ret = NULL;
    // Access plugin_handle values
    MolObject* plugin_handle = (MolObject*) molpack;
    if (plugin_handle->plugin) {
        plugin = (molfile_plugin_t*) PyMolfileCapsule_AsVoidPtr(plugin_handle->plugin);
        //plugin = plugin_handle->plugin;   
    } else {
        PyErr_Format(PyExc_IOError, "molfile plugin is not active.");
	return NULL;
    } 
    if (plugin_handle->file_handle) {
        file_handle = (void*) PyMolfileCapsule_AsVoidPtr(plugin_handle->file_handle);
        //file_handle = plugin_handle->file_handle;
    } else {
        PyErr_Format(PyExc_IOError, "no file handle in molfile plugin handle.");
	return NULL; 
    } 
    if (plugin_handle->natoms) {
        numatoms = plugin_handle->natoms;
    } else { 
        PyErr_Format(PyExc_IOError, "no assigned number of atoms in molfile plugin handle.");
	return NULL;
    }
    if (plugin->read_next_timestep) {
        PyArrayInterface *inter = NULL;
        inter = (PyArrayInterface*)malloc(sizeof(PyArrayInterface));
        if (inter==NULL)
            return PyErr_NoMemory();
        inter->flags = NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE;
	ret = PyDict_New();
        molfile_timestep_t timestep;
	// Check if the velocities will be supplied from this plugin
	int has_velocities = -1;
	unsigned int total_steps = 0;
	unsigned int bytes_per_step = 0;
	molfile_timestep_metadata_t timestep_metadata;
        if (plugin->read_timestep_metadata) {
	    plugin->read_timestep_metadata(file_handle, &timestep_metadata);
            total_steps = timestep_metadata.count; 
	    has_velocities = timestep_metadata.has_velocities;
	    bytes_per_step = timestep_metadata.avg_bytes_per_timestep;
	} else {
	    total_steps = 0;
	    has_velocities = -2;
	}
	timestep.A=-1;
	timestep.B=-1;
	timestep.C=-1;
	timestep.alpha=-1;
	timestep.beta=-1;
	timestep.gamma=-1;
	timestep.physical_time=-1;
        timestep.coords = (float *)malloc(3*numatoms*sizeof(float));
	if(has_velocities == -2 || has_velocities == 1) { 
            timestep.velocities = (float *)malloc(3*numatoms*sizeof(float));
	    for(i=0;i<numatoms;i++)
	        for(d=0;d<3;d++)
	            timestep.velocities[i*3+d] = -1111*(d+1);
	}
        status = plugin->read_next_timestep(file_handle, numatoms, &timestep);
        if (status == MOLFILE_EOF) {
	    Py_RETURN_NONE;
	}
	else if (status != MOLFILE_SUCCESS) {
            PyErr_Format(PyExc_AttributeError, "Failed in calling read_next_timestep function of plugin.");
	    Py_RETURN_NONE;
        } 
	else {
            nd = 2;
            PyObject *coords_arr = NULL;
            npy_intp dims[2] = { numatoms, 3 };
            npy_intp strides[2] = { 3*sizeof(float), sizeof(float) };
            coords_arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_FLOAT), 
                                              nd, dims,
                                              strides, timestep.coords, 
	                                      inter->flags, NULL);
	    PyDict_SetItemString(ret, "coords", coords_arr);
	    if (timestep.velocities!=NULL && (has_velocities == -2 || has_velocities == 1)) {
		if (-1111 != (int)timestep.velocities[0] && 
		    -2222 != (int)timestep.velocities[1] && 
		    -3333 != (int)timestep.velocities[2]) {
                    PyObject *velocities_arr = NULL;
                    velocities_arr = PyArray_NewFromDescr(&PyArray_Type, PyArray_DescrFromType(NPY_FLOAT), 
                                                          nd, dims,
                                                          strides, timestep.velocities, 
	                                                  inter->flags, NULL);
	            PyDict_SetItemString(ret, "velocities", velocities_arr);
		}
	    } else {
		if(has_velocities == -2 || has_velocities == 1)
		    free(timestep.velocities);
	    }
            if(timestep.A > -1)
	        PyDict_SetItemString(ret, "A", PyFloat_FromDouble((double)timestep.A));
            if(timestep.B > -1)
	        PyDict_SetItemString(ret, "B", PyFloat_FromDouble((double)timestep.B));
            if(timestep.C > -1)
	        PyDict_SetItemString(ret, "C", PyFloat_FromDouble((double)timestep.C));
            if(timestep.alpha > -1)
	        PyDict_SetItemString(ret, "alpha", PyFloat_FromDouble((double)timestep.alpha));
	    if(timestep.beta > -1)
	        PyDict_SetItemString(ret, "beta", PyFloat_FromDouble((double)timestep.beta));
	    if(timestep.gamma > -1)
	        PyDict_SetItemString(ret, "gamma", PyFloat_FromDouble((double)timestep.gamma));
	    if(timestep.physical_time > -1)
	        PyDict_SetItemString(ret, "physical_time", PyFloat_FromDouble(timestep.physical_time));
	    //if(has_velocities > -1)
	    //    PyDict_SetItemString(ret, "has_velocities", PyLong_FromLong((long)has_velocities));
	    if(total_steps > -1)
	        PyDict_SetItemString(ret, "total_steps", PyLong_FromLong((long)total_steps));
	    PyDict_SetItemString(ret, "has_velocities", PyLong_FromLong((long)has_velocities));
            return (PyObject*) ret;
	}
    } else {
        PyErr_Format(PyExc_AttributeError, "molfile plugin does not have read_next_timestep function.");
	Py_RETURN_NONE;
    }
}

PyObject* are_plugins_same(PyObject* molpack_a, PyObject* molpack_b)
{
    molfile_plugin_t* plugin_a;
    molfile_plugin_t* plugin_b;
    PyObject *ret = NULL;
    MolObject* plugin_handle_a = (MolObject*) molpack_a;
    MolObject* plugin_handle_b = (MolObject*) molpack_b;
    if (plugin_handle_a->plugin) {
        plugin_a = (molfile_plugin_t*) PyMolfileCapsule_AsVoidPtr(plugin_handle_a->plugin);
        //plugin_a = plugin_handle_a->plugin;   
    } else {
        PyErr_Format(PyExc_IOError, "Arg 1 of the molfile plugin is not active.");
	return NULL;
    } 
    if (plugin_handle_b->plugin) {
        plugin_b = (molfile_plugin_t*) PyMolfileCapsule_AsVoidPtr(plugin_handle_b->plugin);
        //plugin_b = plugin_handle_b->plugin;   
    } else {
        PyErr_Format(PyExc_IOError, "Arg 2 of the molfile plugin is not active.");
	return NULL; 
    } 
    if(plugin_a == plugin_b){
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
}

PyObject* are_filehandles_same(PyObject* molpack_a, PyObject* molpack_b)
{
    MolObject* plugin_handle_a = (MolObject*) molpack_a;
    MolObject* plugin_handle_b = (MolObject*) molpack_b;
    void* file_handle_a; 
    void* file_handle_b; 
    if (plugin_handle_a->file_handle) {
        file_handle_a = (void*) PyMolfileCapsule_AsVoidPtr(plugin_handle_a->file_handle);
        //file_handle_a = plugin_handle_a->file_handle;   
    } else {
        PyErr_Format(PyExc_IOError, "no file handle in arg 1 of molfile plugin.");
	return NULL;
    } 
    if (plugin_handle_b->file_handle) {
        file_handle_b = (void*) PyMolfileCapsule_AsVoidPtr(plugin_handle_b->file_handle);
        //file_handle_b = plugin_handle_b->file_handle;   
    } else {
        PyErr_Format(PyExc_IOError, "no file handle in arg 2 of molfile plugin.");
	return NULL;
    } 
    if(file_handle_a == file_handle_b){
      Py_RETURN_TRUE;
    } else {
      Py_RETURN_FALSE;
    }
}

