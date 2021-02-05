/* -*- C -*-  (not really, but good for syntax highlighting) */
/* SWIG interface for libpymolfile of VMD molfile_plugins
   Copyright (c) 2017 Berk Onat <b.onat@warwick.ac.uk>
   Published under UIUC LICENSE

   swig -c++ -python -outdir . molfile/libpymolfile.i
*/
%define DOCSTRING
"
:Author:  Berk Onat <b.onat@warwick.ac.uk>
:Year:    2017
:Licence: UIUC LICENSE


"
%enddef

%module(docstring=DOCSTRING) libpymolfile


%{
/* Python SWIG interface to libpymolfile
   Copyright (c) 2017 Berk Onat <b.onat@warwick.ac.uk>
   Published with UIUC LICENSE
 */
#define SWIG_FILE_WITH_INIT
#define __STDC_FORMAT_MACROS
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <inttypes.h>
#include "pymolfile.h"
%}

%include "numpy.i"

%init %{
Py_Initialize();
import_array();
%}


/* 
  Wrapping only high-level plugin functions to register VMD 
  plugins and to retrive the data through molfile_plugin interface.

  Only modifing call signatures. This will help one to access functions 
  without dealing with pointers from python.
*/


/* pymolfile.c 
   initialize and finalize molfile plugins
*/
%feature("autodoc", "0") molfile_plugin_list;
extern PyObject* molfile_plugin_list(int maxsize);

%feature("autodoc", "0") molfile_init;
extern int molfile_init(void);

%feature("autodoc", "0") molfile_finish;
extern int molfile_finish(void);

%feature("autodoc", "0") get_plugin;
extern PyObject* get_plugin(PyObject* molcapsule, int plug_no);


%feature("autodoc", "0") molfile_plugin_info;
extern PyObject * molfile_plugin_info(PyObject* molcapsule, int plugin_no);


%feature("autodoc", "0") my_open_file_read;
%rename (open_file_read) my_open_file_read;
%exception my_open_file_read {
  $action
  if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
PyObject * my_open_file_read(PyObject* molcapsule, char* fname, char* ftype, int natoms) {
    if (PyType_Ready(&MolObjectType) < 0)
        return NULL;
    PyTypeObject *type = &MolObjectType;
    MolObject *plugin_c;
    molfile_plugin_t* plugin = (molfile_plugin_t*) PyMolfileCapsule_AsVoidPtr(molcapsule);
    plugin_c = (MolObject *)type->tp_alloc(type, 0);
    /*plugin_c->plugin = plugin;*/
    plugin_c->plugin = molcapsule;
    void *file_handle = plugin->open_file_read(fname, ftype, &natoms);
    plugin_c->file_handle = PyMolfileCapsule_FromVoidPtr(file_handle, del_molfile_file_handle);
    if (!plugin_c->file_handle) {
        Py_RETURN_NONE;
    } else {
        plugin_c->natoms = natoms;
        return (PyObject *)plugin_c;
    }
  }
%}

%feature("autodoc", "0") my_close_file_read;
%rename (close_file_read) my_close_file_read;
%exception my_close_file_read {
  $action
  if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
PyObject * my_close_file_read(PyObject* molpack) {
    MolObject* plugin_handle = (MolObject*) molpack;
    PyObject* plugincapsule = plugin_handle->plugin;   
    PyObject* filecapsule = plugin_handle->file_handle;
    molfile_plugin_t* plugin = (molfile_plugin_t*) PyMolfileCapsule_AsVoidPtr(plugincapsule);
    void *file_handle = (void*) PyMolfileCapsule_AsVoidPtr(filecapsule);
    plugin->close_file_read(file_handle);
    Py_XDECREF(filecapsule);
    Py_XDECREF(plugin_handle->file_handle);
    return (PyObject *)plugin_handle;
  }
%}

%feature("autodoc", "0") read_fill_structure;
extern PyObject * read_fill_structure(PyObject* molpack, PyObject* prototype);

%feature("autodoc", "0") read_fill_bonds;
extern PyObject * read_fill_bonds(PyObject* molpack);

%feature("autodoc", "0") read_fill_angles;
extern PyObject * read_fill_angles(PyObject* molpack);

%feature("autodoc", "0") read_fill_next_timestep;
extern PyObject * read_fill_next_timestep(PyObject* molpack);

%feature("autodoc", "0") are_plugins_same;
PyObject* are_plugins_same(PyObject* molpack_a, PyObject* molpack_b);

%feature("autodoc", "0") are_filehandles_same;
PyObject* are_filehandles_same(PyObject* molpack_a, PyObject* molpack_b);

