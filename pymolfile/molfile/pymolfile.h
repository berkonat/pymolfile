/* Hey emacs this is -*- C -*- and this is my editor vim.
 * 
 * molfile.c : C and Fortran interfaces for molfile_plugins
 * Copyright (c) Berk Onat <b.onat@warwick.ac.uk> 2017
 *
 * This program is under UIUC LICENSE
 */

/*
 * The code is written following the plugin test 
 * context of f77_molfile.c by Axel Kohlmeyer and 
 * in molfile_plugin/src/f77 and catdcd.c by 
 * Justin Gullingsrud of VMD plugins.
 */

#ifndef _MOLFILE_H_
#define _MOLFILE_H_

#ifdef __cplusplus
extern "C"
{

#endif

#if PY_VERSION_HEX >= 0x03000000
#define NUMPY_IMPORT_ARRAY_RETVAL NULL
#else
#define NUMPY_IMPORT_ARRAY_RETVAL
#endif

#include "molfile_plugin.h"
#include "libmolfile_plugin.h"
#include "vmdplugin.h"
#include "Python.h"
#include "structmember.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>


#ifndef MAXPLUGINS
#define MAXPLUGINS 200
#endif

struct MolObject {
    PyObject_HEAD
    PyObject * plugin;
    PyObject * file_handle;
    int natoms;
    MolObject(void) {}
};
/*
struct MolObject {
    PyObject_HEAD
    molfile_plugin_t* plugin;
    void* file_handle;
    int natoms;
    MolObject(void) {}
};
*/


#if PY_VERSION_HEX >= 0x03000000
#define PyInt_AsSsize_t PyLong_AsSsize_t
//#define PyInt_AsLong PyLong_AsLong
#define PyArray_Check(op) PyObject_TypeCheck(op, &PyArray_Type)
//#define PyString_FromString PyBytes_FromString
#define PyUString_Check PyUnicode_Check
#define PyUString_GET_SIZE PyUnicode_GET_SIZE
#define PyUString_FromFormat PyUnicode_FromFormat
//#define PyInt_FromLong PyLong_FromLong
#define PyString_Type PyBytes_Type
#define PyInt_Type PyLong_Type

void del_molfile_plugin_list(PyObject* molcapsule);
void del_molfile_file_handle(PyObject* molcapsule);

static void * PyMolfileCapsule_AsVoidPtr(PyObject *obj);

static void * PyMolfileCapsule_AsVoidPtr(PyObject *obj)
{
    void *ret = PyCapsule_GetPointer(obj, "plugin_handle");
    if (ret == NULL) {
        PyErr_Clear();
    }   
    return ret;
}

static PyObject * PyMolfileCapsule_FromVoidPtr(void *ptr, void (*destr)(PyObject *));
static PyObject * PyMolfileCapsule_FromVoidPtr(void *ptr, void (*destr)(PyObject *))
{
    PyObject *ret = PyCapsule_New(ptr, "plugin_handle", destr);
    if (ret == NULL) {
        PyErr_Clear();
    }
    return ret;
}

#else
#define PyBytes_FromString PyString_FromString

void del_molfile_plugin_list(void* molcapsule);
void del_molfile_file_handle(void* molcapsule);

static void * PyMolfileCapsule_AsVoidPtr(PyObject *obj);

static void * PyMolfileCapsule_AsVoidPtr(PyObject *obj)
{
    return PyCObject_AsVoidPtr(obj);
}

static PyObject * PyMolfileCapsule_FromVoidPtr(void *ptr, void (*destr)(void *));

static PyObject * PyMolfileCapsule_FromVoidPtr(void *ptr, void (*destr)(void *))
{
    return PyCObject_FromVoidPtr(ptr, destr);
}

#endif

static void MolObject_dealloc(MolObject* self)
{
    Py_XDECREF(self->plugin);
    Py_XDECREF(self->file_handle);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject * MolObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    MolObject *self;

    self = (MolObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->plugin = NULL;
        self->file_handle = NULL;
        self->natoms = 0;
    }

    return (PyObject *)self;
}

static int MolObject_init(MolObject *self, PyObject *args, PyObject *kwds)
{
    //molfile_plugin_t *plugin = NULL;
    //void *file_handle = NULL;
    //molfile_plugin_t *tmp1 = NULL;
    //void *tmp2 = NULL;
    PyObject *plugin = NULL;
    PyObject *file_handle = NULL;
    PyObject *tmp1 = NULL;
    PyObject *tmp2 = NULL;

    static char *kwlist[] = {"plugin", "file_handle", "natoms", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|OOi", kwlist,
                                      &plugin, &file_handle,
                                      &self->natoms))
        return -1;

    if (plugin) {
        tmp1 = self->plugin;
        Py_INCREF(plugin);
        self->plugin = plugin;
        Py_XDECREF(tmp1);
    }

    if (file_handle) {
        tmp2 = self->file_handle;
        Py_INCREF(file_handle);
        self->file_handle = file_handle;
        Py_XDECREF(tmp2);
    }

    return 0;
}

//static molfile_plugin_t* MolObject_plugin(MolObject* self)
static PyObject* MolObject_plugin(MolObject* self)
{
        return self->plugin;
}

//static void* MolObject_file_handle(MolObject* self)
static PyObject* MolObject_file_handle(MolObject* self)
{
        return self->file_handle;
}

static PyObject* MolObject_natoms(MolObject* self)
{
        return PyLong_FromLong((long)self->natoms);
}

static PyMemberDef MolObject_members[] = {
    {"plugin", T_OBJECT_EX, offsetof(MolObject, plugin), 0,
     "molfile_plugin_t type plugin"},
    {"file_handle", T_OBJECT_EX, offsetof(MolObject, file_handle), 0,
     "file handle for plugin"},
    {"natoms", T_INT, offsetof(MolObject, natoms), 0,
     "number of atoms"},
    {NULL, 0, 0, 0, NULL}  /* Sentinel */
};

static PyMethodDef MolObject_methods[] = {
    {"get_plugin", (PyCFunction)MolObject_plugin, METH_NOARGS,
     "Return the plugin"
    },
    {"get_file_handle", (PyCFunction)MolObject_file_handle, METH_NOARGS,
     "Return the plugin"
    },
    {"get_natoms", (PyCFunction)MolObject_natoms, METH_NOARGS,
     "Return the number of atoms"
    },
    {NULL, 0, 0, NULL}  /* Sentinel */
};


#ifndef PyVarObject_HEAD_INIT
    #define PyVarObject_HEAD_INIT(type, size) \
        PyObject_HEAD_INIT(type) size,
#endif

static PyTypeObject MolObjectType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "molobject",                 /*tp_name*/
    sizeof(MolObject),          /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MolObject_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_reserved*/
    0,                         /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
    Py_TPFLAGS_DEFAULT | 
	    Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "molobject objects",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    MolObject_methods,         /* tp_methods */
    MolObject_members,         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)MolObject_init,      /* tp_init */
    0,                         /* tp_alloc */	
    MolObject_new,                 /* tp_new */
};

PyObject * molfile_plugin_info(PyObject* molcapsule, int plugin_no);
PyObject * read_fill_structure(PyObject* molpack, PyObject* prototype);
PyObject * write_fill_structure(PyObject* molpack, PyObject* molarray);
PyObject * read_fill_bonds(PyObject* molpack);
PyObject * write_fill_bonds(PyObject* molpack, PyObject* moldict);
PyObject * read_fill_angles(PyObject* molpack);
PyObject * write_fill_angles(PyObject* molpack, PyObject* moldict);
PyObject * read_fill_next_timestep(PyObject* molpack);
PyObject * write_fill_timestep(PyObject* molpack, PyObject* moldict);
PyObject * are_plugins_same(PyObject* molpack_a, PyObject* molpack_b);
PyObject * are_filehandles_same(PyObject* molpack_a, PyObject* molpack_b);
PyObject * get_plugin(PyObject* molcapsule, int plug_no);
PyObject * molfile_plugin_list(int maxsize);

int molfile_init(void);

int molfile_finish(void);

#ifdef __cplusplus
}
#endif

#endif /* _MOLFILE_H_ */

