# pymolfile
#### Version: 0.0.6
* TNG library links are fixed in CMake compile.

#### Version: 0.0.5
* Fixed bugs in MolfileOpen class.

#### Version: 0.0.4
* Secure the file opens/closes on reader by OpenMolTraj class using 'with'. This provides a stable recall of PyMolfile in series of files.
* Silence all core plugins in molfile_plugins folder at CMake level. This is the default setting now.
* Bug fixed on recursive initialiation of OpenMolFile on trajectory reads. It is safe now to have concurrent multiple calls from MDDataAccess. 

#### Version: 0.0.3
What is new?
* Added script to change printf/fprintf to stdout and stderr at C level on molfile plugins to make the package compatiable with NOMAD parsers.
* Added "nowarnings" option to OpenMolfile to prevent user warnings.

Anything in mind for additional improvements? 
* C level silencing modifications are mandatory. This might be optional with a setup.py parameter.
* pymolfile only supports read functions at plugins. Write functions might also be added to convert data to other formats.

Needs precaution when using?
* Recursive initialization of OpenMolfile leads to breaks at trajectory class when an outer calling function fails (Ex: in MDDataAccess package) since trajectory file can still be open and can not be closed properly without a delete call.

#### Version: 0.0.2

Python interface for molfile plugins. 

### Molfile Plugins:

For more information about VMD molfile plugins with UIUC Open Source License   
please see <http://www.ks.uiuc.edu/Research/vmd/plugins/molfile/>

### Dependencies:

* numpy 
* cmake >3.1
* swig >3.0
* tng_io (installs with pymolfile setup)
* NetCDF (Optional but recomended for full experience)
* Expat
* Babel (Optional but recomended for full experience)
* Tcl/Tk >8.5 (Optional)

### Download:

```
git clone git@gitlab.mpcdf.mpg.de:nomad-lab/pymolfile.git
```

### Installation:

```
cd pymolfile
python3 setup.py install
```

### Example usage:

```python
import pymolfile

moltopo = pymolfile.OpenMolfile('test/DPDP.pdb')

print(moltopo.topology.structure)
print(moltopo.topology.bonds)
print(moltopo.topology.angles)

moltraj = pymolfile.OpenMolfile('test/DPDP.nc', topology=moltopo)

    if moltraj.trajectory is not None:
        step=0
        while True:
            positions = moltraj.trajectory.iread()
            if positions is not None:
                print("STEP:",step)
                print(positions)
                step += 1
            else:
                break

```

