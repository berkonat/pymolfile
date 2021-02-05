# pymolfile
#### Version: 0.0.2

Python interface for molfile plugins. 

### Molfile Plugins:

For more information about VMD molfile plugins with UIUC Open Source License   
please see <http://www.ks.uiuc.edu/Research/vmd/plugins/molfile/>

### Dependencies:

* numpy 
* cmake >2.8.12
* swig >3.0
* tng_io (installs with pymolfile setup)
* NetCDF 
* Expat
* Babel (Optional)
* Tcl/Tk >8.5 (Optional)

### Download:

```
git clone git@gitlab.mpcdf.mpg.de:berko/pymolfile.git
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

