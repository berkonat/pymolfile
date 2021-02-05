import numpy
import pymolfile as pym

cfname=None
path="./test/"
#sfname="./test/DPDP.pdb"
#cfname="./test/DPDP.nc"

#sfname="./test/ala3.pdb"
#sfname="./test/ala3.psf"
#cfname="./test/ala3.dcd"

#sfname="./test/md.gro"
#cfname="./test/md.trr"
#cfname="./test/md.xtc"

#sfname="./test/md_1u19.gro"
#cfname="./test/md_1u19.xtc"

#sfname = path + "1blu.pdb"
#sfname = path + "1blu.mmtf"
#sfname = path + "1CRN.cif"
#sfname = path + "1lee.ccp4"
#sfname = path + "betaGal.mrc"
#sfname = path + "adrenalin.mol2"
#sfname = path + "adrenalin.sdf"
#sfname = path + "esp.dx"
#sfname = path + "md_ascii_trj.gro"
sfname = path + "md_ascii_trj.pdb"
#sfname = path + "3pqr.pqr"
#cfname = path + "3pqr_validation.xml"
#sfname = path + "3pqr-pot.dxbin"
#sfname = path + "3pqr.cns"
#sfname = path + "1cnr.ply"

print("Reading file...")
moltopo = pym.OpenMolfile(sfname, silent=True)
#moltopo = pym.OpenMolfile(cfname, topology=sfname, silent=True)
#moltopo = pym.OpenMolfile(sfname)
if False:
#if True:
    for plugin in pym.list_plugins():
        print([plugin[0],plugin[1],plugin[10]])

if moltopo.kwords["file_format"] is not None:
    print(moltopo.kwords["file_format"])
if moltopo.kwords["file_plugin"] is not None:
    print(moltopo.kwords["file_plugin"])
if moltopo.kwords["topology_format"] is not None:
    print(moltopo.kwords["topology_format"])
if moltopo.kwords["topology_plugin"] is not None:
    print(moltopo.kwords["topology_plugin"])
if moltopo.fplugin is not None:
    print(moltopo.fplugin)
if moltopo.tplugin is not None:
    print(moltopo.tplugin)
if moltopo.topology is not None:
    if moltopo.topology.structure is not None:
        print(moltopo.topology.structure)
    if moltopo.topology.bonds is not None:
        print(moltopo.topology.bonds)
    if moltopo.topology.angles is not None:
        print(moltopo.topology.angles)
if moltopo.trajectory is not None:
    step=0
    while True:
        positions = moltopo.trajectory.iread()
        if positions is not None:
            print("STEP:",step)
            print(positions)
            step += 1
        else:
            break

if(moltopo.topology is None and 
   moltopo.trajectory is None):
    print("Can not read the file!")

if cfname:
    moltraj = pym.OpenMolfile(cfname, topology=moltopo, silent=True)
    print(moltraj)
    print(moltraj.trajectory)

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
#    del moltraj
#del moltopo


