#!/usr/bin/env python

# ORIGINAL WORK
# Cite as: Ashivni Shekhawat, Robert O. Ritchie, "Toughenss and Strength of Nanocrystalline Graphene",
# Nature Communications

# GRAPHITE STRUCTURE FROM THIS SCRIPT
# Cite as  Kenny Jolley, Ben Maerz et al, Carbon 2017  TO BE PUBLISHED


import grainBdr as gb
import polyCrystal as pc
from ase import Atoms
from ase import io
import numpy
import math

# Function generates a set of random rotation matricies.
def gen_random_rot_matrix(N):
    # Returns N random rotation matricies.
    rot_mats = []
    for i in range(N):
        theta = 2 * numpy.pi * numpy.random.rand()
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        v1 = numpy.array([costheta, -sintheta, 0])
        v2 = numpy.array([sintheta,  costheta, 0])
        v3 = numpy.array([0       ,  0       , 1])
        V = numpy.zeros((3, 3))
        V[:, 0], V[:, 1], V[:, 2] = v1, v2, v3
        rot_mats.append(V)
    
    return rot_mats


def writeLBOMDData(cr,fName):
    f = open(fName,'w+')
        
    f.write('%d\n'%(len(cr)))
    f.write('%.9g  %.9g  %.9g \n'%(cr.cell[0,0],cr.cell[1,1],cr.cell[2,2]))
        
    for i in range(len(cr)):
        f.write('C_  %.9g %.9g %.9g  0.0\n'%(cr.positions[i,0],cr.positions[i,1],cr.positions[i,2]))
        
    f.close()


# function generates a graphite poly crystalline structure.
def GenGraphitePolyCrystal(**kwargs):
    # ---------------------------------------------------------------------
    # -  Get Customisable parameters, or set the default keyword arguments
    #
    graphite_a_const = kwargs.get('graphite_a_const',2.4175)  # Graphite lattice constant a
    graphite_c_const = kwargs.get('graphite_c_const',3.358)   # Graphite lattice constant c

    box_x = kwargs.get('box_x',100)          # Structure size in angstroms
    box_y = kwargs.get('box_y',100)          # Structure size in angstroms
    box_z = kwargs.get('box_z',100)          # Structure size in angstroms
    center_z = kwargs.get('center_z',True)   # Bool, center structure

    slab_thickness = kwargs.get('slab_thickness',6)    # Number of graphene layers in each slab
    num_voronoi = kwargs.get('num_voronoi',3)           # Number of voronoi cells in each slab

    num_slab_layers = kwargs.get('num_slab_layers',1)   # Number of layers of graphite slabs

    grain_cutoff = kwargs.get('grain_cutoff',0.2)  # Cutoff for the Filter out 'close' atoms in grain boundary
    #
    # ---------------------------------------------------------------------

    # C-C bond len
    a = float(graphite_a_const)/math.sqrt(3)
    #print(a)

    # box size
    L=numpy.array([int(box_x), int(box_y)])
    # num grains
    N=int(num_voronoi)

    # coordinate arrays
    cr_a = []
    cr_b = []

    # loop over slab layers
    for slab in range(int(num_slab_layers)):
    
        # Voronoi centers (a plane)
        x0_a, y0_a = pc.ptsInBox(N, L)
    
        # Random rotation matrix
        axes = gen_random_rot_matrix(N)
    
        # Voronoi centers (b plane)
        x0_b = numpy.zeros(len(x0_a))
        y0_b = numpy.zeros(len(y0_a))
        for i in range(len(x0_a)):
            x0_b[i] = x0_a[i] - axes[i][1][0]*a
            y0_b[i] = y0_a[i] + axes[i][0][0]*a

        # Atom coordinates (a plane)
        tr_a = pc.periodicVoronoiCell(a=a, L=L, N=N, x0=x0_a, y0=y0_a, axes=axes, cutoff=grain_cutoff)
        tr_rel = pc.cvtRelax(tr_a, verbose=False, tol=1E-5)
        cr_a.append(pc.tr2gr(tr_rel))
        

        # Atom coordinates (b plane)
        tr_b = pc.periodicVoronoiCell(a=a, L=L, N=N, x0=x0_b, y0=y0_b, axes=axes, cutoff=grain_cutoff)
        tr_rel = pc.cvtRelax(tr_b, verbose=False, tol=1E-5)
        cr_b.append(pc.tr2gr(tr_rel))


    # generate the composite structure

    # empty position array
    pos = []

    # set initial z value
    if(center_z):
        z = -graphite_c_const + box_z/2.0 - (slab_thickness * num_slab_layers -1) * graphite_c_const / 2.0
    else:
        z = -graphite_c_const


    # build array of atom coordinates
    for slab in range(int(num_slab_layers)):
        for j in range(slab_thickness):
            z = z + graphite_c_const
            if(j % 2 == 0):
                for i in range(len(cr_a[slab])):
                    pos.append([( cr_a[slab].positions[i,0], cr_a[slab].positions[i,1], z)])
            else:
                for i in range(len(cr_b[slab])):
                    pos.append([( cr_b[slab].positions[i,0], cr_b[slab].positions[i,1], z)])

    pos = numpy.reshape(pos, (len(pos), 3))
    #print len(pos)

    # return ase atoms object
    return Atoms(symbols=['C'] * len(pos), positions=pos, cell=[box_x,box_y,box_z], pbc=[True, True, True])




if __name__ == '__main__':
    print("\n>> Generating a default graphite poly crystalline structure\n")
    
    cr = GenGraphitePolyCrystal()
    
    writeLBOMDData(cr,'default_GraphitePolyCrystal.dat')
    io.write('default_GraphitePolyCrystal.pdb',cr)
    #io.write('default_GraphitePolyCrystal.cfg',cr)

    print(">> Done !  now minimise with AIREBO to complete \n")

