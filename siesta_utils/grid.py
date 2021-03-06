"""Utility functions for real-space grid properties
"""

import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import struct
from .conversions import *
from scipy.interpolate import griddata
rho = np.zeros(2)
rho_val = np.zeros(2)
unitcell = np.zeros(2)
grid = np.zeros(2)



# ================= Data import ====================== #
def get_data_bin(file_path):

    global rho
    global unitcell
    global grid

    #Warning: Only works for cubic cells!!!
    #TODO: Implement for arb. cells

    bin_file = open(file_path, mode = 'rb')

    unitcell = '<I9dI'
    grid = '<I4iI'


    unitcell = np.array(struct.unpack(unitcell,
        bin_file.read(struct.calcsize(unitcell))))[1:-1].reshape(3,3)

    grid = np.array(struct.unpack(grid,bin_file.read(struct.calcsize(grid))))[1:-1]
    if (grid[0] == grid[1] == grid[2]) and grid[3] == 1:
        a = grid[0]
    else:
        raise Exception('get_data_bin cannot handle non-cubic unitcells or spin')

    block = '<' + 'I{}fI'.format(a)*a*a
    content = np.array(struct.unpack(block,bin_file.read(struct.calcsize(block))))

    rho = content.reshape(a+2, a, a, order = 'F')[1:-1,:,:]
    return rho, unitcell, grid

def get_data(file_path):
    """Import data from RHO file (or similar real-space grid files)
    Data is saved in global variables.

    Structure of RHO file:
    first three lines give the unit cell vectors
    fourth line the grid dimensions
    subsequent lines give density on grid

    Parameters:
    -----------

    file_path: string; path to RHO file from which density is read

    Returns:
    --------
    None

    Other:
    ------
    unitcell: (3,3) np.array; saves the unitcell dimension in euclidean coordinates
    grid: (,3) np.array; number of grid points in each euclidean direction
    rho: (grid[1],grid[2],grid[3]) np.array; density on grid
    """
    global rho
    global unitcell
    global grid
    global rhopath
    rhopath = file_path
    unitcell = np.zeros([3, 3])
    grid = np.zeros([4])

    with open(file_path, 'r') as rhofile:

        # unit cell (in Bohr)
        for i in range(0, 3):
            unitcell[i, :] = rhofile.readline().split()

        grid[:] = rhofile.readline().split()
        grid = grid.astype(int)
        n_el = grid[0] * grid[1] * grid[2] * grid[3]

        # initiatialize density with right shape
        rho = np.zeros(grid)

        for z in range(grid[2]):
            for y in range(grid[1]):
                for x in range(grid[0]):
                    rho[x, y, z, 0] = rhofile.readline()

    # closed shell -> we don't care about spin.
    rho = rho[:, :, :, 0]
    return rho, unitcell, grid

def check_norm():
    """ Check normalization of charge density

        Returns
        -------
        float; integrated charge density
    """
    Xm, Ym, Zm = mesh_3d()
    box_vol = unitcell[0, 0] / grid[0] * unitcell[1, 1] / grid[1] * unitcell[
        2, 2] / grid[2]
    return np.sum(rho[Xm, Ym, Zm]) * box_vol

# ==================== Mesh Functions ==================== #

def smallest_box(atom_pos, box_buffer=0.5):
    """Determine smallest box that includes all molecules.
       Called by fit_poly if rmax = -1

       Parameters
       ----------
       atom_pos: (,3) np.array; atomic coordinates
       box_buffer: float; buffer around smallest box

       Returns
       --------
       rmax: (3) list; the maximum box dimensions in 3 euclid. directions
    """

    rmax = [0, 0, 0]
    for a in atom_pos:
        for i in range(3):
            if abs(a[i]) > rmax[i]:
                rmax[i] = abs(a[i])
    for i in range(3):
        rmax[i] = (int)((rmax[i] + box_buffer) * grid[i] / unitcell[i, i])
        if rmax[i] > grid[i]:
            rmax[i] = grid[i]
    return rmax


def plane_cut(data,
              plane,
              height,
              unitcell,
              grid,
              rmin=[0, 0, 0],
              rmax=0,
              return_mesh=False):

    """return_mesh = False : returns a two dimensional cut through 3d data
                     True : instead of data, 2d mesh is returned

      Parameters:
      ----------
         data
         plane = {0: yz-plane, 1: xz-plane, 2:xy-plane}
         unitcell = 3x3 array size of the unitcell
         grid = 3x1 array size of grid
         rmin,rmax = lets you choose the min and max grid cutoff
                       rmax = 0 means the entire grid is used
         return_mesh = boolean; decides wether mesh or cut through data is returned
    """

    if rmax == 0:
        mid_grid = (grid / 2).astype(int)
        rmax = mid_grid

    # resolve the periodic boundary conditions
    x_pbc = list(range(-rmax[0], -rmin[0])) + list(range(rmin[0], rmax[0]))
    y_pbc = list(range(-rmax[1], -rmin[1])) + list(range(rmin[1], rmax[1]))
    z_pbc = list(range(-rmax[2], -rmin[2])) + list(range(rmin[2], rmax[2]))
    height = (int)(np.round(height * grid[plane] / unitcell[plane, plane]))

    pbc_grids = [x_pbc, y_pbc, z_pbc]
    pbc_grids.pop(plane)

    A, B = np.meshgrid(*pbc_grids)

    indeces = [A, B]
    indeces.insert(plane, height)
    if not return_mesh:
        return data[indeces[0], indeces[1], indeces[2]]
    else:
        return A, B


def mesh_3d(rmin=[0, 0, 0], rmax=0, scaled = False, pbc = True, indexing = 'xy'):
    """Returns a 3d mesh taking into account periodic boundary conditions

        Parameters
        ----------
        rmin, rmax: (3) list; lower and upper cutoff
        scaled: boolean; scale the meshes with unitcell size?

        Returns
        -------
        X, Y, Z: np.arrays; meshgrid
    """

    if rmax == 0:
        mid_grid = np.floor(grid / 2).astype(int)
        rmax = mid_grid

    # resolve the periodic boundary conditions
    if pbc:
        x_pbc = list(range(-rmax[0], -rmin[0])) + list(range(rmin[0], rmax[0]+1))
        y_pbc = list(range(-rmax[1], -rmin[1])) + list(range(rmin[1], rmax[1]+1))
        z_pbc = list(range(-rmax[2], -rmin[2])) + list(range(rmin[2], rmax[2]+1))
    else:
        x_pbc = list(range(rmin[0], rmax[0] +1 )) + list(range(-rmax[0], -rmin[0]))
        y_pbc = list(range(rmin[1], rmax[1] +1 )) + list(range(-rmax[1], -rmin[1]))
        z_pbc = list(range(rmin[2], rmax[2] +1 )) + list(range(-rmax[2], -rmin[2]))


    Xm, Ym, Zm = np.meshgrid(x_pbc, y_pbc, z_pbc, indexing = indexing)

    U = np.array(unitcell) # Matrix to go from real space to mesh coordinates
    for i in range(3):
        U[i,:] = U[i,:] / grid[i]

    a = np.linalg.norm(unitcell, axis = 1)/grid[:3]

    Rm = np.concatenate([Xm.reshape(*Xm.shape,1),
                         Ym.reshape(*Xm.shape,1),
                         Zm.reshape(*Xm.shape,1)], axis = 3)

    if scaled:
        R = np.einsum('ij,klmj -> iklm', U.T , Rm)
        X = R[0,:,:,:]
        Y = R[1,:,:,:]
        Z = R[2,:,:,:]

        # Z = Zm * unitcell[2, 2] / grid[2]
        # Y = Ym * unitcell[1, 1] / grid[1]
        # X = Xm * unitcell[0, 0] / grid[0]
        return X,Y,Z
    else:
        return Xm,Ym,Zm

# ================= Plotting ===================== #


def glimpse(rmin=[0, 0, 0], rmax=0, plane=2, height = 0):
    """Take a quick look at the loaded data in a particular plane

        Parameters
        ----------
        rmin,rmax: (3) list; upper and lower cutoffs
        plane = {0: yz-plane, 1: xz-plane, 2: xy-plane}
    """

    RHO = plane_cut(rho, plane, height, unitcell, grid, rmin=rmin, rmax=rmax)

    plt.figure()
    CS = plt.imshow(
        RHO, cmap=plt.cm.jet, origin='lower')
    plt.colorbar()
    plt.show()


def quadrupole_moment(X, Y, Z, V, coord, rho, diagonal = False, verbose = False):
    """Calculates the quadrupole moment in atomic units of a given charge distribution

    Parameters
    ----------
    X, Y, Z: np.array; Mesh arrays
    V: float; Volume of a grid cell
    coord: np.array; atomic coordinates, ordered like [O,H,H,O,H,...]
    n: int; number of gaussians
    diagonal: boolean; Only compute diagonal elements
    verbose: boolean; print Ionic and Electronic contribution
    """

    elec_quadrupole = np.zeros([3,3])

    meshes = [X,Y,Z]

    ionic_quadrupole = np.zeros([3,3])

    charge = [6,1,1] * int(len(coord)/3)

    for i in range(3):
        for j in range(3):
            for a,c in zip(coord,charge):
                if i == j:
                    ionic_quadrupole[i,j] -= c * np.linalg.norm(a)**2
                ionic_quadrupole[i,j] += 3 * c * a[i]*a[j]

    for i in range(3):
        for j in range(i, 3):
            if i == j:
                if i == 2: continue # Determine last diagonal entry by trace cond.

                rsq = np.zeros_like(meshes[0])
                for k in range(3):
                    rsq += (meshes[k])**2
                elec_quadrupole[i,j] -= np.sum(rsq * rho * V)
            elif diagonal: # Only calculate diagonal elements
                continue
            elec_quadrupole[i,j] += np.sum(3 * meshes[i] * meshes[j]  * rho * V)

    #Fill lower triangle
    if not diagonal:
        for i in range(3):
            for j in range(i):
                elec_quadrupole[i,j] = elec_quadrupole[j,i]

    elec_quadrupole[2,2] =  - elec_quadrupole[0,0] - elec_quadrupole[1,1]

    if diagonal: return (ionic_quadrupole - elec_quadrupole).diagonal()
    else: return  (ionic_quadrupole - elec_quadrupole)

def dipole_moment(X, Y, Z, V, coord, rho, verbose = False):
    """Calculates the dipole moment in Debye of a given charge distribution

    Parameters
    ----------
    X, Y, Z: np.array; Mesh arrays
    V: float; Volume of a grid cell
    coord: np.array; atomic coordinates, ordered like [O,H,H,O,H,...]
    par: [float]; Gaussian fitting parameters
    n: int; number of Gaussians
    verbose: boolean; print Ionic and Electronic contribution

    Returns
    --------
    float; Dipole moment in Debye
    """

    charge_com = np.array([ np.sum(mesh * rho * V) for mesh in [X,Y,Z]])

    coord = coord.reshape(-1,3,3)
    ionic_contrib = np.zeros(3)
    for a in coord:
        ionic_contrib += a[1] + a[2] + 6 * a[0]
    if verbose:
        print('Ionic {} [a.u.]'.format(ionic_contrib))
        print('Electronic {} [a.u.]'.format(charge_com))

    return (ionic_contrib - charge_com)/Dtoau

def rho_at_cartesian(xi, siesta, unit = 'A', method = 'linear'):

    if unit == 'A':
        xi *= AtoBohr
    elif unit != 'Bohr':
        raise Exception('Unit has to be either "A" or "Bohr"')

    if np.any(np.abs(xi) > siesta.unitcell[0,0]) or \
    np.any(np.abs(xi) > siesta.unitcell[1,1]) or \
    np.any(np.abs(xi) > siesta.unitcell[2,2]):
        raise Exception('xi out of bounds')

    # Grid size
    a = np.array([siesta.unitcell[i,i]/siesta.grid[i] for i in range(3)]).reshape(1,3)

    # Real space inquiry points to mesh
    Xi = np.round(xi/a).astype(int)

    # Find surrounding mesh points
    Xs = np.array(Xi)
    Xzeros = np.zeros_like(Xi)
    for x in [-1,0,1]:
        for y in [-1,0,1]:
            for z in [-1,0,1]:
                for i, entry in enumerate(np.array([x,y,z])):
                    Xzeros[:,i] = entry
                Xs = np.concatenate([Xs, Xi + Xzeros])
                Xzeros = np.zeros_like(Xi)
    Xs = np.unique(Xs, axis=0)

    # Surrounding mesh points in real space
    xs = Xs * a

    return griddata(xs, siesta.rho[Xs[:,0], Xs[:,1], Xs[:,2]], xi, method = method)
