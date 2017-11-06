"""Utility functions for real-space grid properties
"""

import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd

rho = np.zeros(2)
rho_val = np.zeros(2)
unitcell = np.zeros(2)
grid = np.zeros(2)

AtoBohr = 1.889725989
Dtoau = 0.393430307


# ================= Data import ====================== #

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


def mesh_3d(rmin=[0, 0, 0], rmax=0, scaled = False):
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
        mid_grid = (grid / 2).astype(int)
        rmax = mid_grid

    # resolve the periodic boundary conditions
    x_pbc = list(range(-rmax[0], -rmin[0])) + list(range(rmin[0], rmax[0] + 1))
    y_pbc = list(range(-rmax[1], -rmin[1])) + list(range(rmin[1], rmax[1] + 1))
    z_pbc = list(range(-rmax[2], -rmin[2])) + list(range(rmin[2], rmax[2] + 1))

    Xm, Ym, Zm = np.meshgrid(x_pbc, y_pbc, z_pbc)
    if scaled:
        Z = Zm * unitcell[2, 2] / grid[2]
        Y = Ym * unitcell[1, 1] / grid[1]
        X = Xm * unitcell[0, 0] / grid[0]
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
    """Calculates the quadrupole moment in Debye of a given charge distribution

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

