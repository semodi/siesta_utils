import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


def triu_to_full(m_triu):
    """ Restore full symmetric, quadratic matrix
        m_full from its upper diagonal m_triu. 
        
        Parameters:
        -----------
        m_triu: rank 1 np.array, triangular part of matrix
        
        Returns:
        --------
        m_full: rank 2 np.array, full matrix  
    """    

    m = len(m_triu)
    n = (-1 + np.sqrt(1 + 8*m))*0.5

    if n != int(n):
        raise Exception('Not triang. matrix')

    n = int(n)
    m_full = np.zeros([n,n])
    m_full[np.triu_indices(n)] = m_triu
    m_full.T[np.triu_indices(n)] = m_triu

    return m_full

def full_to_triu(m_full):
    """ Reduce symmetric, quadratic matrix m_full
        to its upper diagonal. 
         
        Parameters:
        --------
        m_full: rank 2 np.array, full matrix  
        
        Returns:
        -----------
        m_triu: rank 1 np.array, triangular part of matrix
    """           

    if m_full.shape[0] != m_full.shape[1]:
        raise Exception('Matrix not quadratic')
    
    n = len(m_full)
    
    return m_full[np.triu_indices(n)]

def import_matrix(data_file, index_file = None):
    """ Reads a siesta format sparse matrix. The format corresponds 
        to the one in which density, overlap and hamiltonian matrix 
        are stored.
        
        Parameters
        ----------
        data_file: path to file that contains the matrix entries
        index_file: path to file that contains sparse matrix support
                    arrays. Default: None; if None use data_file 
                    for indexing
        
        Returns:
        ---------
        d_matrix: rank 2 np.array; non-sparse representation of 
                  imported array

    """
    #TODO: option to save matrix in scipy sparse matrix format

    if index_file == None:
        filepath = data_file
    else:
        filepath = index_file 
    
    dmf = np.genfromtxt(filepath)
    n_orb = int(dmf[0])
    rows = dmf[2: n_orb + 2]

    if np.allclose(rows.astype(int),rows):
        rows = rows.astype(int)
    else:
        raise Exception("Error in DMF file (Number of columns for each row)")

    n_entries = int(np.sum(rows))
    indexing = dmf[n_orb + 2: -n_entries]

    if np.allclose(indexing.astype(int),indexing):
        indexing = indexing.astype(int)
    else:
        raise Exception("Error in DMF file (Indexing)")


    entries = np.genfromtxt(data_file)
    d_matrix = np.zeros([n_orb,n_orb])

    cnt = 0
    for i, row in enumerate(rows):
        for c in range(row):
            j = indexing[cnt] - 1
            d_matrix[i,j] = entries[cnt]
            cnt += 1

    if not np.allclose(d_matrix.T,d_matrix):
        print('Warning! Density Matrix not Symmetric')

    return d_matrix


def diagonals(D, n_molecules, n_m_orb):
    """ Given the full (density) matrix,
        return the block matrices on the diagonal (intra-molecular)
        
        Parameters
        ----------
        D: rank 2 np.array; non-sparse representation of 
           a siesta rank 2 tensor (possibly density matrix)

        n_molecules: integer; number of molecules
        n_m_orb: number of orbitals per molecule 
        
        Returns:
        --------

        diagonals: [rank 2 np.arrays]; list of diagonal blocks
    """


    diagonals = []
    for m in range(n_molecules):
        d_matrix = D[m * n_m_orb : (m+1) * n_m_orb, m * n_m_orb : (m+1) * n_m_orb]

        diagonals.append(d_matrix[np.triu_indices(n_m_orb)])

    return diagonals


def off_diagonals(D, n_molecules, n_m_orb):
    """ Given the full (density) matrix,
        return the block matrices on the off-diagonal (inter-molecular)
        
        Parameters
        ----------
        D: rank 2 np.array; non-sparse representation of 
           a siesta rank 2 tensor (possibly density matrix)

        n_molecules: integer; number of molecules
        n_m_orb: number of orbitals per molecule 
        
        Returns:
        --------

        off_diagonals: [rank 2 np.arrays]; list of off-diagonal blocks
    """
    off_diagonals = []
    for m1 in range(n_molecules - 1):
        for m2 in range(m1 + 1, n_molecules):

            d_matrix = D[m1 * n_m_orb : (m1+1) * n_m_orb, m2 * n_m_orb : (m2+1) * n_m_orb]

            off_diagonals.append(d_matrix.flatten())



    return off_diagonals
