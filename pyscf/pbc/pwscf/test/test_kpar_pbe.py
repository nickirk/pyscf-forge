import numpy as np
from pyscf.pbc import gto
from pyscf.pbc.pwscf import krks
import time

# Get MPI rank for logging control
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
except ImportError:
    rank = 0

def run_diamond():
    cell = gto.Cell(
        atom = "C 0 0 0; C 0.89169994 0.89169994 0.89169994",
        a = np.asarray([
                [0.       , 1.78339987, 1.78339987],
                [1.78339987, 0.        , 1.78339987],
                [1.78339987, 1.78339987, 0.        ]]),
        basis="gth-szv",
        ke_cutoff=100,
        pseudo="gth-pade",
    )
    cell.build()
    # Only print on rank 0
    cell.verbose = 4 if rank == 0 else 0

    kmesh = [4, 4, 4]
    kpts = cell.make_kpts(kmesh)
    
    start_time = time.time()
    mf = krks.PWKRKS(cell, kpts, xc="PBE", ecut_wf=50)
    mf.nvir = 4  # converge first 4 virtual bands
    mf.kernel()
    end_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"Total Energy: {mf.e_tot:.12f}")
    print(f"Execution Time: {end_time - start_time:.4f} seconds")
    print(f"{'='*60}\n")
    return mf.e_tot

if __name__ == "__main__":
    run_diamond()
