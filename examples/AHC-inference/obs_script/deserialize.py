# This file will be used to serialize data from Dan Parker's SCHF code

import h5py
import numpy as np

class SCHFData():
    """
        Attributes:
            self.E : stores various energies
            self.rho_SCHF : stores converged density matrix
            self.ks : stores the value of the planewaves

    """

    def __init__(self, filename, initial_state):
        self.read_state(filename, initial_state)

    def read_state(self, filename, initial_state):
        """Reads out data about a particular initial state from an `.h5` file.

        Args:
            filename (str): The file name of the `.h5` file to be read.
            initial_state (int): The index of the initial state to be read. 
        """
        with h5py.File(filename, "r") as f:
            g_initial_state = f[f"initial_state_{initial_state}"]
            g_SCHF = g_initial_state["SCHF_bandstructure"]
            g_sp = g_initial_state["singleparticle_bandstructure"]

            try:
                # an earlier version of the code had a typo
                g_manybody = g_initial_state["Maby_body_parameters"]
            except:
                g_manybody = g_initial_state["Many_body_parameters"]

            # get phi data
            # The original Julia data was shaped [orbital, planewave, bands, k1ind, k2ind]
            # After reshape, it becomes [orbital*planewave, bands, k1ind, k2ind]
            # In python, becomes of the difference between row and column ordering,
            # phi is shaped [k2ind, k1ind, bands, planewave * orbital]
            # we will transpose it to be labeld by [k1ind, k2ind, bands, planeave, orbital]
            Norbital = 2 # hard coded for two band model
            phi_sp = g_sp["phi"][()]
            Nk2, Nk1, Nbands, Ncombined = phi_sp.shape
            Nplanewave = Ncombined // Norbital
            phi_sp = phi_sp.reshape(Nk2, Nk1, Nbands, Nplanewave, Norbital)
            self.phi_sp = phi_sp.transpose([1, 0, 2, 3, 4]) #swap k1 and k2

            phi_SCHF = g_SCHF["phi"][()]
            phi_SCHF = phi_SCHF.reshape(Nk2, Nk1, Nbands, Nplanewave, Norbital)
            self.phi_SCHF = phi_SCHF.transpose([1, 0, 2, 3, 4]) #swap k1 and k2

            # get momentum data
            self.G1 = g_SCHF["model"]["lattice"]["G1"][()]
            self.G2 = g_SCHF["model"]["lattice"]["G2"][()]

            # get bz data
            Ks = g_SCHF["bz"]["Ks"][()]
            # the first and second index are swapped due to python vs julia ordering
            # Current ordering is therefore [k2, k1, vector]
            # We will turn this to [k1, k2, vector] ordering by swapping the first two components
            self.Ks_in_BZ = Ks.transpose([1, 0, 2])
            self.Nk = g_SCHF["bz"]["Nk"][()]
            
            # get shell structure data
            self.Gmn = g_SCHF["model"]["ss"]["Gmn"][()]
            self.mn = g_SCHF["model"]["ss"]["mn"][()]

            # get interaction data
            self.lambda_s = g_SCHF["model"]["lambda_s"][()]
            self.rs = g_manybody["interaction_potential"]["rs"][()]
            try:
                self.delta = g_SCHF["model"]["Delta"][()]
            except:
                print("could not load delta from .h5 file. Resorting to default value 200")
                self.delta = 200

            try:
                self.gate_distance = g_manybody["interaction_potential"]["gate_distance"][()]
                self.is_interaction_scale_fixed = g_initial_state["summary"]["is_interaction_scale_fixed"]
            except:
                pass

            # get energy data
            E_SCHF = g_initial_state["summary"]["E_SCHF"][()]
            E_kin = g_initial_state["summary"]["E_kin"][()]
            E_coulomb = g_initial_state["summary"]["E_coulomb"][()]

            self.energies = {}
            self.energies["Total"] = E_SCHF
            self.energies["E_kin"] = E_kin
            self.energies["E_coulomb"] = E_coulomb

            # get density matrix in the band basis
            rho_SCHF_band_basis = g_initial_state["SCHF_data"]["rho"][()]
            # again, the julia convention was to save it in (Nband, Nband, Nk1, Nk2) shaped tensor
            # Now it has been converted to (Nk2, Nk1, Nband, Nband)
            # We will act with transpose to make k1 and k2 swapped, and bra and ket swapped to the correct position
            self.rho_SCHF_band_basis = rho_SCHF_band_basis.transpose([1, 0, 3, 2]) # [Nk1, Nk2, Nband, Nband]

        # some post processing to get the data format right
        # phis is an Nx * Ny * N_bands * N_planewaves * N_components shape tensor
        # self.ks is an Nx * Ny * (N_planewaves * N_components) shape tensor which stores the value of the plane wave
        Nk1, Nk2, _, N_planewaves, _ = self.phi_sp.shape
        self.ks_for_all_plane_waves = np.zeros((Nk1, Nk2, N_planewaves, 2))
        for x in range(Nk1):
            for y in range(Nk2):
                for ind1 in range(N_planewaves):
                    # k_ref = self.G1 * (x - (Nx)//2)/Nx + self.G2 * (y - (Ny)//2)/Ny # take care of even and odd cases correctly
                    k_ref = self.Ks_in_BZ[x, y]
                    # ind2 = ind1//2 # map index for (N_planewaves * N_components) to (N_planewaves)
                    self.ks_for_all_plane_waves[x, y, ind1] = k_ref + self.Gmn[ind1]

        # convert rho to the planewave basis
        self.rho_SCHF_planewavebasis = np.einsum("ijaop, ijab, ijbmq -> ijopmq", self.phi_sp, self.rho_SCHF_band_basis, self.phi_sp.conj())
        # shaped [k1, k2, planewave, orbital, planewave, orbital]

        # Compute various energies
        E_2DEG, E_diag, E_offdiag = self.compute_kinetic_energies()
        self.energies["2DEG"] = E_2DEG
        self.energies["E_diag"] = E_diag
        self.energies["E_offdiag"] = E_offdiag

        # Some sanity checks
        # Checking the first axis and second axis correspond to G1 and G2
        assert np.all(np.isclose(self.Ks_in_BZ[1, 0] - self.Ks_in_BZ[0, 0], self.G1/self.Nk[0]))
        assert np.all(np.isclose(self.Ks_in_BZ[0, 1] - self.Ks_in_BZ[0, 0], self.G2/self.Nk[1]))

    def compute_kinetic_energies(self):
        # First, let's compute the kinetic energy to be sure we know the energy correctly
        Nk1, Nk2, Nbands, N_planewaves, Norbital = self.phi_sp.shape

        H = np.zeros((Nk1, Nk2, N_planewaves, Norbital, Norbital), dtype=complex)
        H_2DEG = np.zeros_like(H)
        H_diag = np.zeros_like(H)
        H_offdiag = np.zeros_like(H)

        for k1 in range(Nk1):
            for k2 in range(Nk2):
                for pl in range(N_planewaves):
                    for orbital in range(Norbital):
                        kvec = self.ks_for_all_plane_waves[k1, k2, pl]
                        k = kvec[0] + 1.j*kvec[1]
                        kmag = np.linalg.norm(kvec)
                        H_2DEG[k1, k2, pl, [0, 1], [0, 1]] = [1/self.rs**2 * kmag**2] * 2
                        #TODO need to multiply by Delta
                        H_diag[k1, k2, pl, 1, 1] = self.delta
                        H_diag[k1, k2, pl, 0, 0] = self.lambda_s**2 * kmag**2 * self.delta
                        #TODO make sure the sign is right
                        H_offdiag[k1, k2, pl, 0, 1] = -self.lambda_s * k.conj() * self.delta
                        H_offdiag[k1, k2, pl, 1, 0] = H_offdiag[k1, k2, pl, 0, 1].conj()

        self.H_2DEG = H_2DEG

        E_2DEG = np.einsum("ijkom, ijkmko ->", H_2DEG, self.rho_SCHF_planewavebasis)
        E_diag = np.einsum("ijkom, ijkmko ->", H_diag, self.rho_SCHF_planewavebasis)
        E_offdiag = np.einsum("ijkom, ijkmko ->", H_offdiag, self.rho_SCHF_planewavebasis)
        #assert np.isclose((E_2DEG+E_diag+E_offdiag) / np.prod(self.Nk), self.energies["E_kin"])

        return E_2DEG, E_diag, E_offdiag
 

        




    

