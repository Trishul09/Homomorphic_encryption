import tenseal as ts
import numpy as np
import time as t
class MatrixMultiplication:
    def __new__(cls, context, ni, nj, nk, nl, nm):
        return object.__new__(cls)

    def __init__(self, context, ni, nj, nk, nl, nm):
        # Initialize context and matrix dimensions
        self.context = context
        self.ni = ni
        self.nj = nj
        self.nk = nk
        self.nl = nl
        self.nm = nm

    def initialize_arrays(self):
        # Initialize matrices A, B, C, D with ones
        A = np.ones((self.ni, self.nk))
        B = np.ones((self.nk, self.nj))
        C = np.ones((self.nj, self.nm))
        D = np.ones((self.nm, self.nl))
        for i in range(self.ni):
            for j in range(self.nk):
                A[i][j] = (i * j + 1) % self.ni / (5 * self.ni)

        for i in range(self.nk):
            for j in range(self.nj):
                B[i][j] = (i * (j + 1) + 2) % self.nj / (5 * self.nj)

        for i in range(self.nj):
            for j in range(self.nm):
                C[i][j] = i * (j + 3) % self.nl / (5 * self.nl)

        for i in range(self.nm):
            for j in range(self.nl):
                D[i][j] = (i * (j + 2) + 2) % self.nk / (5 * self.nk)
  
        return A, B, C, D

    def run_multiplication(self):
        A, B, C, D = self.initialize_arrays()
        t1=t.time()
        result1=A.dot(B)
        result2=C.dot(D)
        t2=t.time()
        print(t2-t1)
        print()
        # Encrypt matrices
        enc_A = ts.ckks_tensor(self.context, A)
        enc_B = ts.ckks_tensor(self.context, B)
        enc_C = ts.ckks_tensor(self.context, C)
        enc_D = ts.ckks_tensor(self.context, D)
        
        t3=t.time()
        # Perform matrix multiplication
        E = enc_A.dot(enc_B)
        F = enc_C.dot(enc_D)
        G = E.dot(F)
       
        # Decrypt the result
        decrypted_G = G.decrypt()
        t4=t.time()
        print(t4-t3)
        return decrypted_G

# TenSEAL context setup
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60])
context.generate_galois_keys()
context.global_scale = 2**40

# Initialize matrix sizes
ni = int(input("enter ni "))
nj = int(input("input nj "))
nk = int(input("input nk "))
nl = int(input("input nl "))
nm = int(input("input nm "))


# Create MatrixMultiplication instance
matrix_mult = MatrixMultiplication(context, ni, nj, nk, nl, nm)

# Perform matrix multiplication
result = matrix_mult.run_multiplication()

# Print the decrypted result
print("Decrypted result:")
print(result.tolist())

