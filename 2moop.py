import tenseal as ts
import numpy as np
import time as t

class _2mm:
    def __new__(cls, context, ni, nj, nk, nl, alpha, beta):
        return object.__new__(cls)

    def __init__(self, context, ni, nj, nk, nl, alpha, beta):
        self.context = context
        self.ni = ni
        self.nj = nj
        self.nk = nk
        self.nl = nl
        self.alpha = alpha
        self.beta = beta

    def initialize_arrays(self):
        # Initialize matrices A, B, C, D with ones
        A = np.ones((self.ni, self.nk))
        B = np.ones((self.nk, self.nj))
        C = np.ones((self.nj, self.nl))
        D = np.ones((self.ni, self.nl))
        for i in range(self.ni):
            for j in range(self.nk):
                A[i][j] = 1

        for i in range(self.nk):
            for j in range(self.nj):
                B[i][j] = 1

        for i in range(self.nj):
            for j in range(self.nl):
                C[i][j] = 1

        for i in range(self.ni):
            for j in range(self.nl):
                D[i][j] = 1

        return A, B, C, D

    def run_multiplication(self):
        A, B, C, D = self.initialize_arrays()

        # Perform classical matrix multiplication
        r1 = A.dot(B).dot(C)
        r1 *= self.alpha
        r2 = self.beta * D
        r3 = r1 + r2
        print("Classical multiplication time:", t.time() - t1)

        # Encrypt matrices
        enc_A = ts.ckks_tensor(self.context, A)
        enc_B = ts.ckks_tensor(self.context, B)
        enc_C = ts.ckks_tensor(self.context, C)
        enc_D = ts.ckks_tensor(self.context, D)

        # Perform encrypted matrix multiplication
        t3 = t.time()
        E = enc_A.dot(enc_B).dot(enc_C)
        E *= self.alpha
        F = self.beta * enc_D
        G = E.dot(F)
        t4 = t.time()
        print("Encrypted multiplication time:", t4 - t3)

        # Decrypt the result
        decrypted_G = G.decrypt()
        return decrypted_G

# TenSEAL context setup
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[40, 21, 21, 21, 21, 21, 21, 40])
context.generate_galois_keys()
context.global_scale = 2**40

# Initialize matrix sizes
ni = int(input("Enter ni: "))
nj = int(input("Enter nj: "))
nk = int(input("Enter nk: "))
nl = int(input("Enter nl: "))
alpha = 2
beta = 2

# Create MatrixMultiplication instance
matrix_mult = _2mm(context, ni, nj, nk, nl, alpha, beta)

# Perform matrix multiplication
t1 = t.time()
result = matrix_mult.run_multiplication()

# Print the decrypted result
print("Decrypted result:")
print(result.tolist())

