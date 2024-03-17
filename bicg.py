import tenseal as ts
import numpy as np
import time as t

class Bicg:
    def __new__(cls, context, M, N):
        return object.__new__(cls)
        
    def __init__(self, context, M, N):
        self.context=context
        self.M = M
        self.N = N

    def print_array_custom(self, array, name):
        if name == 's':
            loop_bound = self.M
        else:
            loop_bound = self.N

        for i in range(0, loop_bound):
            if i % 20 == 0:
                print('\n')
            print(array[i])

    def initialize_array(self, M, N):
        A = [[0] * M for _ in range(N)]
        s = [0] * M
        q = [0] * N
        p = [(i % M) / M for i in range(M)]
        r = [(i % N) / N for i in range(N)]

        for i in range(N):
            for j in range(M):
                A[i][j] = (i * (j + 1) % N) / N

        return A, s, q, p, r

    def kernel(self, A, s, q, p, r):
        A = np.array(A)
        s = np.array(s, dtype=float)
        q = np.array(q, dtype=float)  # Ensure q is of type float
        p = np.array(p)
        r = np.array(r)
        t1 = t.time()
        # Reset s
        s[:] = 0

        # Calculate s without loop
        s[:] = r.dot(A)

        # Calculate q without loop
        q[:] = 0.0
        q[:] = A.dot(p)
        t2 = t.time()
        print("Normal time is ", t2-t1)

        enc_A = ts.ckks_tensor(self.context, A)
        enc_p = ts.ckks_tensor(self.context, p)
        enc_r = ts.ckks_tensor(self.context, r)
        
        t1_enc = t.time()
        # Calculate s and q using encrypted tensors
        enc_s = enc_r.dot(enc_A)
        enc_q = enc_A.dot(enc_p)
        t2_enc = t.time()
        print("Encrypted time is ", t2_enc - t1_enc)

        # Decrypt results
        s[:] = enc_s.decrypt().tolist()
        q[:] = enc_q.decrypt().tolist()

        return s, q

# TenSEAL context setup
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60])
context.generate_galois_keys()
context.global_scale = 2**40

# Example usage:
M = int(input("Enter M "))
N = int(input("Enter N "))
bicg = Bicg(context, M, N)
A, s, q, p, r = bicg.initialize_array(M, N)
s_result, q_result = bicg.kernel(A, s, q, p, r)
print("s:", s_result)
print("q:", q_result)

