import numpy as np
import tenseal as ts
# TenSEAL context setup
context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60])
context.generate_galois_keys()
context.global_scale = 2**40

class TrmmImplementation:
    def __init__(self, context, M, N):
        self.context=context
        self.M = M
        self.N = N

    def initialize_arrays(self, alpha):
        A = np.zeros((self.M, self.M))
        B = np.zeros((self.M, self.N))

        for i in range(self.M):
            for j in range(i):
                A[i][j] = (i + j) % self.M / self.M
            A[i][i] = 1.0
            for j in range(self.N):
                B[i][j] = (self.N + (i - j)) % self.N / self.N

        return A, B

    def trmm_kernel(self, alpha, A, B):
      
        for i in range(self.M):
            for j in range(self.N):
                v1=[]
                v2=[]
                for k in range(i + 1, self.M):
                    v1.append(A[k][i])
                    v2.append(B[k][j])
                    enc_v1=ts.ckks_tensor(context, v1)
                    enc_v2=ts.ckks_tensor(context, v2)
                    mul=enc_v1.dot(enc_v2) 	
                    B[i][j] += mul.decrypt().tolist()
                B[i][j] = alpha * B[i][j]

    def run_benchmark(self, alpha):
        A, B = self.initialize_arrays(alpha)
        self.trmm_kernel(alpha, A, B)
        return B

# Example usage:
M = 2
N = 2
alpha = 1.5

trmm_impl = TrmmImplementation(context, M, N)
result = trmm_impl.run_benchmark(alpha)
print(result)

