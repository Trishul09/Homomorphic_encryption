import numpy as np
import tenseal as ts
import time as t

class Gesummv:
    def __new__(cls, context, N):
        return object.__new__(cls)
        
    def __init__(self, context, N):
        self.context = context
        self.N = N

    def initialize_arrays(self):
        self.A = np.zeros((self.N, self.N))
        self.B = np.zeros((self.N, self.N))
        self.tmp = np.zeros(self.N)
        self.x = np.zeros(self.N)
        self.y = np.zeros(self.N)

        for i in range(self.N):
            self.x[i] = np.float64(i % self.N) / self.N
            for j in range(self.N):
                self.A[i][j] = np.float64((i * j + 1) % self.N) / self.N
                self.B[i][j] = np.float64((i * j + 2) % self.N) / self.N

    def kernel(self, alpha, beta, A, B, tmp, x, y):
        # Convert lists to numpy arrays for tensor operations
        A = np.array(A)
        B = np.array(B)
        tmp = np.zeros(self.N)
        y = np.zeros(self.N)
        
        t1=t.time()
        tmp = np.dot(A, x) + tmp
        y = np.dot(B, x) + y
        y = alpha * tmp + beta * y
        t2=t.time()
        print("Normal time", t2-t1)
        
        t3=t.time()
        enc_A=ts.ckks_tensor(context, A)
        enc_B=ts.ckks_tensor(context, B)
        enc_x=ts.ckks_tensor(context, x)
       
        # Perform the computation using tensor operations
        tmp = np.dot(enc_A, enc_x) + tmp
        y = np.dot(enc_B, enc_x) + y
        y = alpha * tmp + beta * y
        t4=t.time()
        print("encryption computation time ", t4-t3)
        
        return y.decrypt().tolist()

    def run_benchmark(self, alpha, beta):
        self.initialize_arrays()
        result = self.kernel(alpha, beta, self.A, self.B, self.tmp, self.x, self.y)
        return result
        
# TenSEAL context setup
context = ts.Context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60])
context.generate_galois_keys()
context.global_scale = 2**40

# Example usage:
if __name__ == "__main__":
    N = int(input("Enter N size of matrix: "))
    alpha = 1.5
    beta = 1.2

    benchmark = Gesummv(context, N)
    result = benchmark.run_benchmark(alpha, beta)
    print(result)

