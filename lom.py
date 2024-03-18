import numpy as np
import tenseal as ts

# TenSEAL context setup
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60])
context.generate_galois_keys()
context.global_scale = 2**40

class ArrayList:
    def __init__(self):
        self.arrays = []

    def append(self, array):
        if isinstance(array, np.ndarray) and array.ndim == 2:
            enc_v = ts.ckks_tensor(context, array)
            self.arrays.append(enc_v)
        else:
            raise ValueError("Input must be a 2D NumPy array.")

    def decrypt(self):
        decrypted_arrays = []
        for array in self.arrays:
            decrypted_array = array.decrypt().tolist()
            decrypted_arrays.append(decrypted_array)
        return decrypted_arrays

# Example usage:
if __name__ == "__main__":
    # Create an ArrayList object
    array_list = ArrayList()

    # Create some 2D NumPy arrays
    array1 = np.array([[1, 2, 3], [4, 5, 6]])
    array2 = np.array([[7, 8], [9, 10], [11, 12]])
    array3 = np.array([[13, 14, 15], [16, 17, 18]])

    # Append arrays to the ArrayList
    array_list.append(array1)
    array_list.append(array2)
    array_list.append(array3)

    # Decrypt the encrypted arrays
    decrypted_arrays = array_list.decrypt()
    for decrypted_array in decrypted_arrays:
        print(decrypted_array)

