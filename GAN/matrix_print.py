
import numpy as np

# Load the .npy file
data1 = np.load('raw_fun.npy')
data2 = np.load('raw_rating.npy')
data3 = np.load('satis_matrix.npy')
data4 = np.load('fun_matrix.npy')


# Now, you can use the loaded data as a NumPy array
print('raw_fun',data1)
print('raw_rating',data2)
print('satis_matrix',data3)
print('fun_matrix',data4)