import numpy as np

def KL_Distance(H1, H2):
    sum = 0
    for i in range(len(H1)):
        sum += H1[i] * np.log2(H1[i]/H2[i])
    return sum

def Bhattacharyya_Distance(H1, H2):
    sum = 0
    for i in range(len(H1)):
        sum += np.sqrt(H1[i] * H2[i])
    return -np.log(sum)

H1 = [ 0.24, 0.2, 0.16, 0.12, 0.08, 0.04, 0.12, 0.04]
H2 = [ 0.22, 0.23, 0.16, 0.13, 0.11, 0.08, 0.05, 0.02]

print("KL Distance(H1,H2): ", KL_Distance(H1, H2))
print("KL Distance(H2,H1): ", KL_Distance(H2, H1))
print("Bhattacharyya Distance: ", Bhattacharyya_Distance(H1, H2))