import numpy as np
# a = np.array([[1,2], [3,4]])
# print(a)


# idx = (a >= 2) & (a <=3)
# print(idx)

# print(a[idx])

# print([False] * 3)


a = np.asarray([[0,0,0], [0,1,0]])
sol = a[0, :]
print(np.nonzero(sol)[0][0])