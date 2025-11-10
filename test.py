import numpy as np



# a = np.arange(25).reshape(5,5)
# b = np.arange(5)
# c = np.arange(6).reshape(2,3)


# np.einsum('ii', a)



mm = [[1,2,3],[6,5,4],[8,9,7]]
a = [1,2,3,4]
b = [5,4,3,2]
ee = [row for row in mm]


eee = [x for row in mm for x in row]
# equvalent to:
eee = []
for row in mm:
    for x in row:
        eee += [x]

print("finish")