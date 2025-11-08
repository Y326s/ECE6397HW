# Midterm

# Problem 2
def f_x(x):
    """
    A given function in the figure
    """
    return fx

x0 = 0.5
step_size = 0.1
delta = 1e-5
decay = 0.98
criterion = 1e-5

x = x0
dx = 100
ii = 0
while abs(dx) >= criterion and  ii <= 100:
    gradient = (f_x(x+delta) - f_x(x)) / delta
    dx = -step_size*gradient
    x = x + dx

    step_size = step_size*decay
    if ii == 100:
        print("Waring: Reach maximum iteration, Exit.")
    ii = ii + 1
print("minimum at: {:.4f}".format(x))
print("Finish!")






import numpy as np
# Problem 5
# Inputs
x1 = np.array([2,4,3,5])
x2 = np.array([1,2,4,1])
d = len(x1)

Wq = np.array([[0.73,0.28,0.91],
               [0.45,0.82,0.17],
               [0.64,0.39,0.56],
               [0.12,0.88,0.71]])

Wk = np.array([[0.81,0.15,0.67],
               [0.29,0.94,0.42],
               [0.53,0.38,0.76],
               [0.91,0.22,0.58]])

Wv = np.array([[0.47,0.82,0.19],
               [0.65,0.31,0.88],
               [0.14,0.73,0.56],
               [0.92,0.27,0.41]])

# Quiry, Key, and Value

q1 = np.matmul(x1,Wq)
k1 = np.matmul(x1,Wk)
v1 = np.matmul(x1,Wv)
print("Query 1: ", q1, "Key 1: ", k1, "Value 1: ", v1)
q2 = np.matmul(x2,Wq)
k2 = np.matmul(x2,Wk)
v2 = np.matmul(x2,Wv)
print("Query 2: ", q2, "Key 2: ", k2, "Value 2: ", v2)
print("--------------------------------")

# Score
score1 = np.outer(k1,q1)
score2 = np.outer(k2,q2)
print("Score 1: ")
print(score1)
print("Score 2: ")
print(score2)
print("--------------------------------")

# divided by sqrt(d)
e1 = score1/np.sqrt(d)
e2 = score2/np.sqrt(d)
print("Score 1 divided by sqrt(d): ")
print(e1)
print("Score 2 divided by sqrt(d): ")
print(e2)
print("--------------------------------")

# Attention (Apply softmax)
a1 = np.exp(e1)/np.sum(np.exp(e1))
a2 = np.exp(e2)/np.sum(np.exp(e2))
print("Attention 1 (Apply softmax): ")
print(a1)
print("Attention 2 (Apply softmax): ")
print(a2)
print("--------------------------------")

# Output y
y1 = np.matmul(v1,a1)
y2 = np.matmul(v2,a2)
print("Output 1 (y1): ")
print(y1)
print("Output 2 (y2): ")
print(y2)
print("--------------------------------")
print("Finish!")