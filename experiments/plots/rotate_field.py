import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import numpy as np

# vector field


def f(X, Y):
    U = np.maximum(0.3, X + 1)
    V = 0.5 * Y
    return 0.04 * U, 0.04 * V

# action of the group


def act(X, Y, a, tx=0, ty=0):
    newX = np.cos(a) * X - np.sin(a) * Y + tx
    newY = np.sin(a) * X + np.cos(a) * Y + ty
    return newX, newY


x = np.linspace(-1, 1, 7)
X, Y = np.meshgrid(x, x)
a = np.pi / 2

quiver_param = {
    'units': 'width',
    'scale': 1,
    'pivot': 'middle',
    'headwidth': 2.5,
    'headlength': 5
}

br = 0.01
bb = 0.12
h = 4
# (1 - bb) h == w (1/3 - 2 br)
plt.figure(figsize=(h * (1 - bb), h * (1/3 - 2 * br)))


plt.axes([0, bb, 1/3 - 2*br, 1 - bb])
plt.gca().axis('off')
U, V = f(X, Y)
Q = plt.quiver(X, Y, U, V, **quiver_param)
plt.text(0, -1.3, r"$f(x)$", horizontalalignment='center')

plt.axes([1/2 - 1/6 + br, bb, 1/3 - 2*br, 1 - bb])
plt.gca().axis('off')
U, V = f(*act(X, Y, -a))
Q = plt.quiver(X, Y, U, V, **quiver_param)
plt.text(0, -1.3, r"$f(g^{-1} x)$", horizontalalignment='center')

plt.axes([1 - 1/3 + 2*br, bb, 1/3 - 2*br, 1 - bb])
plt.gca().axis('off')
U, V = act(*f(*act(X, Y, -a)), a)
Q = plt.quiver(X, Y, U, V, **quiver_param)
plt.text(0, -1.3, r"$\rho(g) f(g^{-1} x)$", horizontalalignment='center')


plt.savefig("rotate_field.pdf", transparent=True)
