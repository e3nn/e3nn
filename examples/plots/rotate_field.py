import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import numpy as np

# vector field


def f(X, Y):
    U = np.maximum(0.3, X + 1)
    V = 0.5 * Y
    return 0.07 * U, 0.07 * V

# action of the group


def act(X, Y, a, tx=0, ty=0):
    newX = np.cos(a) * X - np.sin(a) * Y + tx
    newY = np.sin(a) * X + np.cos(a) * Y + ty
    return newX, newY


x = np.linspace(-1, 1, 6)
X, Y = np.meshgrid(x, x)
a = np.pi / 2

quiver_param = {
    'units': 'width',
    'scale': 1,
    'pivot': 'middle',
    'headwidth': 2.5,
    'headlength': 5
}

br = 0.02
bb = 0.25
h = 3.3
# (1 - bb) h == w (1/3 - 2 br)
plt.figure(figsize=(h * (1 - bb), h * (1/3 - 2 * br)))

view = 1.2
ty = -1.8 # text position

plt.axes([0, bb, 1/3 - 2*br, 1 - bb])
plt.gca().axis('off')
U, V = f(X, Y)
plt.quiver(X, Y, U, V, **quiver_param)
plt.xlim(-view, view)
plt.ylim(-view, view)
plt.text(0, ty, r"$f(x)$", horizontalalignment='center')

plt.axes([1/2 - 1/6 + br, bb, 1/3 - 2*br, 1 - bb])
plt.gca().axis('off')
U, V = f(*act(X, Y, -a))
plt.quiver(X, Y, U, V, **quiver_param)
plt.xlim(-view, view)
plt.ylim(-view, view)
plt.text(0, ty, r"$f(g^{-1} x)$", horizontalalignment='center')

plt.axes([1 - 1/3 + 2*br, bb, 1/3 - 2*br, 1 - bb])
plt.gca().axis('off')
U, V = act(*f(*act(X, Y, -a)), a)
plt.quiver(X, Y, U, V, **quiver_param)
plt.xlim(-view, view)
plt.ylim(-view, view)
plt.text(0, ty, r"$\rho(g) f(g^{-1} x)$", horizontalalignment='center')


plt.savefig("rotate_field.pdf", transparent=True)
