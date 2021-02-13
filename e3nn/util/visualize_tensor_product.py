import numpy as np


def _intersection(x, u, y, v):
    u2 = np.sum(u**2)
    v2 = np.sum(v**2)
    uv = np.sum(u * v)
    det = u2 * v2 - uv**2
    mu = np.sum((u * uv - v * u2) * (y - x)) / det
    return y + mu * v


def visualize_tensor_product(tp):
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    import matplotlib.patches as patches

    fig, ax = plt.subplots()

    # hexagon
    verts = [
        np.array([np.cos(a * 2 * np.pi / 6), np.sin(a * 2 * np.pi / 6)])
        for a in range(6)
    ]

    codes = [
        Path.MOVETO,
        Path.LINETO,

        Path.MOVETO,
        Path.LINETO,

        Path.MOVETO,
        Path.LINETO,
    ]

    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=1)
    ax.add_patch(patch)

    n = len(tp.irreps_in1)
    b, a = verts[2:4]

    c_in1 = (a + b) / 2
    s_in1 = [a + (i + 1) / (n + 1) * (b - a) for i in range(n)]

    n = len(tp.irreps_in2)
    b, a = verts[:2]

    c_in2 = (a + b) / 2
    s_in2 = [a + (i + 1) / (n + 1) * (b - a) for i in range(n)]

    n = len(tp.irreps_out)
    a, b = verts[4:6]

    s_out = [a + (i + 1) / (n + 1) * (b - a) for i in range(n)]

    for ins in tp.instructions:
        y = _intersection(s_in1[ins.i_in1], c_in1, s_in2[ins.i_in2], c_in2)

        verts = []
        codes = []
        verts += [s_out[ins.i_out], y]
        codes += [Path.MOVETO, Path.LINETO]
        verts += [s_in1[ins.i_in1], y]
        codes += [Path.MOVETO, Path.LINETO]
        verts += [s_in2[ins.i_in2], y]
        codes += [Path.MOVETO, Path.LINETO]

        ax.add_patch(patches.PathPatch(
            Path(verts, codes),
            facecolor='none',
            edgecolor='red' if ins.has_weight else 'black',
            alpha=0.5,
            ls='-',
            lw=ins.path_weight / min(i.path_weight for i in tp.instructions),
        ))

    for i, ir in enumerate(tp.irreps_in1):
        ax.annotate(ir, s_in1[i], horizontalalignment='right')

    for i, ir in enumerate(tp.irreps_in2):
        ax.annotate(ir, s_in2[i], horizontalalignment='left')

    for i, ir in enumerate(tp.irreps_out):
        ax.annotate(ir, s_out[i], horizontalalignment='center', verticalalignment='top', rotation=90)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.axis('equal')
    ax.axis('off')
