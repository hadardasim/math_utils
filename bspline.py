from scipy.interpolate import BSpline
import numpy as np

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
# https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-basis.html
# http://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node17.html


def create_bspline(k, y_values, x_values=None, clamped=True, fix_shift=True, extrapolate=True):
    """
    create bspline curve of degree k that follows y_values.
    note that in general, the curve will not pass the y_values
    :param k: spline order / degree. higher value will make the spline smoother and include more neighbors
    :param y_values: list / array of control points
    :param x_values: [optional] ascending list / array scalar in of size len(y_values)
    :param clamped: if true the spline will go through both end points
    :param fix_shift: adjust x axis to better match the control points
    :param extrapolate: see documentation of scipy.interpolate bspline. for debugging it is recommended to set this value
    to False
    :return: bspline
    """

    if not x_values:
        x_values = range(0, len(y_values))
    assert len(x_values) == len(y_values)

    dx_end = x_values[-1] - x_values[-2]
    dx_start = x_values[1] - x_values[0]

    if fix_shift:
        left_shift = 1 + int(max(k-1, 0)/2)
        x_values = list(x_values[0] + np.linspace(-left_shift, -1, left_shift) * dx_start) + x_values

        if k > 0 and k % 2 == 0:
            # add one fake key at the beginning and get the average of each 2 adjacent keys
            x_values = [x_values[0] - dx_start*0.5] + list(0.5*np.array(x_values[0:-1]) + 0.5*np.array(x_values[1:]))
    else:
        left_shift = 0

    if clamped:
        rep = max(k-1, 0)
        x_values = list(x_values[0] + np.linspace(-rep, -1, rep) * dx_start) + x_values + \
                   list(x_values[-1] + np.linspace(1, rep, rep) * dx_end)
        dy_start = y_values[1] - y_values[0]
        dy_end = y_values[-1] - y_values[-2]

        def my_outer(scale_vector, point):
            if isinstance(point, (list, np.ndarray)):
                return np.outer(scale_vector, point)
            return scale_vector*point

        control_points = list(y_values[0] + my_outer(np.linspace(-rep, -1, rep), dy_start)) + list(y_values) +\
                         list(y_values[-1] + my_outer(np.linspace(1, rep, rep), dy_end))
    else:
        control_points = y_values

    assert len(x_values) == len(control_points) + left_shift
    assert k >= 0

    # extend the tail of x_values
    knots = x_values + list(x_values[-1] + np.linspace(1, k - left_shift + 1, k - left_shift + 1) * dx_end)

    return BSpline(knots, control_points, k, extrapolate=extrapolate)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # ==================
    # 2D spline examples

    xy_in = np.asarray([[0,0], [1,0], [1,1], [0,1], [-1, 0]])
    spl_2d = create_bspline(3, xy_in)
    tt = np.linspace(0, len(xy_in)-1, 100)
    xy_out = spl_2d(tt)

    fig, ax = plt.subplots()
    ax.plot(xy_in[:, 0], xy_in[:, 1], 'o', label='control points')
    ax.plot(xy_out[:, 0], xy_out[:, 1], '-', lw=3, alpha=0.7, label='BSpline k = 3')
    ax.grid(True)
    ax.legend(loc='best')
    plt.title('2d spline')
    plt.show()

    # y = [3.0, 0.0, 1.0, 2.0, 1.0, 3.0, 2.5, 0.5, 1.0]
    # y = [4, 4, 4.0, 0.0, 3.0, 0.0, 2.0, 0.0, 3.0, 3, 3]

    # ===================
    # 1D spline examples

    y = [4.0, 0.0, 3.0, 0.0, 2.0, 0.0, 3.0]
    x = range(0, len(y))

    uniform = False
    if not uniform:
        for ind in range(3, len(x)):
            x[ind] += 2
        x[-1] += 1

    x_2_y = dict()
    for ind in range(len(x)):
        x_2_y[x[ind]] = y[ind]
    print('x->y ', x_2_y)

    clamped = True
    fix_shift = True
    spl0 = create_bspline(0, y, x, clamped=clamped, fix_shift=fix_shift, extrapolate=False)
    spl1 = create_bspline(1, y, x, clamped=clamped, fix_shift=fix_shift, extrapolate=False)
    spl2 = create_bspline(2, y, x, clamped=clamped, fix_shift=fix_shift, extrapolate=False)
    spl3 = create_bspline(3, y, x, clamped=clamped, fix_shift=fix_shift, extrapolate=False)
    spl4 = create_bspline(4, y, x, clamped=clamped, fix_shift=fix_shift, extrapolate=False)
    x_trim = 0
    xx = np.linspace(x_trim + x[0], x[-1] - x_trim, 500)

    fig, ax = plt.subplots()
    ax.plot(x, y, 'o', label='control points')
    ax.plot(xx, spl0(xx), '-', lw=3, alpha=0.7, label='BSpline k = 0')
    ax.plot(xx, spl1(xx), 'b-', lw=3, alpha=0.7, label='BSpline k = 1')
    ax.plot(xx, spl2(xx), 'g-', lw=3, alpha=0.7, label='BSpline k = 2')
    ax.plot(xx, spl3(xx), 'r-', lw=3, alpha=0.7, label='BSpline k = 3')
    ax.plot(xx, spl4(xx), '-', lw=3, alpha=0.7, label='BSpline k = 4')
    ax.grid(True)
    ax.legend(loc='best')
    plt.title('clamped={}, uniform={}, shift_fix={}'.format(clamped, uniform, fix_shift))
    plt.show()

    nu = 1
    fig, ax = plt.subplots()
    ax.plot(xx, spl0(xx, nu=nu), '-', lw=3, alpha=0.7, label='BSpline k = 0')
    ax.plot(xx, spl1(xx, nu=nu), 'b-', lw=3, alpha=0.7, label='BSpline k = 1')
    ax.plot(xx, spl2(xx, nu=nu), 'g-', lw=3, alpha=0.7, label='BSpline k = 2')
    ax.plot(xx, spl3(xx, nu=nu), 'r-', lw=3, alpha=0.7, label='BSpline k = 3')
    ax.plot(xx, spl4(xx, nu=nu), '-', lw=3, alpha=0.7, label='BSpline k = 4')
    ax.grid(True)
    ax.legend(loc='best')
    plt.title('y={}, clamped={}, nu={}'.format(y, clamped, nu))
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(xx, spl3(xx, nu=0), '-', lw=3, alpha=0.7, label='BSpline nu = 0 (pos c2)')
    ax.plot(xx, spl3(xx, nu=1), 'b-', lw=3, alpha=0.7, label='BSpline nu = 1 (vel c1)')
    ax.plot(xx, spl3(xx, nu=2), 'g-', lw=3, alpha=0.7, label='BSpline nu = 2 (acc c0)')

    ax.grid(True)
    ax.legend(loc='best')
    plt.title('y={}, clamped={}, k={}'.format(y, clamped, 3))
    plt.show()


