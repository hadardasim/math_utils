from scipy.interpolate import BSpline
import numpy as np

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html
# https://pages.mtu.edu/~shene/COURSES/cs3621/NOTES/spline/B-spline/bspline-basis.html
# http://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node17.html


def create_bspline(k, y_values, x_values=None, clamped=False):
    """
    create bspline curve of degree k that follows y_values.
    note that in general, the curve will not pass the y_values
    :param k: spline order / degree. higher value will make the spline smoother and include more neighbors
    :param y_values: list / array of control points
    :param x_values: [optional] ascending list / array scalar in of size len(y_values)
    :param clamped:
    :return: bspline
    """

    if not x_values:
        x_values = range(0, len(y_values))
    assert len(x_values) == len(y_values)

    dx_end = x_values[-1] - x_values[-2]
    dx_start = x_values[1] - x_values[0]

    if clamped:
        rep = max(k-1, 0)
        x_values = list(x_values[0] + np.linspace(-rep, -1, rep) * dx_start) + x_values + \
                   list(x_values[-1] + np.linspace(1, rep, rep) * dx_end)
        dy_start = y_values[1] - y_values[0]
        dy_end = y_values[-1] - y_values[-2]

        control_points = list(y_values[0] + np.linspace(-rep, -1, rep) * dy_start/dx_start) + y_values + \
                         list(y_values[-1] + np.linspace(1, rep, rep) * dy_end/dx_end)
        left_shift = 1 + max(k-1, 0)*0.5
    else:
        control_points = y_values
        left_shift = 1

    assert len(x_values) == len(control_points)
    x_values = list(np.array(x_values) - (x_values[1] - x_values[0])*left_shift)

    assert len(x_values) == len(control_points)
    assert k >= 0

    knots = x_values + list(x_values[-1] + np.linspace(1, k + 1, k + 1) * dx_end)

    print 'knots=', knots
    print 'control=', control_points

    # knots = np.array(knots) - (k - 1) * 0.5
    return BSpline(knots, control_points, k, extrapolate=False)




if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # y = [3.0, 0.0, 1.0, 2.0, 1.0, 3.0, 2.5, 0.5, 1.0]
    # y = [4, 4, 4.0, 0.0, 3.0, 0.0, 2.0, 0.0, 3.0, 3, 3]
    y = [4.0, 0.0, 3.0, 0.0, 2.0, 0.0, 3.0]
    clamped = True
    spl0 = create_bspline(0, y, clamped=clamped)
    spl1 = create_bspline(1, y, clamped=clamped)
    spl2 = create_bspline(2, y, clamped=clamped)
    spl3 = create_bspline(3, y, clamped=clamped)
    spl4 = create_bspline(4, y, clamped=clamped)
    x_trim = 0
    xx = np.linspace(x_trim, len(y) - 1 - x_trim, 500)

    fig, ax = plt.subplots()
    ax.plot(xx, spl0(xx), '-', lw=3, alpha=0.7, label='BSpline k = 0')
    ax.plot(xx, spl1(xx), 'b-', lw=3, alpha=0.7, label='BSpline k = 1')
    ax.plot(xx, spl2(xx), 'g-', lw=3, alpha=0.7, label='BSpline k = 2')
    ax.plot(xx, spl3(xx), 'r-', lw=3, alpha=0.7, label='BSpline k = 3')
    ax.plot(xx, spl4(xx), '-', lw=3, alpha=0.7, label='BSpline k = 4')
    ax.grid(True)
    ax.legend(loc='best')
    plt.title('y={}, clamped={}'.format(y, clamped))
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
    ax.plot(xx, spl3(xx, nu=0), '-', lw=3, alpha=0.7, label='BSpline nu = 0 (pos)')
    ax.plot(xx, spl3(xx, nu=1), 'b-', lw=3, alpha=0.7, label='BSpline nu = 1 (vel)')
    ax.plot(xx, spl3(xx, nu=2), 'g-', lw=3, alpha=0.7, label='BSpline nu = 2 (acc)')

    ax.grid(True)
    ax.legend(loc='best')
    plt.title('y={}, clamped={}, k={}'.format(y, clamped, 3))
    plt.show()


