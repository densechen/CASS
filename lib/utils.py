import numpy as np
import math


def make_box():
    """
    function to make grids on a 3D unit box
    @param lower: lower bound
    @param upper: upper bound
    @param num: number of points on an axis. Default 18
    rvalue: 2D numpy array of dim0 = num**2*6, num1 = 3. Meaning a point cloud
    """
    lower = -0.5
    upper = 0.5
    num = 18
    a = np.linspace(lower, upper, num)
    b = np.linspace(lower, upper, num)
    grid = np.transpose([np.tile(a, len(b)), np.repeat(b, len(a))])

    c1 = np.repeat(0.5, len(grid))
    c1 = np.reshape(c1, (len(c1), -1))
    c2 = np.repeat(-0.5, len(grid))
    c2 = np.reshape(c2, (len(c2), -1))

    up = np.hstack((grid, c1))  # upper face, z == 0.5
    low = np.hstack((grid, c2))  # lower face, z == -0.5
    front = up[:, [0, 2, 1]]  # front face, y == 0.5
    back = low[:, [0, 2, 1]]  # back face, y == -0.5
    right = up[:, [2, 0, 1]]  # right face, x == 0.5
    left = low[:, [2, 0, 1]]  # left face, x == -0.5

    six_faces = np.vstack((front, back, right, left, up, low))
    return six_faces


def make_cylinder():
    """
    function to make a grid from a cyliner centered at (0, 0, 0). The cyliner's radius is 1, height is 0.5
    Method:
    1) the surrounding surface is 4 times the area of the upper and lower cicle. So we sample 4 times more points from it
    2) to match with the box, total number of points is 1944
    3) for the upper and lower surface, points are sampled with fixed degree and fixed distance along the radius
    4) for the middle surface, points are sampled along fixed lines along the height
    """
    # make the upper and lower face, which is not inclusive of the boundary points
    theta = 10  # dimension
    n = 9  # number of points for every radius
    r = 0.5
    radius_all = np.linspace(0, 0.5, n + 2)[1:10]  # radius of sub-circles
    res = []
    for i, theta in enumerate(range(0, 360, 10)):
        x = math.sin(theta)
        y = math.cos(theta)
        for r in radius_all:
            res.append([r * x, r * y])
    # add z axis
    z = np.reshape(np.repeat(0.5, len(res)), (len(res), -1))
    upper = np.hstack((np.array(res), z))  # upper face
    z = np.reshape(np.repeat(-0.5, len(res)), (len(res), -1))
    lower = np.hstack((np.array(res), z))  # lower face

    # design of middle layer: theta = 5 degree, with every divide is 18 points including boundaries
    height = np.linspace(-0.5, 0.5, 18)
    res = []
    for theta in range(0, 360, 5):
        x = 0.5 * math.sin(theta)
        y = 0.5 * math.cos(theta)
        for z in height:
            res.append([x, y, z])
    middle = np.array(res)

    cylinder = np.vstack((upper, lower, middle))
    return cylinder


def make_sphere():
    """
    function to sample a grid from a sphere
    """
    theta = np.linspace(0, 360, 36)  # determining x and y
    phi = np.linspace(0, 360, 54)  # determining z

    res = []
    for p in phi:
        z = math.sin(p) * 0.5
        r0 = math.cos(p) * 0.5
        for t in theta:
            x = math.sin(t) * r0
            y = math.cos(t) * r0
            res.append([x, y, z])

    sphere = np.array(res)
    return sphere
