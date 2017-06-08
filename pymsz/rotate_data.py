import numpy as np
from scipy.spatial import cKDTree


def rotate_data(pos, axis):
    r""" rotate the data points (3d) to a given direction. Returns data points (2d) at line of sight.

    Parameter:
    ----------
    pos     : input data points in 3D.
    axis    : can be 'x', 'y', 'z', or a list of degrees [alpha, beta, gamma],
              which will rotate the data points by $\alpha$ around the x-axis,
              $\beta$ around the y-axis, and $\gamma$ around the z-axis

    Notes:
    --------
    This function does not work with yt data currrently.
    """

    if isinstance(axis, type('')):
        if axis == 'y':  # x-z plane
            pos[:, 1] = pos[:, 2]
        elif axis == 'x':  # y - z plane
            pos[:, 0] = pos[:, 1]
            pos[:, 1] = pos[:, 2]
        else:
            if axis != 'z':  # project to xy plane
                raise ValueError(
                    "Do not accept this value %s for projection" % axis)
    elif isinstance(axis, type([])):
        if len(axis) == 3:
            sa, ca = np.sin(axis[0] / 180. *
                            np.pi), np.cos(axis[0] / 180. * np.pi)
            sb, cb = np.sin(axis[1] / 180. *
                            np.pi), np.cos(axis[1] / 180. * np.pi)
            sg, cg = np.sin(axis[2] / 180. *
                            np.pi), np.cos(axis[2] / 180. * np.pi)
            # ratation matrix from
            # http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/
            Rxyz = np.array(
                [[cb * cg, cg * sa * sb - ca * sg, ca * cg * sb + sa * sg],
                 [cb * sg, ca * cg + sa * sb * sg, ca * sb * sg - cg * sa],
                 [-sb,     cb * sa,                ca * cb]], dtype=np.float64)
            pos = np.dot(pos, Rxyz)
        else:
            raise ValueError(
                "Do not accept this value %s for projection" % axis)
    return pos

    # Now grid the data
    # pmax, pmin = np.max(pos, axis=0), np.min(pos, axis=0)
    # grid_x, grid_y = np.mgrid[pmin[0]:pmax[0]:nx, pmin[1]:pmax[1]:nx]
    # self.grid_mass = np.histogram2d(pos[:, 0], pos[:, 1], bins=[
    #                                 nx, nx], weights=self.mass)[0]
    # ids = self.grid_mass > 0
    # self.grid_age = np.histogram2d(pos[:, 0], pos[:, 1], bins=[
    #                                nx, nx], weights=self.temp * self.mass)[0]
    # self.grid_age[ids] /= self.grid_mass[ids]  # mass weighted age
    # self.grid_metal = np.histogram2d(pos[:, 0], pos[:, 1], bins=[
    #                                  nx, nx], weights=self.metal * self.mass)[0]
    # self.grid_metal[ids] /= self.grid_mass[ids]  # mass weighted metal
    # dx = (pmax[0] - pmin[0] + pmax[0] * 0.001) / nx
    # dy = (pmax[1] - pmin[1] + pmax[1] * 0.001) / nx
    # self.grids = np.int32(
    #     np.floor((pos[:, :2] - pmin[:2]) / np.array([dx, dy])))


# For SPH kernels, we always use the total weights = 1,
# thus the constant C can be ignored
def sph_kernel_cubic(x):
    # C = 2.5464790894703255
    # if x <= 0.5:
    #     kernel = 1.-6.*x*x*(1.-x)
    # elif x > 0.5 and x <= 1.0:
    #     kernel = 2.*(1.-x)*(1.-x)*(1.-x)
    # else:
    #     kernel = 0.
    # return kernel * C
    kernel = np.zeros(x.size, dtype=float)
    ids = x <= 0.5
    kernel[ids] = 1.-6.*x[ids]*x[ids]*(1.-x[ids])
    ids = (x > 0.5) & (x < 1.0)
    kernel[ids] = 2.*(1.-x[ids])*(1.-x[ids])*(1.-x[ids])
    return kernel


# quartic spline
def sph_kernel_quartic(x):
    # C = 5.**6/512/np.pi
    # if x < 1:
    #     kernel = (1.-x)**4
    #     if x < 3./5:
    #         kernel -= 5*(3./5-x)**4
    #         if x < 1./5:
    #             kernel += 10*(1./5-x)**4
    # else:
    #     kernel = 0.
    # return kernel * C
    kernel = np.zeros(x.size, dtype=float)
    ids = x < 0.2
    kernel[ids] = (1.-x[ids])**4 + 10*(0.2-x[ids])**4
    ids = (x >= 0.2) & (x < 0.6)
    kernel[ids] = (1.-x[ids])**4
    ids = (x >= 0.6) & (x < 1)
    kernel[ids] = (1.-x[ids])**4 - 5*(0.6-x[ids])**4
    return kernel


# quintic spline
def sph_kernel_quintic(x):
    # C = 3.**7/40/np.pi
    # if x < 1:
    #     kernel = (1.-x)**5
    #     if x < 2./3:
    #         kernel -= 6*(2./3-x)**5
    #         if x < 1./3:
    #             kernel += 15*(1./3-x)**5
    # else:
    #     kernel = 0.
    # return kernel * C
    kernel = np.zeros(x.size, dtype=float)
    ids = x < 0.3333333333333
    kernel[ids] = (1.-x[ids])**5 + 15*(0.3333333333333-x[ids])**5
    ids = (x >= 0.3333333333333) & (x < 0.6666666666666)
    kernel[ids] = (1.-x[ids])**5
    ids = (x >= 0.6666666666666) & (x < 1)
    kernel[ids] = (1.-x[ids])**5 - 6*(2./3-x[ids])**5
    return kernel


# Wendland C2
def sph_kernel_wendland2(x):
    # C = 21./2/np.pi
    # if x < 1:
    #     kernel = (1.-x)**4 * (1+4*x)
    # else:
    #     kernel = 0.
    # return kernel * C
    kernel = np.zeros(x.size, dtype=float)
    ids = x < 1
    kernel[ids] = (1.-x[ids])**4 * (1+4*x[ids])
    return kernel


# Wendland C4
def sph_kernel_wendland4(x):
    # C = 495./32/np.pi
    # if x < 1:
    #     kernel = (1.-x)**6 * (1+6*x+35./3*x**2)
    # else:
    #     kernel = 0.
    # return kernel * C
    kernel = np.zeros(x.size, dtype=float)
    ids = x < 1
    kernel[ids] = (1.-x[ids])**6 * (1+6*x[ids]+35./3*x[ids]**2)
    return kernel


# Wendland C6
def sph_kernel_wendland6(x):
    # C = 1365./64/np.pi
    # if x < 1:
    #     kernel = (1.-x)**8 * (1+8*x+25*x**2+32*x**3)
    # else:
    #     kernel = 0.
    # return kernel * C
    kernel = np.zeros(x.size, dtype=float)
    ids = x < 1
    kernel[ids] = (1.-x[ids])**8 * (1+8*x[ids]+25*x[ids]**2+32*x[ids]**3)
    return kernel


def SPH_smoothing(wdata, pos, pxls, hsml=None, neighbors=64, pxln=None,
                  kernel_name='cubic'):
    r"""SPH smoothing for given data

    Parameters
    ----------
    wdata    : the data to be smoothed. Type: array or list of arrays
    pos      : the position of your SPH particles. Type: array.
                It can be 3D or 2D
    pxls     : grid pixel size of the mesh, where the data is smoothed into.
                Type: float.
                It must be in the same units of positions
    hsml     : smoothing length of these SPH particles. Type: array or float.
                If it is None, then the neighbours will be used to do the smoothing.
    neighbors: how many nearby mesh points the SPH particles smoothed into.
                Type: int. Default: 64
    pxln     : number of pixels for the mesh. Type: int.
                If it is None (Default), it will calculate from the particle positions.
                I.E. pxln = (max(pos)-min(pos))/pxls
    kernel_name : the SPH kernel used to make the smoothing. Type: str.
                Default : 'cubic'. Since a normalization will applied in the
                smoothing, the constants in the kernel are always ignored.

    Returns
    -------
    (list of) Smoothed 2D or 3D mesh data.

    See also
    --------


    Notes
    -----


    Example
    -------
    ydata = SPH_smoothing(wdata, pos, 10, hsml=HSML)
    """

    if kernel_name.lower() == 'cubic':
        sphkernel = sph_kernel_cubic
    elif kernel_name.lower() == 'quartic':
        sphkernel = sph_kernel_quartic
    elif kernel_name.lower() == 'quintic':
        sphkernel = sph_kernel_quintic
    elif kernel_name.lower() == 'wendland2':
        sphkernel = sph_kernel_wendland2
    elif kernel_name.lower() == 'wendland4':
        sphkernel = sph_kernel_wendland4
    elif kernel_name.lower() == 'wendland6':
        sphkernel = sph_kernel_wendland6
    else:
        raise ValueError("Do not accept this kernel name %s" % kernel_name)

    SD = pos.shape[1]
    if SD not in [2, 3]:
        raise ValueError("pos shape %d not correct, the second dimension must be 2 or 3" % pos.shape)

    if pxln is None:  # use min max
        minx = pos[:, 0].min()
        maxx = pos[:, 0].max()
        miny = pos[:, 1].min()
        maxy = pos[:, 1].max()
        if SD == 3:
            minz = pos[:, 2].min()
            maxz = pos[:, 2].max()
    else:  # use pos center
        medpos = np.median(pos, axis=0)
        minx = medpos[0] - pxls * pxln / 2
        maxx = medpos[0] + pxls * pxln / 2
        miny = medpos[1] - pxls * pxln / 2
        maxy = medpos[1] + pxls * pxln / 2
        if (minx > pos[:, 0].min()) or (maxx < pos[:, 0].max()) or \
           (miny > pos[:, 1].min()) or (maxy < pos[:, 1].max()):
            print("Warning, mesh is small for SPH data!")
        if SD == 3:
            minz = medpos[2] - pxls * pxln / 2
            maxz = medpos[2] + pxls * pxln / 2

    if SD == 2:
        pos = (pos - [minx, miny]) / pxls  # in units of pixel size
        nx = np.int32(np.ceil((maxx - minx) / pxls)) + 1
        ny = np.int32(np.ceil((maxy - miny) / pxls)) + 1
        x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
        indxyz = np.concatenate((x.reshape(x.size, 1), y.reshape(y.size, 1)), axis=1)
        if isinstance(wdata, type(np.array([1]))):
            ydata = np.zeros((nx, ny), dtype=np.float32)
        else:
            if len(wdata) > 20:
                raise ValueError("Too many data to be smoothed %d" % len(wdata))
            else:
                ydata = {}
                for i in range(len(wdata)):
                    ydata[str(i)] = np.zeros((nx, ny), dtype=np.float32)
    else:
        pos = (pos - [minx, miny, minz]) / pxls
        nx = np.int32(np.ceil((maxx - minx) / pxls)) + 1
        ny = np.int32(np.ceil((maxy - miny) / pxls)) + 1
        nz = np.int32(np.ceil((maxz - minz) / pxls)) + 1
        x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
        indxyz = np.concatenate((x.reshape(x.size, 1), y.reshape(y.size, 1),
                                z.reshape(z.size, 1)), axis=1)
        if isinstance(wdata, type(np.array([1]))):
            ydata = np.zeros((nx, ny, nz), dtype=np.float32)
        else:
            if len(wdata) > 20:
                raise ValueError("Too many data to be smoothed %d" % len(wdata))
            else:
                ydata = {}
                for i in range(len(wdata)):
                    ydata[str(i)] = np.zeros((nx, ny, nz), dtype=np.float32)

    # Federico's method
    if hsml is not None:
        hsml /= pxls
    for i in np.arange(pos.shape[0]):
        x = np.arange(np.int32(pos[i, 0] - hsml[i]), np.int32(pos[i, 0] + hsml[i]), 1)
        y = np.arange(np.int32(pos[i, 1] - hsml[i]), np.int32(pos[i, 1] + hsml[i]), 1)
        x, y = np.meshgrid(x, y, indexing='ij')
        xyz = np.concatenate((x.reshape(x.size, 1), y.reshape(y.size, 1)), axis=1)
        dist = np.sqrt(np.sum((xyz - pos[i])**2, axis=1)) / hsml[i]
        if len(dist[dist < 1]) >= 1:
            wsph = sphkernel(dist)
            ids = (xyz[:, 0] >= 0) & (xyz[:, 0] < nx) & (xyz[:, 1] >= 0) & (xyz[:, 1] < ny)
            if wsph[ids].sum() > 0:
                ydata[xyz[ids, 0], xyz[ids, 1]] += wdata[i] * wsph[ids] / wsph[ids].sum()

    # mtree = cKDTree(indxyz)
    # if hsml is None:
    #     dist, idst = mtree.query(pos, neighbors)
    # else:
    #     hsml /= pxls
    # for i in np.arange(pos.shape[0]):
    #     if hsml is not None:
    #         ids = mtree.query_ball_point(pos[i], hsml[i])
    #         if len(ids) != 0:
    #             dist = np.sqrt(np.sum((pos[i] - mtree.data[ids])**2, axis=1))
    #             wsph = sphkernel(dist/hsml[i])
    #         else:
    #             continue
    #     else:
    #         ids = idst[i]
    #         wsph = sphkernel(dist[i]/dist[i].max())
    #
    #     if isinstance(wdata, type(np.array([1]))):
    #         if SD == 2:
    #             ydata[indxyz[ids, 0], indxyz[ids, 1]] += wdata[i] * wsph / wsph.sum()
    #         else:
    #             ydata[indxyz[ids, 0], indxyz[ids, 1], indxyz[ids, 2]] += wdata[i] * wsph / wsph.sum()
    #     else:
    #         for j in range(len(wdata)):
    #             if SD == 2:
    #                 ydata[str(j)][indxyz[ids, 0], indxyz[ids, 1]] += wdata[j][i] * wsph / wsph.sum()
    #             else:
    #                 ydata[str(j)][indxyz[ids, 0], indxyz[ids, 1], indxyz[ids, 2]] += wdata[j][i] * wsph / wsph.sum()
    return ydata
