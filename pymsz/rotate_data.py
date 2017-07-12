import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Process, cpu_count, Queue, freeze_support  # , current_process, Array
# import ctypes


def rotate_data(pos, axis):
    r""" rotate the data points (3d) to a given direction. Returns data points (2d) at line of sight.

    Parameter:
    ----------
    pos     : input data points in 3D.
    axis    : can be 'x', 'y', 'z' (must be 2D), or a list of degrees [alpha, beta, gamma],
              which will rotate the data points by $\alpha$ around the x-axis,
              $\beta$ around the y-axis, and $\gamma$ around the z-axis

    Notes:
    --------
    This function does not work with yt data currrently.
    """

    if isinstance(axis, type('')):
        if axis.lower() == 'y':  # x-z plane
            npos = np.copy(pos)
            npos[:, 1] = pos[:, 2]
            npos[:, 2] = pos[:, 1]
            return npos
        elif axis.lower() == 'x':  # y - z plane
            npos = np.copy(pos)
            npos[:, 0] = pos[:, 1]
            npos[:, 1] = pos[:, 2]
            npos[:, 2] = pos[:, 0]
            return npos
        elif axis.lower() == 'z':
            return pos
        else:
            # if axis != 'z':  # project to xy plane
            raise ValueError("Do not accept this value %s for projection" % axis)
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
            return np.dot(pos, Rxyz)
        else:
            raise ValueError("Do not accept this value %s for projection" % axis)
    else:
        raise ValueError("Do not accept this value %s for projection" % axis)

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


def worker(input, output):
    for func, args in iter(input.get, 'STOP'):
        result = calculate(func, args)
        output.put(result)


def calculate(func, args):
    return func(*args)
    # result = func(*args)
    # return '%s says that %s%s = %s' % \
    #     (current_process().name, func.__name__, args, result)


# No boundary effects are taken into account!
def cal_sph_hsml(n, mtree, pos, hsml, pxln, indxyz, sphkernel, wdata):
    if np.max(n) >= pos.shape[0]:
        n = n[n < pos.shape[0]]

    if isinstance(wdata, type(np.array([1]))) or isinstance(wdata, type([])):
        if pos.shape[1] == 2:
            ydata = np.zeros((pxln, pxln), dtype=np.float64)
            for i in n:
                ids = mtree.query_ball_point(pos[i], hsml[i])
                if len(ids) != 0:
                    dist = np.sqrt(np.sum((pos[i] - mtree.data[ids])**2, axis=1))
                    wsph = sphkernel(dist/hsml[i])
                    ydata[indxyz[ids, 0], indxyz[ids, 1]] += wdata[i] * wsph / wsph.sum()
                # else:
                #     dist, ids = mtree.query(pos[i], k=1)
                #     ydata[indxyz[ids, 0], indxyz[ids, 1]] += wdata[i]
        elif pos.shape[1] == 3:
            ydata = np.zeros((pxln, pxln, pxln), dtype=np.float32)
            for i in n:
                ids = mtree.query_ball_point(pos[i], hsml[i])
                if len(ids) != 0:
                    dist = np.sqrt(np.sum((pos[i] - mtree.data[ids])**2, axis=1))
                    wsph = sphkernel(dist/hsml[i])
                    ydata[indxyz[ids, 0], indxyz[ids, 1], indxyz[ids, 2]] += wdata[i] * wsph / wsph.sum()
                else:
                    dist, ids = mtree.query(pos[i], k=1)
                    ydata[indxyz[ids, 0], indxyz[ids, 1], indxyz[ids, 2]] += wdata[i]
    else:
        ydata = {}
        if pos.shape[1] == 2:
            for i in wdata.keys():
                ydata[i] = np.zeros((pxln, pxln), dtype=np.float64)
            for i in n:
                ids = mtree.query_ball_point(pos[i], hsml[i])
                if len(ids) != 0:
                    dist = np.sqrt(np.sum((pos[i] - mtree.data[ids])**2, axis=1))
                    wsph = sphkernel(dist/hsml[i])
                    for j in wdata.keys():
                        ydata[j][indxyz[ids, 0], indxyz[ids, 1]] += wdata[j][i] * wsph / wsph.sum()
                else:
                    dist, ids = mtree.query(pos[i], k=1)
                    for j in wdata.keys():
                        ydata[j][indxyz[ids, 0], indxyz[ids, 1]] += wdata[j][i]
        elif pos.shape[1] == 3:
            for i in wdata.keys():
                # There is a problem using multiprocessing with (return) really big objects
                # https://bugs.python.org/issue17560
                ydata[i] = np.zeros((pxln, pxln, pxln), dtype=np.float32)
            for i in n:
                ids = mtree.query_ball_point(pos[i], hsml[i])
                if len(ids) != 0:
                    dist = np.sqrt(np.sum((pos[i] - mtree.data[ids])**2, axis=1))
                    wsph = sphkernel(dist/hsml[i])
                    for j in wdata.keys():
                        ydata[j][indxyz[ids, 0], indxyz[ids, 1], indxyz[ids, 2]] += wdata[j][i] * wsph / wsph.sum()
                else:
                    dist, ids = mtree.query(pos[i], k=1)
                    for j in wdata.keys():
                        ydata[j][indxyz[ids, 0], indxyz[ids, 1], indxyz[ids, 2]] += wdata[j][i]

    return ydata


def cal_sph_neib(n, idst, dist, pos, pxln, indxyz, sphkernel, wdata):
    if np.max(n) >= pos.shape[0]:
        n = n[n < pos.shape[0]]

    if isinstance(wdata, type(np.array([1]))) or isinstance(wdata, type([])):
        if pos.shape[1] == 2:
            ydata = np.zeros((pxln, pxln), dtype=np.float64)
            for i in np.arange(pos.shape[0]):
                ids = idst[i]
                wsph = sphkernel(dist[i]/dist[i].max())
                ydata[indxyz[ids, 0], indxyz[ids, 1]] += wdata[i] * wsph / wsph.sum()
        elif pos.shape[1] == 3:
            ydata = np.zeros((pxln, pxln, pxln), dtype=np.float32)
            for i in np.arange(pos.shape[0]):
                ids = idst[i]
                wsph = sphkernel(dist[i]/dist[i].max())
                ydata[indxyz[ids, 0], indxyz[ids, 1], indxyz[ids, 2]] += wdata[i] * wsph / wsph.sum()
    else:
        if pos.shape[1] == 2:
            for i in wdata.keys():
                ydata[i] = np.zeros((pxln, pxln), dtype=np.float64)
            for i in np.arange(pos.shape[0]):
                ids = idst[i]
                wsph = sphkernel(dist[i]/dist[i].max())
                for j in wdata.keys():
                    ydata[j][indxyz[ids, 0], indxyz[ids, 1]] += wdata[j][i] * wsph / wsph.sum()
        elif pos.shape[1] == 3:
            for i in wdata.keys():
                ydata[i] = np.zeros((pxln, pxln, pxln), dtype=np.float32)
            for i in np.arange(pos.shape[0]):
                ids = idst[i]
                wsph = sphkernel(dist[i]/dist[i].max())
                for j in wdata.keys():
                    ydata[j][indxyz[ids, 0], indxyz[ids, 1], indxyz[ids, 2]] += wdata[j][i] * wsph / wsph.sum()
    return ydata


def SPH_smoothing(wdata, pos, pxls, hsml=None, neighbors=64, pxln=None, Ncpu=None,
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
    pxln     : number of pixels for the mesh. Type: int. Must be set.
                # If it is None (Default), it will calculate from the particle positions.
                # I.E. pxln = (max(pos)-min(pos))/pxls
    Ncpu     : number of CPU for parallel calculation. Type: int. Default: None, all cpus from the
                computer will be used.
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
        raise ValueError("pos shape %d is not correct, the second dimension must be 2 or 3" % pos.shape)

    minx = -pxls * pxln / 2

    # if SD == 3:
    #     minz = minx
    # maxz = maxx  # +pxls * pxln / 2

    if SD == 2:
        pos = (pos - [minx, minx]) / pxls  # in units of pixel size
        # nx = np.int32(np.ceil((maxx - minx) / pxls))
        # ny = np.int32(np.ceil((maxy - miny) / pxls))
        x, y = np.meshgrid(np.arange(0.5, pxln, 1.0), np.arange(0.5, pxln, 1.0), indexing='ij')
        indxyz = np.concatenate((x.reshape(x.size, 1), y.reshape(y.size, 1)), axis=1)
        if isinstance(wdata, type(np.array([1]))) or isinstance(wdata, type([])):
            ydata = np.zeros((pxln, pxln), dtype=np.float64)
            # ydata_base = Array(ctypes.c_double, pxln**2)
            # ydata = np.ctypeslib.as_array(ydata_base.get_obj())
            # ydata = ydata.reshape(pxln, pxln)
        elif isinstance(wdata, type({})):
            if len(wdata) > 20:
                raise ValueError("Too many data to be smoothed %d" % len(wdata))
            else:
                ydata = {}
                for i in wdata.keys():
                    ydata[i] = np.zeros((pxln, pxln), dtype=np.float32)
    else:
        pos = (pos - [minx, minx, minx]) / pxls
        x, y, z = np.meshgrid(np.arange(0.5, pxln, 1.0), np.arange(0.5, pxln, 1.0),
                              np.arange(0.5, pxln, 1.0), indexing='ij')
        indxyz = np.concatenate((x.reshape(x.size, 1), y.reshape(y.size, 1),
                                z.reshape(z.size, 1)), axis=1)
        if isinstance(wdata, type(np.array([1]))) or isinstance(wdata, type([])):
            ydata = np.zeros((pxln, pxln, pxln), dtype=np.float64)
        else:
            if len(wdata) > 20:
                raise ValueError("Too many data to be smoothed %d" % len(wdata))
            else:
                ydata = {}
                for i in wdata.keys():
                    ydata[i] = np.zeros((pxln, pxln, pxln), dtype=np.float64)

    # Federico's method
    # if hsml is not None:
    #     hsml /= pxls
    # for i in np.arange(pos.shape[0]):
    #     x = np.arange(np.int32(pos[i, 0] - hsml[i]), np.int32(pos[i, 0] + hsml[i]), 1)
    #     y = np.arange(np.int32(pos[i, 1] - hsml[i]), np.int32(pos[i, 1] + hsml[i]), 1)
    #     x, y = np.meshgrid(x, y, indexing='ij')
    #     xyz = np.concatenate((x.reshape(x.size, 1), y.reshape(y.size, 1)), axis=1)
    #     dist = np.sqrt(np.sum((xyz - pos[i])**2, axis=1)) / hsml[i]
    #     if len(dist[dist < 1]) >= 1:
    #         wsph = sphkernel(dist)
    #         ids = (xyz[:, 0] >= 0) & (xyz[:, 0] < pxln) & (xyz[:, 1] >= 0) & (xyz[:, 1] < pxln)
    #         if wsph[ids].sum() > 0:
    #             ydata[xyz[ids, 0], xyz[ids, 1]] += wdata[i] * wsph[ids] / wsph[ids].sum()

    mtree = cKDTree(indxyz)
    indxyz = np.int32(indxyz)
    
    freeze_support()
    if Ncpu is None:
        NUMBER_OF_PROCESSES = cpu_count()
    else:
        NUMBER_OF_PROCESSES = Ncpu
    N = np.int32(np.ceil(pos.shape[0]/NUMBER_OF_PROCESSES))
    # Create queues
    task_queue = Queue()
    done_queue = Queue()

    if hsml is None:  # nearest neighbors
        dist, idst = mtree.query(pos, neighbors)

        Tasks = [(cal_sph_neib, (range(i*N, (i+1)*N), idst, dist, pos, pxln, indxyz,
                 sphkernel, wdata)) for i in range(NUMBER_OF_PROCESSES)]
    else:  # use hsml
        hsml /= pxls

        Tasks = [(cal_sph_hsml, (range(i*N, (i+1)*N), mtree, pos, hsml, pxln, indxyz,
                 sphkernel, wdata)) for i in range(NUMBER_OF_PROCESSES)]

    # Submit tasks
    for task in Tasks:
        task_queue.put(task)

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=worker, args=(task_queue, done_queue)).start()

    # Get results
    for i in range(len(Tasks)):
        if isinstance(wdata, type(np.array([1]))) or isinstance(wdata, type([])):
            ydata += done_queue.get()
        else:
            for j in wdata.keys():
                ydata[j] = done_queue.get()[j]

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put('STOP')

        #         # for i in range(0, 6):
        #         #     p = Process(target=cal_sph_2d, args=(range(i*N, (i+1)*N), mtree, pos, hsml,
        #         #                                          pxln, indxyz, sphkernel, wdata, ydata))
        #         #     p.start()
        #         # p.join()
        #
        #         # for i in np.arange(pos.shape[0]):
        #         #     ids = mtree.query_ball_point(pos[i], hsml[i])
        #         #     if len(ids) != 0:
        #         #         dist = np.sqrt(np.sum((pos[i] - mtree.data[ids])**2, axis=1))
        #         #         wsph = sphkernel(dist/hsml[i])
        #         #         ydata[indxyz[ids, 0], indxyz[ids, 1]] += wdata[i] * wsph / wsph.sum()
        #         #     else:
        #         #         dist, ids = mtree.query(pos[i], k=1)
        #         #         ydata[indxyz[ids, 0], indxyz[ids, 1]] += wdata[i]
        #     elif SD == 3:
        #         for i in np.arange(pos.shape[0]):
        #             ids = mtree.query_ball_point(pos[i], hsml[i])
        #             if len(ids) != 0:
        #                 dist = np.sqrt(np.sum((pos[i] - mtree.data[ids])**2, axis=1))
        #                 wsph = sphkernel(dist/hsml[i])
        #                 ydata[indxyz[ids, 0], indxyz[ids, 1], indxyz[ids, 2]] += wdata[i] * wsph / wsph.sum()
        #             else:
        #                 dist, ids = mtree.query(pos[i], k=1)
        #                 ydata[indxyz[ids, 0], indxyz[ids, 1], indxyz[ids, 2]] += wdata[i]
        #     else:
        #         raise ValueError("Don't accept this data dimension %d" % SD)
        # else:
        #     if SD == 2:
        #         for i in np.arange(pos.shape[0]):
        #             ids = mtree.query_ball_point(pos[i], hsml[i])
        #             if len(ids) != 0:
        #                 dist = np.sqrt(np.sum((pos[i] - mtree.data[ids])**2, axis=1))
        #                 wsph = sphkernel(dist/hsml[i])
        #                 for j in range(len(wdata)):
        #                     ydata[str(j)][indxyz[ids, 0], indxyz[ids, 1]] += wdata[i] * wsph / wsph.sum()
        #             else:
        #                 dist, ids = mtree.query(pos[i], k=1)
        #                 for j in range(len(wdata)):
        #                     ydata[str(j)][indxyz[ids, 0], indxyz[ids, 1]] += wdata[i]
        #     elif SD == 3:
        #         for i in np.arange(pos.shape[0]):
        #             ids = mtree.query_ball_point(pos[i], hsml[i])
        #             if len(ids) != 0:
        #                 dist = np.sqrt(np.sum((pos[i] - mtree.data[ids])**2, axis=1))
        #                 wsph = sphkernel(dist/hsml[i])
        #                 for j in range(len(wdata)):
        #                     ydata[str(j)][indxyz[ids, 0], indxyz[ids, 1], indxyz[ids, 2]] += wdata[i]*wsph/wsph.sum()
        #             else:
        #                 dist, ids = mtree.query(pos[i], k=1)
        #                 for j in range(len(wdata)):
        #                     ydata[str(j)][indxyz[ids, 0], indxyz[ids, 1], indxyz[ids, 2]] += wdata[i]
        #     else:
        #         raise ValueError("Don't accept this data dimension %d" % SD)

    return ydata
