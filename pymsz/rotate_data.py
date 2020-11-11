import numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Process, cpu_count, Queue, freeze_support, Array # , current_process, Array
import ctypes
import os, psutil, sys

def memlog(msg):
    process = psutil.Process(os.getpid())
    print('%s, RAM=%.4g GB'%(msg,process.memory_info()[0]/2.**30))

def rotate_data(pos, axis, vel=None, bvel=None):
    r""" rotate the data points (3d) to a given direction. Returns data points (2d) at line of sight.

    Parameter:
    ----------
    pos     : input data points in 3D.
    axis    : can be 'x', 'y', 'z' (must be 2D), or a list or numpy array of degrees
              [alpha, beta, gamma], which will rotate the data points by $\alpha$ around
              the x-axis, $\beta$ around the y-axis, and $\gamma$ around the z-axis.
              or a numpy array with the rotation matrix directly, must be 3x3 matrix.
    vel     : 3D velocity of the input data points. Default: None, will return an empty list.
                Otherwise, rotate_data will also return the velocity in the axis direction.
    bvel    : bulk velocity of the cluster in 3D, defualt None, resturn 0 for the bulk velocity
              along line of sight. If it is not None, bulk velocity along line of sight will be return.
    Notes:
    --------
    When you have vel is not None, the function will return two arrays: pos, vel in axis direction.
    This function does not work with yt data currrently.
    """

    nvel = []; nbvel = 0;
    if isinstance(axis, type('')):
        npos = np.copy(pos)
        if axis.lower() == 'y':  # x-z plane
            npos[:, 1] = pos[:, 2]
            npos[:, 2] = pos[:, 1]
            if vel is not None:
                nvel = vel[:, 1]
            if bvel is not None:
                nbvel = bvel[1]
        elif axis.lower() == 'x':  # y - z plane
            npos[:, 0] = pos[:, 1]
            npos[:, 1] = pos[:, 2]
            npos[:, 2] = pos[:, 0]
            if vel is not None:
                nvel = vel[:, 0]
            if bvel is not None:
                nbvel = bvel[0]
        elif axis.lower() == 'z':
            if vel is not None:
                nvel = vel[:, 2]
            if bvel is not None:
                nbvel = bvel[2]
        else:
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
            npos = np.dot(pos, Rxyz)
            if vel is not None:
                nvel = np.dot(vel, Rxyz)[:, 2]
            if bvel is not None:
                nbvel = np.dot(bvel, Rxyz)[2]
        else:
            raise ValueError("Do not accept this value %s for projection" % axis)
    elif isinstance(axis, type(np.array([]))):
        if len(axis.shape) == 1:
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
            npos = np.dot(pos, Rxyz)
            if vel is not None:
                nvel = np.dot(vel, Rxyz)[:, 2]
            if bvel is not None:
                nbvel = np.dot(bvel, Rxyz)[2]
        elif len(axis.shape) == 2:
            if axis.shape[0] == axis.shape[1] == 3:
                npos = np.dot(pos, axis)
                if vel is not None:
                    nvel = np.dot(vel, axis)[:, 2]
                if bvel is not None:
                    nbvel = np.dot(bvel, axis)[2]
            else:
                raise ValueError("Axis shape is not 3x3: ", axis.shape)
        else:
            raise ValueError("Do not accept this shape of axis %s for projection!" % axis)
    else:
        raise ValueError("Do not accept this value %s for projection!" % axis)
    return npos, nvel, nbvel


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
    kernel[ids] = 1. - 6. * x[ids] * x[ids] * (1. - x[ids])
    ids = (x > 0.5) & (x < 1.0)
    kernel[ids] = 2. * (1. - x[ids]) * (1. - x[ids]) * (1. - x[ids])
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
    kernel[ids] = (1. - x[ids])**4 - 5 * (0.6 - x[ids])**4 + 10 * (0.2 - x[ids])**4
    ids = (x >= 0.2) & (x < 0.6)
    kernel[ids] = (1. - x[ids])**4 - 5 * (0.6 - x[ids])**4
    ids = (x >= 0.6) & (x < 1)
    kernel[ids] = (1. - x[ids])**4
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
    kernel[ids] = (1. - x[ids])**5 - 6 * (2. / 3 - x[ids])**5 + 15 * (0.3333333333333 - x[ids])**5
    ids = (x >= 0.3333333333333) & (x < 0.6666666666666)
    kernel[ids] = (1. - x[ids])**5 - 6 * (2. / 3 - x[ids])**5
    ids = (x >= 0.6666666666666) & (x < 1)
    kernel[ids] = (1. - x[ids])**5
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
    kernel[ids] = (1. - x[ids])**4 * (1 + 4 * x[ids])
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
    kernel[ids] = (1. - x[ids])**6 * (1 + 6 * x[ids] + 35. / 3 * x[ids]**2)
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
    kernel[ids] = (1. - x[ids])**8 * (1 + 8 * x[ids] + 25 * x[ids]**2 + 32 * x[ids]**3)
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
def cal_sph_hsml(idst, centp, hsml, posd, pxln, sphkernel, wdata):
    imin, jmin, kmin, imax, jmax, kmax = pxln,pxln,pxln,0,0,0
    if isinstance(wdata, type(np.array([1]))) or isinstance(wdata, type([])):
        if posd == 2:
            ydata = np.zeros((pxln, pxln), dtype=np.float64)
            for i in range(hsml.size):
                if len(idst[i]) <= 0:
                    px, py = np.array([np.int32(np.round(centp[i,0]))]), np.array([np.int32(np.round(centp[i,1]))])
                    if px[0]>=pxln: px[0]=pxln-1
                    if py[0]>=pxln: py[0]=pxln-1
                else:
                    px, py = np.int32(np.array(idst[i])/pxln), np.array(idst[i])%pxln
                dist = np.sqrt((px - centp[i,0])**2 + (py - centp[i, 1])**2)
                wsph = sphkernel(dist / hsml[i])
                if imin > px.min(): imin = px.min()
                if imax < px.max(): imax = px.max()
                if jmin > py.min(): jmin = py.min()
                if jmax < py.max(): jmax = py.max()
                # devided by len(ids) is to change N_e to number density
                if wsph.sum() > 0:
                    ydata[px, py] += wdata[i] * wsph / wsph.sum()
                else:
                    # if len(dist[i][idin]) == 0:
                    ydata[px, py] += wdata[i]
                # else:  # we also add particles with hsml < pixel size to its nearest four pixels.
                #     #    Then, the y-map looks no smoothed (with some noisy pixels).
                #     dist, ids = mtree.query(pos[i], k=4)
                #     ydata[indxyz[ids, 0], indxyz[ids, 1]] += wdata[i] * \
                #         (1 - dist / np.sum(dist)) / 3.
            imax+=1; jmax+=1
            ydata=ydata[imin:imax, jmin:jmax]
        elif posd == 3:
            ydata = np.zeros((pxln, pxln, pxln), dtype=np.float32)
            for i in range(hsml.size):
                if len(idst[i]) <= 0:
                    px, py, pz = np.array(np.int32(np.round(centp[i,0]))), np.array(np.int32(np.round(centp[i,1]))), np.array(np.int32(np.round(centp[i,2])))
                    if px[0]>=pxln: px[0]=pxln-1
                    if py[0]>=pxln: py[0]=pxln-1
                    if pz[0]>=pxln: pz[0]=pxln-1
                else:
                    px, py, pz = np.int32(np.array(idst[i])/pxln/pxln), np.int32(np.array(idst[i])%(pxln*pxln)/pxln), np.array(idst[i])%(pxln*pxln)%pxln
                dist = np.sqrt((px - centp[i,0])**2 + (py - centp[i, 1])**2 + (pz - centp[i, 2])**2)
                wsph = sphkernel(dist / hsml[i])                
                if imin > px.min(): imin = px.min()
                if imax < px.max(): imax = px.max()
                if jmin > py.min(): jmin = py.min()
                if jmax < py.max(): jmax = py.max()
                if kmin > pz.min(): kmin = pz.min()
                if kmax < pz.max(): kmax = pz.max()
                if wsph.sum() > 0:
                    ydata[px, py, pz] += wdata[i] * wsph / wsph.sum()
                else:
                    # if len(dist[i][idin]) == 0:
                    ydata[px, py, pz] += wdata[i] 
                #     dist, ids = mtree.query(pos[i], k=8)
                #     ydata[indxyz[ids, 0], indxyz[ids, 1], indxyz[ids, 2]
                #           ] += wdata[i] * (1 - dist / np.sum(dist)) / 7.
            imax+=1; jmax+=1; kmax+=1
            ydata=ydata[imin:imax, jmin:jmax, kmin:kmax]
    else:
        ydata = {}
        if posd == 2:
            for i in wdata.keys():
                ydata[i] = np.zeros((pxln, pxln), dtype=np.float64)
            for i in range(hsml.size):
                if len(idst[i]) <= 0:
                    px, py = np.array(np.int32(np.round(centp[i,0]))), np.array(np.int32(np.round(centp[i,1])))
                    if px[0]>=pxln: px[0]=pxln-1
                    if py[0]>=pxln: py[0]=pxln-1
                else:
                    px, py = np.int32(np.array(idst[i])/pxln), np.array(idst[i])%pxln
                dist = np.sqrt((px - centp[i,0])**2 + (py - centp[i, 1])**2)
                wsph = sphkernel(dist / hsml[i])
                if imin > px.min(): imin = px.min()
                if imax < px.max(): imax = px.max()
                if jmin > py.min(): jmin = py.min()
                if jmax < py.max(): jmax = py.max()
                for j in wdata.keys():
                    if wsph.sum() > 0:
                        ydata[j][px, py] += wdata[j][i] * wsph / wsph.sum()
                    else:
                        # if len(dist[i][idin]) == 0:
                        ydata[j][px, py] += wdata[j][i]
                #     dist, ids = mtree.query(pos[i], k=4)
                #     for j in wdata.keys():
                #         ydata[j][indxyz[ids, 0], indxyz[ids, 1]] += wdata[j][i] * \
                #             (1 - dist / np.sum(dist)) / 3.
            imax+=1; jmax+=1
            for i in wdata.keys():
                ydata[i] = ydata[i][imin:imax, jmin:jmax]
        elif posd == 3:
            for i in wdata.keys():
                # There is a problem using multiprocessing with (return) really big objects
                # https://bugs.python.org/issue17560
                ydata[i] = np.zeros((pxln, pxln, pxln), dtype=np.float32)
            for i in range(hsml.size):
                if len(idst[i]) <= 0:
                    px, py, pz = np.array(np.int32(np.round(centp[i,0]))), np.array(np.int32(np.round(centp[i,1]))), np.array(np.int32(np.round(centp[i,2])))
                    #check boundary
                    if px[0]>=pxln: px[0]=pxln-1
                    if py[0]>=pxln: py[0]=pxln-1
                    if pz[0]>=pxln: pz[0]=pxln-1
                else:
                    px, py, pz = np.int32(np.array(idst[i])/pxln/pxln), np.int32(np.array(idst[i])%(pxln*pxln)/pxln), np.array(idst[i])%(pxln*pxln)%pxln
                dist = np.sqrt((px - centp[i,0])**2 + (py - centp[i, 1])**2 + (pz - centp[i, 2])**2)
                wsph = sphkernel(dist / hsml[i])
                if imin > px.min(): imin = px.min()
                if imax < px.max(): imax = px.max()
                if jmin > py.min(): jmin = py.min()
                if jmax < py.max(): jmax = py.max()
                if kmin > pz.min(): kmin = pz.min()
                if kmax < pz.max(): kmax = pz.max()
                for j in wdata.keys():
                    if wsph.sum() > 0:
                        ydata[j][px, py, pz] += wdata[j][i] * wsph / wsph.sum()
                    else:
                        # if len(dist[i][idin]) == 0:
                        ydata[j][px, py, pz] += wdata[j][i]
                #     dist, ids = mtree.query(pos[i], k=8)
                #     for j in wdata.keys():
                #         ydata[j][indxyz[ids, 0], indxyz[ids, 1], indxyz[ids, 2]
                #                  ] += wdata[j][i] * (1 - dist / np.sum(dist)) / 7.
            imax+=1; jmax+=1; kmax+=1
            for i in wdata.keys():
                ydata[i] = ydata[i][imin:imax, jmin:jmax, kmin:kmax]
    return ydata, [imin, jmin, kmin, imax, jmax, kmax]


def cal_sph_neib(idst, dist, posd, pxln, sphkernel, wdata):
    imin, jmin, kmin, imax, jmax, kmax = pxln,pxln,pxln,0,0,0
    if isinstance(wdata, type(np.array([1]))) or isinstance(wdata, type([])):
        if posd == 2:
            ydata = np.zeros((pxln, pxln), dtype=np.float64)  # need to think about reducing this return array later
            for i in range(idst.shape[0]):
                ids = np.array(idst[i])
                wsph = sphkernel(dist[i] / dist[i].max())
                px, py = np.int32(ids/pxln), ids%pxln
                if imin > px.min(): imin = px.min()
                if imax < px.max(): imax = px.max()
                if jmin > py.min(): jmin = py.min()
                if jmax < py.max(): jmax = py.max()
                if wsph.sum() > 0:
                    ydata[px, py] += wdata[i] * wsph / wsph.sum()
            imax+=1; jmax+=1
            ydata=ydata[imin:imax, jmin:jmax]
        elif posd == 3:
            ydata = np.zeros((pxln, pxln, pxln), dtype=np.float32)
            for i in range(idst.shape[0]):
                ids = np.array(idst[i])
                wsph = sphkernel(dist[i] / dist[i].max())
                px, py, pz = np.int32(ids/pxln/pxln), np.int32(ids%(pxln*pxln)/pxln), ids%(pxln*pxln)%pxln
                if imin > px.min(): imin = px.min()
                if imax < px.max(): imax = px.max()
                if jmin > py.min(): jmin = py.min()
                if jmax < py.max(): jmax = py.max()
                if kmin > pz.min(): kmin = pz.min()
                if kmax < pz.max(): kmax = pz.max()
                if wsph.sum() > 0:
                    ydata[px, py, pz] += wdata[i] * wsph / wsph.sum()
            imax+=1; jmax+=1; kmax+=1
            ydata=ydata[imin:imax, jmin:jmax, kmin:kmax]
    else:
        if posd == 2:
            for i in wdata.keys():
                ydata[i] = np.zeros((pxln, pxln), dtype=np.float64)
            for i in range(idst.shape[0]):
                ids = np.array(idst[i])
                wsph = sphkernel(dist[i] / dist[i].max())
                px, py = np.int32(ids/pxln), ids%pxln
                if imin > px.min(): imin = px.min()
                if imax < px.max(): imax = px.max()
                if jmin > py.min(): jmin = py.min()
                if jmax < py.max(): jmax = py.max()
                for j in wdata.keys():
                    if wsph.sum() > 0:
                        ydata[j][px, py] += wdata[j][i] * wsph / wsph.sum()
            imax+=1; jmax+=1
            for i in wdata.keys():
                ydata[i] = ydata[i][imin:imax, jmin:jmax]
        elif posd == 3:
            for i in wdata.keys():
                ydata[i] = np.zeros((pxln, pxln, pxln), dtype=np.float32)
            for i in range(idst.shape[0]):
                ids = np.array(idst[i])
                wsph = sphkernel(dist[i] / dist[i].max())
                px, py, pz = np.int32(ids/pxln/pxln), np.int32(ids%(pxln*pxln)/pxln), ids%(pxln*pxln)%pxln
                if imin > px.min(): imin = px.min()
                if imax < px.max(): imax = px.max()
                if jmin > py.min(): jmin = py.min()
                if jmax < py.max(): jmax = py.max()
                if kmin > pz.min(): kmin = pz.min()
                if kmax < pz.max(): kmax = pz.max()
                for j in wdata.keys():
                    if wsph.sum() > 0:
                        ydata[j][px, py, pz] += wdata[j][i] * wsph / wsph.sum()
            imax+=1; jmax+=1; kmax+=1
            for i in wdata.keys():
                ydata[i] = ydata[i][imin:imax, jmin:jmax, kmin:kmax]
    return ydata, [imin, jmin, kmin, imax, jmax, kmax]


def cal_sph_hsml_v3(ctree, centp, hsml, posd, pxln, sphkernel, wdata):
    imin, jmin, kmin, imax, jmax, kmax = pxln,pxln,pxln,0,0,0
    if isinstance(wdata, type(np.array([1]))) or isinstance(wdata, type([])):
        if posd == 2:
            ydata = np.zeros((pxln, pxln), dtype=np.float64)
            for i in range(hsml.size):
                # idin = dist[i] <= hsml[i]
                # if len(dist[i][idin]) == 0:
                #     idin = [0]  # if the grid point within hsml is none, use the closest one.
                # ids = idst[i][idin]
                idst = np.array(ctree.query_ball_point(centp[i], hsml[i]))
                if len(idst) <= 0:
                    px, py = np.array([np.int32(np.round(centp[i,0]))]), np.array([np.int32(np.round(centp[i,1]))])
                    if px[0]>=pxln: px[0]=pxln-1
                    if py[0]>=pxln: py[0]=pxln-1
                else:
                    px, py = np.int32(idst/pxln), idst%pxln
                dist = np.sqrt((px - centp[i,0])**2 + (py - centp[i, 1])**2)
                wsph = sphkernel(dist / hsml[i])
                if imin > px.min(): imin = px.min()
                if imax < px.max(): imax = px.max()
                if jmin > py.min(): jmin = py.min()
                if jmax < py.max(): jmax = py.max()
                # devided by len(ids) is to change N_e to number density
                if wsph.sum() > 0:
                    ydata[px, py] += wdata[i] * wsph / wsph.sum()
                else:
                    # if len(dist[i][idin]) == 0:
                    ydata[px, py] += wdata[i]
                # else:  # we also add particles with hsml < pixel size to its nearest four pixels.
                #     #    Then, the y-map looks no smoothed (with some noisy pixels).
                #     dist, ids = mtree.query(pos[i], k=4)
                #     ydata[indxyz[ids, 0], indxyz[ids, 1]] += wdata[i] * \
                #         (1 - dist / np.sum(dist)) / 3.
            imax+=1; jmax+=1
            ydata=ydata[imin:imax, jmin:jmax]
        elif posd == 3:
            ydata = np.zeros((pxln, pxln, pxln), dtype=np.float32)
            for i in range(hsml.size):
                # idin = dist[i] <= hsml[i]
                # if len(dist[i][idin]) == 0:
                #     idin = [0]  # if the grid point within hsml is none, use the closest one.                
                # ids = idst[i][idin]
                idst = np.array(ctree.query_ball_point(centp[i], hsml[i]))
                if len(idst) <= 0:
                    px, py, pz = np.array(np.int32(np.round(centp[i,0]))), np.array(np.int32(np.round(centp[i,1]))), np.array(np.int32(np.round(centp[i,2])))
                    if px[0]>=pxln: px[0]=pxln-1
                    if py[0]>=pxln: py[0]=pxln-1
                    if pz[0]>=pxln: pz[0]=pxln-1
                else:
                    px, py, pz = np.int32(idst/pxln/pxln), np.int32(idst%(pxln*pxln)/pxln), idst%(pxln*pxln)%pxln
                dist = np.sqrt((px - centp[i,0])**2 + (py - centp[i, 1])**2 + (pz - centp[i, 2])**2)
                wsph = sphkernel(dist / hsml[i])                
                if imin > px.min(): imin = px.min()
                if imax < px.max(): imax = px.max()
                if jmin > py.min(): jmin = py.min()
                if jmax < py.max(): jmax = py.max()
                if kmin > pz.min(): kmin = pz.min()
                if kmax < pz.max(): kmax = pz.max()
                if wsph.sum() > 0:
                    ydata[px, py, pz] += wdata[i] * wsph / wsph.sum()
                else:
                    # if len(dist[i][idin]) == 0:
                    ydata[px, py, pz] += wdata[i] 
                #     dist, ids = mtree.query(pos[i], k=8)
                #     ydata[indxyz[ids, 0], indxyz[ids, 1], indxyz[ids, 2]
                #           ] += wdata[i] * (1 - dist / np.sum(dist)) / 7.
            imax+=1; jmax+=1; kmax+=1
            ydata=ydata[imin:imax, jmin:jmax, kmin:kmax]
    else:
        ydata = {}
        if posd == 2:
            for i in wdata.keys():
                ydata[i] = np.zeros((pxln, pxln), dtype=np.float64)
            for i in range(hsml.size):
                # idin = dist[i] <= hsml[i]
                # if len(dist[i][idin]) == 0:
                #     idin = [0]  # if the grid point within hsml is none, use the closest one.  
                # ids = idst[i][idin]
                idst = np.array(ctree.query_ball_point(centp[i], hsml[i]))
                if len(idst) <= 0:
                    px, py = np.array(np.int32(np.round(centp[i,0]))), np.array(np.int32(np.round(centp[i,1])))
                    if px[0]>=pxln: px[0]=pxln-1
                    if py[0]>=pxln: py[0]=pxln-1
                else:
                    px, py = np.int32(idst/pxln), idst%pxln
                dist = np.sqrt((px - centp[i,0])**2 + (py - centp[i, 1])**2)
                wsph = sphkernel(dist / hsml[i])
                if imin > px.min(): imin = px.min()
                if imax < px.max(): imax = px.max()
                if jmin > py.min(): jmin = py.min()
                if jmax < py.max(): jmax = py.max()
                for j in wdata.keys():
                    if wsph.sum() > 0:
                        ydata[j][px, py] += wdata[j][i] * wsph / wsph.sum()
                    else:
                        # if len(dist[i][idin]) == 0:
                        ydata[j][px, py] += wdata[j][i]
                #     dist, ids = mtree.query(pos[i], k=4)
                #     for j in wdata.keys():
                #         ydata[j][indxyz[ids, 0], indxyz[ids, 1]] += wdata[j][i] * \
                #             (1 - dist / np.sum(dist)) / 3.
            imax+=1; jmax+=1
            for i in wdata.keys():
                ydata[i] = ydata[i][imin:imax, jmin:jmax]
        elif posd == 3:
            for i in wdata.keys():
                # There is a problem using multiprocessing with (return) really big objects
                # https://bugs.python.org/issue17560
                ydata[i] = np.zeros((pxln, pxln, pxln), dtype=np.float32)
            for i in range(hsml.size):
                # idin = dist[i] <= hsml[i]
                # if len(dist[i][idin]) == 0:
                #     idin = [0]  # if the grid point within hsml is none, use the closest one. 
                # ids = idst[i][idin]
                idst = np.array(ctree.query_ball_point(centp[i], hsml[i]))
                if len(idst) <= 0:
                    px, py, pz = np.array(np.int32(np.round(centp[i,0]))), np.array(np.int32(np.round(centp[i,1]))), np.array(np.int32(np.round(centp[i,2])))
                    #check boundary
                    if px[0]>=pxln: px[0]=pxln-1
                    if py[0]>=pxln: py[0]=pxln-1
                    if pz[0]>=pxln: pz[0]=pxln-1
                else:
                    px, py, pz = np.int32(idst/pxln/pxln), np.int32(idst%(pxln*pxln)/pxln), idst%(pxln*pxln)%pxln
                dist = np.sqrt((px - centp[i,0])**2 + (py - centp[i, 1])**2 + (pz - centp[i, 2])**2)
                wsph = sphkernel(dist / hsml[i])
                if imin > px.min(): imin = px.min()
                if imax < px.max(): imax = px.max()
                if jmin > py.min(): jmin = py.min()
                if jmax < py.max(): jmax = py.max()
                if kmin > pz.min(): kmin = pz.min()
                if kmax < pz.max(): kmax = pz.max()
                for j in wdata.keys():
                    if wsph.sum() > 0:
                        ydata[j][px, py, pz] += wdata[j][i] * wsph / wsph.sum()
                    else:
                        # if len(dist[i][idin]) == 0:
                        ydata[j][px, py, pz] += wdata[j][i]
                #     dist, ids = mtree.query(pos[i], k=8)
                #     for j in wdata.keys():
                #         ydata[j][indxyz[ids, 0], indxyz[ids, 1], indxyz[ids, 2]
                #                  ] += wdata[j][i] * (1 - dist / np.sum(dist)) / 7.
            imax+=1; jmax+=1; kmax+=1
            for i in wdata.keys():
                ydata[i] = ydata[i][imin:imax, jmin:jmax, kmin:kmax]
    return ydata, [imin, jmin, kmin, imax, jmax, kmax]


def cal_sph_neib_v3(ctree, centp, neighbors, pxln, sphkernel, wdata):
    imin, jmin, kmin, imax, jmax, kmax = pxln,pxln,pxln,0,0,0
    if isinstance(wdata, type(np.array([1]))) or isinstance(wdata, type([])):
        if posd == 2:
            ydata = np.zeros((pxln, pxln), dtype=np.float64)  # need to think about reducing this return array later
            for i in range(centp.shape[0]):
                dist, idst = mtree.query(centp[i], neighbors)
                ids = np.array(idst)
                wsph = sphkernel(np.array(dist) / dist.max())
                px, py = np.int32(ids/pxln), ids%pxln
                if imin > px.min(): imin = px.min()
                if imax < px.max(): imax = px.max()
                if jmin > py.min(): jmin = py.min()
                if jmax < py.max(): jmax = py.max()
                if wsph.sum() > 0:
                    ydata[px, py] += wdata[i] * wsph / wsph.sum()
            imax+=1; jmax+=1
            ydata=ydata[imin:imax, jmin:jmax]
        elif posd == 3:
            ydata = np.zeros((pxln, pxln, pxln), dtype=np.float32)
            for i in range(centp.shape[0]):
                dist, idst = mtree.query(centp[i], neighbors)
                ids = np.array(idst)
                wsph = sphkernel(np.array(dist) / dist.max())
                px, py, pz = np.int32(ids/pxln/pxln), np.int32(ids%(pxln*pxln)/pxln), ids%(pxln*pxln)%pxln
                if imin > px.min(): imin = px.min()
                if imax < px.max(): imax = px.max()
                if jmin > py.min(): jmin = py.min()
                if jmax < py.max(): jmax = py.max()
                if kmin > pz.min(): kmin = pz.min()
                if kmax < pz.max(): kmax = pz.max()
                if wsph.sum() > 0:
                    ydata[px, py, pz] += wdata[i] * wsph / wsph.sum()
            imax+=1; jmax+=1; kmax+=1
            ydata=ydata[imin:imax, jmin:jmax, kmin:kmax]
    else:
        if posd == 2:
            for i in wdata.keys():
                ydata[i] = np.zeros((pxln, pxln), dtype=np.float64)
            for i in range(centp.shape[0]):
                dist, idst = mtree.query(centp[i], neighbors)
                ids = np.array(idst)
                wsph = sphkernel(np.array(dist) / dist.max())
                px, py = np.int32(ids/pxln), ids%pxln
                if imin > px.min(): imin = px.min()
                if imax < px.max(): imax = px.max()
                if jmin > py.min(): jmin = py.min()
                if jmax < py.max(): jmax = py.max()
                for j in wdata.keys():
                    if wsph.sum() > 0:
                        ydata[j][px, py] += wdata[j][i] * wsph / wsph.sum()
            imax+=1; jmax+=1
            for i in wdata.keys():
                ydata[i] = ydata[i][imin:imax, jmin:jmax]
        elif posd == 3:
            for i in wdata.keys():
                ydata[i] = np.zeros((pxln, pxln, pxln), dtype=np.float32)
            for i in range(centp.shape[0]):
                dist, idst = mtree.query(centp[i], neighbors)
                ids = np.array(idst)
                wsph = sphkernel(np.array(dist) / dist.max())
                px, py, pz = np.int32(ids/pxln/pxln), np.int32(ids%(pxln*pxln)/pxln), ids%(pxln*pxln)%pxln
                if imin > px.min(): imin = px.min()
                if imax < px.max(): imax = px.max()
                if jmin > py.min(): jmin = py.min()
                if jmax < py.max(): jmax = py.max()
                if kmin > pz.min(): kmin = pz.min()
                if kmax < pz.max(): kmax = pz.max()
                for j in wdata.keys():
                    if wsph.sum() > 0:
                        ydata[j][px, py, pz] += wdata[j][i] * wsph / wsph.sum()
            imax+=1; jmax+=1; kmax+=1
            for i in wdata.keys():
                ydata[i] = ydata[i][imin:imax, jmin:jmax, kmin:kmax]
    return ydata, [imin, jmin, kmin, imax, jmax, kmax]


def SPH_smoothing(wdata, pos, pxls, neighbors, hsml=None, pxln=None, Memreduce=False, 
                  Ncpu=None, Ntasks=None, kernel_name='cubic'):
    r"""SPH smoothing for given data

    Parameters
    ----------
    wdata    : the data to be smoothed. Type: array or list of arrays
    pos      : the position of your SPH particles. Type: array.
                It can be 3D or 2D
    pxls     : grid pixel size of the mesh, where the data is smoothed into.
                Type: float.
                It must be in the same units of positions
    neighbors: how many nearby mesh points the SPH particles smoothed into.
                Type: int. Note this has to be rescaled to the image pixel size!
    hsml     : smoothing length of these SPH particles. Type: array or float.
                If it is None, then the neighbours will be used to do the smoothing.
    pxln     : number of pixels for the mesh. Type: int. Must be set.
                # If it is None (Default), it will calculate from the particle positions.
                # I.E. pxln = (max(pos)-min(pos))/pxls
    Memreduce: Try to reduce the memory in parallel calculation by passing the cKDTree class. This overcomes the 
                memory cost in the cKDTree query, but should require Python>3.8 to pass the class > 4Gb.
                Default: False.
    Ncpu     : number of CPU for parallel calculation. Type: int. Default: None, all cpus from the
                computer will be used.
    Ntasks   : number of tasks for separating the calculation. Type: int. Default: None,
                the same number of Ncpu will be used. Ideally, it should be larger than or equal to the number of Ncpu.
                Set this to a much larger value if you encounter an 'IO Error: bad message length'
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

    # shared memory data
    # global shm_pos, shm_hsml, shm_idst, shm_dist
    # shared_array_base = Array(ctypes.c_float, pos.size)
    # shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    # shm_pos = shared_array.reshape(pos.shape[0], pos.shape[1])
    # shm_pos[:, 0] = np.copy(pos[:, 0]); shm_pos[:, 1] = np.copy(pos[:, 1]); shm_pos[:, 2] = np.copy(pos[:, 2])
    # if  hsml is not None:
    #     shared_array_base = Array(ctypes.c_float, hsml.size)
    #     shm_hsml = np.ctypeslib.as_array(shared_array_base.get_obj())
    #     shm_hsml = np.copy(hsml)
    memlog('Init smoothing')
    
    SD = pos.shape[1]
    if SD not in [2, 3]:
        raise ValueError(
            "pos shape %d is not correct, the second dimension must be 2 or 3" % pos.shape)

    minx = -pxls * pxln / 2

    if SD == 2:
        idls= np.lexsort((pos[:,1],pos[:,0]))
        pos = (pos - [minx, minx]) / pxls  # in units of pixel size
        x, y = np.meshgrid(np.arange(0.5, pxln, 1.0), np.arange(0.5, pxln, 1.0), indexing='ij')
        indxyz = np.concatenate((x.reshape(x.size, 1), y.reshape(y.size, 1)), axis=1)
        if isinstance(wdata, type(np.array([1]))) or isinstance(wdata, type([])):
            wdata = wdata[idls]
            ydata = np.zeros((pxln, pxln), dtype=np.float64)
        elif isinstance(wdata, type({})):
            if len(wdata) > 20:
                raise ValueError("Too many data to be smoothed %d" % len(wdata))
            else:
                ydata = {}
                for i in wdata.keys():
                    wdata[i] = wdata[i][idls]
                    ydata[i] = np.zeros((pxln, pxln), dtype=np.float32)
    else:
        idls= np.lexsort((pos[:,2],pos[:,1],pos[:,0]))
        pos = (pos - [minx, minx, minx]) / pxls
        x, y, z = np.meshgrid(np.arange(0.5, pxln, 1.0), np.arange(0.5, pxln, 1.0),
                              np.arange(0.5, pxln, 1.0), indexing='ij')
        indxyz = np.concatenate((x.reshape(x.size, 1), y.reshape(y.size, 1),
                                 z.reshape(z.size, 1)), axis=1)
        if isinstance(wdata, type(np.array([1]))) or isinstance(wdata, type([])):
            wdata = wdata[idls]
            ydata = np.zeros((pxln, pxln, pxln), dtype=np.float64)
        else:
            if len(wdata) > 20:
                raise ValueError("Too many data to be smoothed %d" % len(wdata))
            else:
                ydata = {}
                for i in wdata.keys():
                    wdata[i] = wdata[i][idls]
                    ydata[i] = np.zeros((pxln, pxln, pxln), dtype=np.float64)

    # lex sorted arrays before next step
    pos = pos[idls]
    if hsml is not None:
        hsml = hsml[idls] / pxls

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

    mtree = cKDTree(indxyz, boxsize=pxln, leafsize=np.int32(np.ceil(pxln/20)))
    # indxyz = np.int32(indxyz)
    memlog('After cKDTree ')

    freeze_support()
    if Ncpu is None:
        NUMBER_OF_PROCESSES = cpu_count()
    else:
        NUMBER_OF_PROCESSES = Ncpu
    if Ntasks is None:
        Ntasks = NUMBER_OF_PROCESSES
    N = np.int32(pos.shape[0] / Ntasks)
    listn = np.append(np.arange(0, pos.shape[0], N), pos.shape[0])

    # Create queues
    task_queue = Queue()
    done_queue = Queue()
    
    if (sys.version_info.major >= 3) and (sys.version_info.minor >= 8) and (Memreduce):  # only tested with python3.8
        # we can transfer the cKDTree class directly, seems no limitations on the transferred data size now
        print("Directly pass the cKDTree to tasks with Python version", sys.version)
        if hsml is None:
            if isinstance(wdata, type(np.array([1]))) or isinstance(wdata, type([])):
                Tasks = [(cal_sph_neib_v3, (mtree, pos[listn[i]:listn[i + 1]], neighbors, SD, pxln,
                                         sphkernel, wdata[listn[i]:listn[i + 1]])) for i in range(listn.size-1)]
            else:
                Tasks = []
                for i in range(listn.size-1):
                    tmpwd = {}
                    for j in wdata.keys():
                        tmpwd[j] =  wdata[listn[i]:listn[i + 1]]
                    Tasks.append((cal_sph_neib_v3, (mtree, pos[listn[i]:listn[i + 1]], neighbors,
                                                 SD, pxln, sphkernel, tmpwd)))
        else:
            if isinstance(wdata, type(np.array([1]))) or isinstance(wdata, type([])):
                Tasks = [(cal_sph_hsml_v3, (mtree, pos[listn[i]:listn[i + 1]], hsml[listn[i]:listn[i + 1]],
                                         SD, pxln, sphkernel, wdata[listn[i]:listn[i + 1]])) for i in range(listn.size-1)]
            else:
                Tasks = []
                for i in range(listn.size-1):
                    tmpwd = {}
                    for j in wdata.keys():
                        tmpwd[j] =  wdata[listn[i]:listn[i + 1]]
                    Tasks.append((cal_sph_hsml_v3, (mtree, pos[listn[i]:listn[i + 1]], hsml[listn[i]:listn[i + 1]],
                                                 SD, pxln, sphkernel, tmpwd)))
    else: # we need do the query first as the sending data size is limited. Sometimes We can use Ntasks to overcome the memory issue.
        if hsml is None:
            dist, idst = mtree.query(pos, neighbors, n_jobs=NUMBER_OF_PROCESSES)  # estimate the neighbors and distance
            memlog('After query ')
            if isinstance(wdata, type(np.array([1]))) or isinstance(wdata, type([])):
                Tasks = [(cal_sph_neib, (idst[listn[i]:listn[i + 1]], dist[listn[i]:listn[i + 1]], SD, pxln,
                                         sphkernel, wdata[listn[i]:listn[i + 1]])) for i in range(listn.size-1)]
            else:
                Tasks = []
                for i in range(listn.size-1):
                    tmpwd = {}
                    for j in wdata.keys():
                        tmpwd[j] =  wdata[listn[i]:listn[i + 1]]
                    Tasks.append((cal_sph_neib, (idst[listn[i]:lintn[i + 1]], dist[listn[i]:listn[i + 1]],
                                                 SD, pxln, sphkernel, tmpwd)))
        else:
            # However, this may help with the extremly memory cost of cKDTree query
            # idst=np.empty(0, dtype=object)
            # for i in range(listn.size-1):
            #     tid = mtree.query_ball_point(pos[listn[i]:listn[i + 1]], hsml[listn[i]:listn[i + 1]], n_jobs=NUMBER_OF_PROCESSES)
            #     idst=np.append(idst, np.array([np.array(xi) for xi in tid]))
            # del(tid)
            idst = mtree.query_ball_point(pos, hsml, n_jobs=NUMBER_OF_PROCESSES)
            memlog('After query ')
            if isinstance(wdata, type(np.array([1]))) or isinstance(wdata, type([])):
                Tasks = [(cal_sph_hsml, (idst[listn[i]:listn[i + 1]], pos[listn[i]:listn[i + 1]], hsml[listn[i]:listn[i + 1]],
                                         SD, pxln, sphkernel, wdata[listn[i]:listn[i + 1]])) for i in range(listn.size-1)]
            else:
                Tasks = []
                for i in range(listn.size-1):
                    tmpwd = {}
                    for j in wdata.keys():
                        tmpwd[j] =  wdata[listn[i]:listn[i + 1]]
                    Tasks.append((cal_sph_hsml, (idst[listn[i]:listn[i + 1]], pos[listn[i]:listn[i + 1]], hsml[listn[i]:listn[i + 1]],
                                                 SD, pxln, sphkernel, tmpwd)))
        # Tasks = [(cal_sph_hsml, (range(i * N, (i + 1) * N), mtree, pos, hsml, pxln, indxyz,
        #                          sphkernel, wdata)) for i in range(NUMBER_OF_PROCESSES)]

    # Submit tasks
    for task in Tasks:
        task_queue.put(task)

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        Process(target=worker, args=(task_queue, done_queue)).start()

    # Get results
    for i in range(len(Tasks)):
        if isinstance(wdata, type(np.array([1]))) or isinstance(wdata, type([])):
            td, idx = done_queue.get()
            if SD == 2:
                ydata[idx[0]:idx[3], idx[1]:idx[4]] += td
            elif SD == 3:
                ydata[idx[0]:idx[3], idx[1]:idx[4], idx[2]:idx[5]] += td
        else:
            for j in wdata.keys():
                if SD == 2:
                    ydata[j][idx[0]:idx[3], idx[1]:idx[4]] += td[j]
                elif SD == 3:
                    ydata[j][idx[0]:idx[3], idx[1]:idx[4], idx[2]:idx[5]] += td[j]

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
