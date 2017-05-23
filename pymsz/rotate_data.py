import numpy as np


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
    return pos[:, :2]

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
