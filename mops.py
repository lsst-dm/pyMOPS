# This file is part of the LSST Solar System Processing.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
mops

LSST Solar System Processing routines for
linking of observations of moving objects.

Classical MOPS. Algorithm based on Kubica et al. 2007

Implementation: S. Eggl 20191215
"""

# Accelerators
import numpy as np
import numba

# Clustering
import scipy.spatial as scsp
import sklearn.cluster as cluster

#######################################
#
# Classical Moving Object Processing
# utility Routines
#
#######################################
@numba.njit
def norm(v):
    """Calculate 2-norm for vectors 1D and 2D arrays.

    Parameters:
    -----------
    v ... vector or 2d array of vectors

    Returns:
    --------
    u ... norm (length) of vector(s)

    """

    if(v.ndim == 1):
        n = np.vdot(v, v)
    elif(v.ndim == 2):
        lv = len(v[:, 0])
        n = np.zeros(lv)
        for i in range(lv):
            n[i] = np.vdot(v[i, :], v[i, :])
    else:
        raise TypeError

    return np.sqrt(n)


@numba.njit
def unit_vector(v):
    """Normalize vectors (1D and 2D arrays).

    Parameters:
    -----------
    v ... vector or 2d array of vectors

    Returns:
    --------
    u ... unit lenght vector or 2d array of vectors of unit lenght

    """

    if(v.ndim == 1):
        u = v/norm(v)
    elif(v.ndim == 2):
        lv = len(v[:, 0])
        dim = len(v[0, :])
        u = np.zeros((lv, dim))
        for i in range(lv):
            n = norm(v[i, :])
            for j in range(dim):
                u[i, j] = v[i, j]/n
    else:
        raise TypeError

    return u


def rotate_vector(angle, axis, vector, deg=True):
    """Rotate vector about arbitrary axis by an angle.

    Parameters:
    -----------
    angle  ... rotation angle 
    axis   ... rotation axis: numpy array (n,3)
    vector ... vector to be rotated: numpy array(n,3)
    deg    ... True: angles given in [deg], 
               False: angles given in [rad]
               
    Returns:
    --------
    vrot   ... rotated vector
    """
    if(deg):
        angl=np.deg2rad(angle)
    else:
        angl=angle
        
    sina = np.sin(angl)
    cosa = np.cos(angl)
    
    u = unit_vector(axis)
    uxv = np.cross(u, vector)
    uuv = u*(np.vdot(u, vector))
    vrot = uuv.T+cosa*np.cross(uxv, u).T+sina*uxv.T
    return vrot.T

def propagate_arrows_linear(x, v, t, tp):
    """Linear propagation of arrows to the same time.

    Parameters:
    -----------
    x  ... array of 3D positions
    v  ... array of 3D velocities
    t  ... array of epochs for states (x,v)
    tp ... epoch to propagate state to

    Returns:
    --------
    xp ... array of propagated 3D positions
    dt ... array of time deltas wrt the propatation epoch: tp-t
    """
    dt = tp-t
    xp = x + (v*np.array([dt, dt, dt]).T)
    return xp, dt

def propagate_arrows_celestial_sphere(x, v, t, tp):
    
    """Linear propagation of arrows to the same time.
    Parameters:
    -----------
    x  ... array of 3D positions
    v  ... array of 3D velocities
    t  ... array of epochs for states (x,v)
    tp ... epoch to propagate state to

    Returns:
    --------
    xp ... array of propagated 3D positions
    vp ... array of propagated 3D velocities
    dt ... array of time deltas wrt the propatation epoch: tp-t
    """
    dt = tp-t
    xr = x + v*np.array([dt, dt, dt]).T
    xp = unit_vector(xr)
    vp = np.divide(xp-x,np.array([dt, dt, dt]).T)
    
    return xp, vp, dt

def xv2dradecdt(x,v):
    """
    Find on-sky velocities dRA/dt and dDEC/dt from ICRF positions and velocities
    Parameters:
    -----------
    x  ... array of 3D positions
    v  ... array of 3D velocities

    Returns:
    --------
    dradt  ... dRA/dt
    ddecdt ... dDEC/dt
    """
    if(x.ndim == 1):
        dradt = (x[0]*v[1]-x[1]*v[0])/np.sqrt(x[0]*x[0]+v[0]*v[0])
        ddecdt = v[2]/np.sqrt(1-x[2]*x[2])
        
    elif(x.ndim == 2):
        dradt = np.divide((x[:,0]*v[:,1]-x[:,1]*v[:,0]),np.sqrt(x[:,0]*x[:,0]+v[:,0]*v[:,0]))
        ddecdt = np.dividie(v[:,2],np.sqrt(1-x[:,2]*x[:,2]))
    else:
        raise TypeError

    return dradt, ddecdt    

def xyz2radec(xyz, deg=True):
    "Transform heliocentric ICRF coordinates to RA DEC"
    
    if(xyz.ndim == 1):
        r = np.linalg.norm(xyz)
        RA = np.atan2(xyz[:,1],xyz[:,0])
        DEC = np.arccos(np.div(xyz[:,2],r))
    
    elif(xyz.ndim == 2):
        r = np.linalg.norm(xyz, axis=1)
        RA = np.atan2(xyz[:,1],xyz[:,0])
        DEC = np.arccos(np.div(xyz[:,2],r))
                  
    else:
        raise TypeError
    
    if(deg):
        RA_out = np.rad2deg(RA)
        DEC_out = np.rad2deg(DEC)
    else:
        RA_out = RA
        DEC_out = DEC
                      
    return RA_out, DEC_out


def clustering(data, *args, **kwargs):
    """ Clustering data with DBSCAN.
    
    Parameters:
    -----------
    data ... data to cluster with DBSCAN
    
    Returns:
    --------
    clstr ... DBSCAN output 
    """
    clstr = cluster.DBSCAN(*args, **kwargs).fit(data)
    return clstr


def radec2icrfu(ra, dec, **kwargs):
    """Convert Right Ascension and Declination to ICRF xyz unit vector

    Parameters:
    -----------
    ra ... Right Ascension
    dec ... Declination

    Keyword Arguments:
    ------------------
    deg ... Are angles in degrees: True or radians: False

    Returns:
    --------
    x,y,z ... 3D vector of unit length (ICRF)
    """
    options = {'deg': False}

    options.update(kwargs)

    if (options['deg']):
        a = np.deg2rad(ra)
        d = np.deg2rad(dec)
    else:
        a = ra
        d = dec

    cosd = np.cos(d)
    x = cosd*np.cos(a)
    y = cosd*np.sin(a)
    z = np.sin(d)

    return [x, y, z]


@numba.njit
def RaDec2IcrfU_deg(ra, dec):
    """Convert Right Ascension and Declination to ICRF xyz unit vector

    Parameters:
    -----------
    ra ... Right Ascension [deg]
    dec ... Declination [deg]

    Returns:
    --------
    x,y,z ... 3D vector of unit length (ICRF)
    """

    a = np.deg2rad(ra)
    d = np.deg2rad(dec)

    cosd = np.cos(d)
    x = cosd*np.cos(a)
    y = cosd*np.sin(a)
    z = np.sin(d)

    return np.array([x, y, z])


@numba.njit
def RaDec2IcrfU_rad(ra, dec):
    """Convert Right Ascension and Declination to ICRF xyz unit vector

    Parameters:
    -----------
    ra ... Right Ascension [rad]
    dec ... Declination [rad]

    Returns:
    --------
    x,y,z ... 3D vector of unit length (ICRF)
    """

    cosd = np.cos(dec)
    x = cosd*np.cos(ra)
    y = cosd*np.sin(ra)
    z = np.sin(dec)

    return np.array([x, y, z])


def CullSameTimePairs(pairs, df, time_column_name):
    """Cull all pairs that occur at the same time."""

    tn = time_column_name
    goodpairs = []
    goodpairs_app = goodpairs.append
    for p in pairs:
        if (df[tn][p[0]] < df[tn][p[1]]):
            goodpairs_app(p)
    return goodpairs

def SelectTrackletsFromObsData(pairs, df, time_column_name):
    """Select data in trackelts from observation data frame"""
    goodpairs = CullSameTimePairs(pairs, df, time_column_name)
    index_list = np.unique((np.array(goodpairs).flatten()))
    df2 = (df.iloc[index_list]).reset_index()

    return df2, goodpairs


def correct_pairs(df, pairs):
    """ Find out which pairs of observations can actually be linked into
        a tracklet by ruling out pairs that are observed at the same time."""

    p = np.array(pairs)
    # find out which pairs are actually good
    pair_obj = np.array([df['obj'][p[:, 0]].values,
                         df['obj'][p[:, 1]].values]).T
    # print(pair_obj)
    # correct=np.where(df['obj'][p[:,0]].values == df['obj'][p[:,1]].values)
    correct = np.where(pair_obj[:, 0] == pair_obj[:, 1])
    return correct[0]


def create_arrows(df):
    """create arrows from dataframe containing
       RADEC observations and timestamps"""

    # Transform RADEC observations into positions on the unit sphere (US)
    xyz = radec2icrfu(df['RA'], df['DEC'], deg=True)
    posu = np.array([xyz[0].values, xyz[1].values, xyz[2].values]).T

    # So we build your KDTree
    kdtree = scsp.cKDTree(posu, leafsize=16, compact_nodes=True,
                          copy_data=False, balanced_tree=True, boxsize=None)
    # rule out un-physical combinations of observations with kdtree
    # c_aupd=173.145
    # rkdtree=c_aupd*0.001*dt
    dt = df.time.values[1]-df.time.values[0]
    rkdtree = 100/150e6*86400 # limit of v<100km/s
    # Query KDTree for good pairs of observations
    pairs = kdtree.query_pairs(rkdtree*dt)

    # Discard impossible pairs (same timestamp)
    [df2, goodpairs] = SelectTrackletsFromObsData(pairs, df, 'time')

    x = []
    x_add = x.append
    v = []
    v_add = v.append
    t = []
    t_add = t.append
    for p in goodpairs:
        x_add(posu[p[0]])
        t_add(df['time'][p[0]])
        v_add((posu[p[1]]-posu[p[0]])/(df['time'][p[1]]-df['time'][p[0]]))

    return np.array(x), np.array(v), np.array(t), goodpairs
