import numpy as np
import random

# in place, hopefully
def apply_symmetry_features_example(many_planes, i, s):
    assert len(many_planes.shape) == 4
    if (s & 1) != 0: # flip x
        many_planes[i,:,:,:] = many_planes[i,::-1,:,:]
    if (s & 2) != 0: # flip y
        many_planes[i,:,:,:] = many_planes[i,:,::-1,:]
    if (s & 4) != 0: # swap x and y
        many_planes[i,:,:,:] = many_planes[i,:,:,:].transpose(1, 0, 2)


def apply_symmetry_planes(planes, s):
    assert len(planes.shape) == 3
    if (s & 1) != 0: # flip x
        np.copyto(planes, planes[::-1,:,:])
    if (s & 2) != 0: # flip y
        np.copyto(planes, planes[:,::-1,:])
    if (s & 4) != 0: # swap x and y
        np.copyto(planes, np.transpose(planes[:,:,:], (1,0,2)))

def apply_symmetry_plane(plane, s):
    assert len(plane.shape) == 2
    if (s & 1) != 0: # flip x
        np.copyto(plane, plane[::-1,:])
    if (s & 2) != 0: # flip y
        np.copyto(plane, plane[:,::-1])
    if (s & 4) != 0: # swap x and y
        np.copyto(plane, np.transpose(plane[:,:], (1,0)))

def invert_symmetry_plane(plane, s):
    assert len(plane.shape) == 2
    # note reverse order of 4,2,1
    if (s & 4) != 0: # swap x and y
        np.copyto(plane, np.transpose(plane[:,:], (1,0)))
    if (s & 2) != 0: # flip y
        np.copyto(plane, plane[:,::-1])
    if (s & 1) != 0: # flip x
        np.copyto(plane, plane[::-1,:])

def apply_symmetry_vertex(vertex, N, s):
    assert vertex.size == 2
    if (s & 1) != 0: # flip x
        vertex[0] = N - vertex[0] - 1
    if (s & 2) != 0: # flip y
        vertex[1] = N - vertex[1] - 1
    if (s & 4) != 0: # swap x and y
        np.copyto(vertex, vertex[::-1])
    assert 0 <= vertex[0] < N
    assert 0 <= vertex[1] < N

def get_symmetry_vertex_tuple(vertex, N, s):
    x,y = vertex
    if (s & 1) != 0: # flip x
        x = N - x - 1
    if (s & 2) != 0: # flip y
        y = N - y - 1
    if (s & 4) != 0: # swap x and y
        x,y = y,x
    assert 0 <= x < N
    assert 0 <= y < N
    return (x,y)

def get_inverse_symmetry_vertex_tuple(vertex, N, s):
    x,y = vertex
    # note reverse order of 4,2,1
    if (s & 4) != 0: # swap x and y
        x,y = y,x
    if (s & 2) != 0: # flip y
        y = N - y - 1
    if (s & 1) != 0: # flip x
        x = N - x - 1
    assert 0 <= x < N
    assert 0 <= y < N
    return (x,y)

def make_symmetry_batch(features):
    assert len(features.shape) == 3
    N = features.shape[0]
    Nfeat = features.shape[2]
    feature_batch = np.empty((8, N, N, Nfeat), dtype=features.dtype)
    for s in xrange(8):
        feature_batch[s,:,:,:] = features
        apply_symmetry_planes(feature_batch[s,:,:,:], s)
    return feature_batch

def average_plane_over_symmetries(planes, N):
    assert planes.shape == (8, N*N)
    planes = planes.reshape((8, N, N))
    for s in xrange(8):
        invert_symmetry_plane(planes[s,:,:], s)
    mean_plane = planes.mean(axis=0)
    mean_plane = mean_plane.reshape((N*N,))
    return mean_plane

