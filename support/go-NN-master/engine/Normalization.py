#!/usr/bin/python
import numpy as np
import os
import math
import sys
import math
import random

def apply_grand_normalization(feature_planes, grand_mean, grand_rescaling_factor):
    np.copyto(feature_planes, (feature_planes - grand_mean) * grand_rescaling_factor)

# some parameters I measured on KGS for Features.make_feature_planes_stones_3liberties_4history_ko
def apply_grand_normalization_A(feature_planes):
    apply_grand_normalization(feature_planes, grand_mean=0.154, grand_rescaling_factor=2.77)


def apply_featurewise_normalization(feature_planes, feature_means, feature_rescaling_factors):
    np.copyto(feature_planes, (feature_planes - feature_means) * feature_rescaling_factors)

# some parameters I measured on KGS for Features.make_feature_planes_stones_3liberties_4history_ko
# Capped rescaling factors at 10 and made some stuff up for the ones plane.
def apply_featurewise_normalization_B(feature_planes):
    apply_featurewise_normalization(feature_planes,
         feature_means=np.array([0.146, 0.148, 0.706, 0.682, 0.005, 0.018, 0.124, 0.004, 0.018, 0.126, 0.003, 0.003, 0.003, 0.003, 0]),
         feature_rescaling_factors=np.array([2.829, 2.818, 2.195, 2.148, 10, 7.504, 3.0370, 10, 7.576, 3.013, 10, 10, 10, 10, 10]))

# some parameters I measured on GoGoD for Features.make_feature_planes_stones_4liberties_4history_ko_4captures
# Capped rescaling factors at 10 and made some stuff up for the ones plane.
def apply_featurewise_normalization_C(feature_planes):
    apply_featurewise_normalization(feature_planes,
            feature_means=np.array([1.482e-01, 1.498e-01, 7.021e-01, 0.682, 4.428e-03, 1.769e-02, 2.616e-02, 9.988e-02, 4.065e-03, 1.742e-02, 2.636e-02, 1.019e-01, 2.756e-03, 2.745e-03,
                                    2.732e-03, 2.720e-03, 7.553e-05, 2.534e-03, 3.763e-04, 1.052e-04, 7.250e-05]),
            feature_rescaling_factors=np.array([2.815, 2.802, 2.187, 2.148, 10, 7.585, 6.266, 3.335, 10, 7.643, 6.242, 3.305, 10, 10, 
                                                10, 10, 10, 10, 10, 10, 10]))

# like C but for Features.make_feature_planes_stones_4liberties_4history_ko_4captures_komi. 
def apply_featurewise_normalization_D(feature_planes):
    apply_featurewise_normalization(feature_planes,
            feature_means=np.array([1.482e-01, 1.498e-01, 7.021e-01, 0.682, 4.428e-03, 1.769e-02, 2.616e-02, 9.988e-02, 4.065e-03, 1.742e-02, 2.636e-02, 1.019e-01, 2.756e-03, 2.745e-03,
                                    2.732e-03, 2.720e-03, 7.553e-05, 2.534e-03, 3.763e-04, 1.052e-04, 7.250e-05, 0.0]),
            feature_rescaling_factors=np.array([2.815, 2.802, 2.187, 2.148, 10, 7.585, 6.266, 3.335, 10, 7.643, 6.242, 3.305, 10, 10, 
                                                10, 10, 10, 10, 10, 10, 10, 0.2]))



def get_svd_normalized_features(feature_planes, feature_means, whitening_matrix):
    return np.dot(feature_planes - feature_means, whitening_matrix)



def get_sample(npz_dir, Nfiles):
    print "getting sample data from", npz_dir
    filenames = os.listdir(npz_dir)[:Nfiles]
    random.shuffle(filenames)

    feature_batches = []
    for fn in filenames:
        filename = os.path.join(npz_dir, fn)
        npz = np.load(filename)
        features = npz['feature_planes'].astype(np.float64)
        npz.close()
        feature_batches.append(features)

    Nperfile = feature_batches[0].shape[0]
    N = feature_batches[0].shape[1]
    Nfeat = feature_batches[0].shape[3]

    big_matrix = np.empty((Nperfile*Nfiles*N*N, Nfeat), dtype=np.float32)

    for i in xrange(len(feature_batches)):
        big_matrix[i*Nperfile*N*N:(i+1)*Nperfile*N*N, :] = feature_batches[i].reshape(Nperfile*N*N, Nfeat)

    print "got %d samples from %d files" % (big_matrix.shape[0], Nfiles)
    return big_matrix


def compute_grand_normalization(sample):
    grand_mean = sample.mean()
    grand_variance = np.square(sample).mean() - grand_mean**2
    grand_rescaling_factor = 1/math.sqrt(grand_variance)
    print "grand_mean =", grand_mean
    print "grand_variance =", grand_variance
    print "grand_resaling_factor =", grand_rescaling_factor


def compute_featurewise_normalization(sample):
    feature_means = sample.mean(axis=0)
    feature_variances = np.square(sample).mean(axis=0) - np.square(feature_means)
    feature_rescaling_factors = np.reciprocal(np.sqrt(feature_variances))
    print "feature_means =\n", repr(feature_means)
    print "feature_variances =\n", repr(feature_variances)
    print "feature_rescaling_factors = \n", repr(feature_rescaling_factors)

def compute_svd_normalization(samples, Ndiscard, max_rescale):
    # S is list of singular values in descending order
    # Each row of V is a list of the weights of the features in a given principal component
    centered_samples = samples - samples.mean(axis=0) # subtract columnwise means
    U, S, V = np.linalg.svd(centered_samples, full_matrices=False)
    Nsamp = samples.shape[0]
    component_stddevs = S / math.sqrt(Nsamp)

    print "singular values ="
    print S
    print "component standard deviations ="
    print component_stddevs
    print "V matrix ="
    print V

    Nfeat = samples.shape[1]
    rescaling_factors = np.minimum(np.reciprocal(component_stddevs[:Nfeat-Ndiscard]), max_rescale)
    whitening_matrix = np.dot(V[:Nfeat-Ndiscard].T, np.diag(rescaling_factors))

    print "Ndiscard =", Ndiscard
    print "max_rescale =", max_rescale
    print "rescaling_factors ="
    print rescaling_factors
    print "whitening_matrix ="
    print repr(whitening_matrix)

def test_normalizations():
    np.set_printoptions(precision=3, linewidth=200)

    #npz_dir = '/home/greg/coding/ML/go/NN/data/KGS/processed/stones_3lib_4hist_ko_Nf15-randomized-2'
    npz_dir = '/home/greg/coding/ML/go/NN/data/GoGoD/move_examples/stones_4lib_4hist_ko_4cap_Nf21/train'
    sample = get_sample(npz_dir, Nfiles=100)

    print "Grand normalization:"
    compute_grand_normalization(sample)
    print
    grand_mean = 0.15368
    grand_rescaling_factor = 2.77283290021
    print "after grand normalization with mean = %f, rescaling factor = %f:" % (grand_mean, grand_rescaling_factor)
    norm_sample = sample.copy()
    apply_grand_normalization(norm_sample, grand_mean, grand_rescaling_factor)
    compute_grand_normalization(norm_sample)
    print
    print

    print "Feature-wise normalization:"
    compute_featurewise_normalization(sample)
    print
    feature_means = np.array([1.463e-01, 1.478e-01, 7.058e-01, 1.000e+00, 4.529e-03, 1.809e-02, 1.237e-01, 4.053e-03, 1.774e-02, 1.260e-01, 2.758e-03, 2.749e-03, 2.738e-03, 2.729e-03, 6.471e-05])
    feature_rescaling_factors = np.array([2.829, 2.818, 2.195, 1, 14.893, 7.504, 3.037, 15.739, 7.576, 3.013, 19.067, 19.098, 19.136, 19.17, 124.319])
    print "after grand normalization with feature_means ="
    print feature_means
    print "and feature_rescaling_factors ="
    print feature_rescaling_factors
    print
    norm_sample = sample.copy()
    apply_featurewise_normalization(norm_sample, feature_means, feature_rescaling_factors)
    compute_featurewise_normalization(norm_sample)
    print
    print

    print "SVD normalization:"
    compute_svd_normalization(sample, Ndiscard=4, max_rescale=10)
    print
    feature_means = np.array([1.463e-01, 1.478e-01, 7.058e-01, 1.000e+00, 4.529e-03, 1.809e-02, 1.237e-01, 4.053e-03, 1.774e-02, 1.260e-01, 2.758e-03, 2.749e-03, 2.738e-03, 2.729e-03, 6.471e-05])
    whitening_matrix = np.array(
      [[  4.896e-01,   8.627e-01,  -5.059e-01,  -9.773e-01,  -8.486e-01,   6.300e-01,  -2.864e-03,   3.666e-02,   3.972e-02,   1.383e-01,  -2.034e-03],
       [  5.107e-01,  -8.432e-01,  -3.439e-01,   1.053e+00,   2.028e-01,  -1.071e+00,   5.276e-03,  -1.653e-02,  -1.063e-01,  -6.966e-02,   5.143e-04],
       [ -1.000e+00,  -1.944e-02,   8.498e-01,  -7.573e-02,   6.458e-01,   4.414e-01,  -2.412e-03,  -2.013e-02,   6.658e-02,  -6.860e-02,   1.520e-03],
       [ -1.913e-17,  -4.535e-17,   0.000e+00,   4.224e-16,  -1.643e-15,   5.215e-16,  -3.578e-15,  -1.968e-16,   7.443e-16,   1.391e-15,   8.676e-15],
       [  1.097e-02,   1.540e-02,  -2.354e-01,  -1.868e-01,  -4.651e+00,   1.108e+00,  -4.241e-03,   1.199e-01,   1.351e-02,   1.668e-01,  -8.618e-04],
       [  4.540e-02,   6.488e-02,  -2.013e+00,  -1.943e+00,   2.234e+00,  -2.755e-01,   5.424e-04,  -3.261e-02,   1.135e-02,   2.028e-02,  -1.027e-03],
       [  4.333e-01,   7.824e-01,   1.742e+00,   1.153e+00,   1.569e+00,  -2.023e-01,   8.342e-04,  -5.069e-02,   1.487e-02,  -4.884e-02,  -1.455e-04],
       [  1.003e-02,  -1.306e-02,  -1.812e-01,   2.033e-01,  -9.171e-01,  -4.812e+00,  -5.255e-04,  -9.824e-03,  -1.013e-01,  -4.267e-02,   1.751e-04],
       [  4.548e-02,  -6.035e-02,  -1.682e+00,   2.277e+00,   6.604e-01,   2.161e+00,  -5.762e-03,  -3.906e-03,  -8.245e-03,  -1.503e-02,   1.729e-04],
       [  4.552e-01,  -7.698e-01,   1.519e+00,  -1.427e+00,   4.594e-01,   1.579e+00,   1.156e-02,  -2.799e-03,   3.249e-03,  -1.197e-02,   1.664e-04],
       [  9.599e-03,  -1.591e-02,  -5.855e-03,   2.012e-02,   1.825e-03,  -6.652e-02,  -5.039e+00,  -1.514e-01,   4.212e+00,   3.367e-01,  -3.382e-05],
       [  8.968e-03,   1.559e-02,  -3.520e-02,  -4.268e-02,  -1.825e-01,   6.745e-02,   1.356e-01,  -4.908e+00,   3.367e-01,  -4.366e+00,   6.097e-03],
       [  9.400e-03,  -1.551e-02,  -1.295e-02,   2.930e-02,   6.547e-04,  -7.986e-02,   4.197e+00,   1.352e-01,   5.059e+00,   3.663e-01,   3.062e-06],
       [  8.786e-03,   1.554e-02,  -2.185e-02,  -3.086e-02,  -3.281e-02,   2.604e-02,  -1.526e-01,   4.352e+00,   3.682e-01,  -4.929e+00,   1.587e-01],
       [ -9.069e-05,   9.252e-07,   4.431e-05,  -6.060e-05,  -2.149e-04,   2.737e-04,  -2.505e-03,   7.133e-02,   6.509e-03,  -8.740e-02,  -9.258e+00]])
    print "after SVD normalization with feature means ="
    print feature_means
    print "and whitening matrix ="
    print whitening_matrix
    norm_sample = get_svd_normalized_features(sample, feature_means, whitening_matrix)
    compute_svd_normalization(norm_sample, Ndiscard=0, max_rescale=999999)



if __name__ == '__main__':
    test_normalizations()


