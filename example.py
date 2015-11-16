import os, sys, warnings
import scipy.io
import numpy as np
import pyrosl
from hyperspy import signals
import hyperspy.hspy as hs

print '---'

GAAS_35_SOURCE =\
                {
                'directory': '../../../source_data/gaas_images/precessed35mrad',
                'filetype' : 'tif',
                'shape' : (100, 30, 144, 144),
                }
                
def load_original_data(directory='.', filetype='bmp', shape=(0,0,0,0), norm=False):
    nm = hs.load('{}/*.{}'.format(directory, filetype), stack=True)
    nm.change_dtype(float)
    nm = signals.Image(nm.data.reshape(shape))
    return nm

def matrixify(image_data, n_components=None):
    d = image_data    
    if len(d.shape)==4:
        return d.reshape(d.shape[0]*d.shape[1],d.shape[2]*d.shape[3])
    elif len(d.shape)==3:
        return d.reshape(d.shape[0],d.shape[1]*d.shape[2])
    else:
        raise NotImplementedError("That doesn't look like a standard image.")
        
# Load the data
D = matrixify(load_original_data(**GAAS_35_SOURCE).data)
D = D.astype(np.double, copy = False)

# Run the sub-sampled version
ss_rosl = pyrosl.ROSL( 
    method='subsample',
    sampling = (36*36,500),
    rank = 3,
    reg = 0.02,
    tol = 1E-5,
    iters = 50,
    verbose = True
)

#ss_rosl.fit_transform(D)
print '---'

# Run the full ROSL algorithm
full_rosl = pyrosl.ROSL(
    method = 'full',
    rank = 3,
    reg = 0.015,
   )
print 'Starting ROSL'
full_rosl.fit_transform(D)
print 'Finished ROSL'
print '---'

# Output some numbers
error1 = np.linalg.norm(D - ss_rosl.model_ - ss_rosl.residuals_, 'fro') / np.linalg.norm(D, 'fro')
error2 = np.linalg.norm(D - full_rosl.model_ - full_rosl.residuals_, 'fro') / np.linalg.norm(D, 'fro')
error3 = np.linalg.norm(ss_rosl.model_ - full_rosl.model_, 'fro') / np.linalg.norm(full_rosl.model_, 'fro')
print 'Subsampled ROSL+ error: %.3f' % error1
print 'Full ROSL error:        %.3f' % error2
print 'ROSL/ROSL+ comparison:  %.3f' % error3
print '---'
