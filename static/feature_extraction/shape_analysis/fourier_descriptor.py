import numpy as np
import sklearn
from sklearn.metrics import pairwise_distances

# Get FourierDesc 
def get_FD(shape_signature):
  N = 128        # Number of samplepoints
  Fs = 300
  T = 1.0 / Fs      
  N_fft = 128   
  y = shape_signature[:,1] # this is the signal
  x = shape_signature[:,0] # along angle

  # Compute the fft.
  yf = np.fft.fft(y,n=N_fft)

  return yf # this is the fourier signal



# Make fourier descriptor invariant to scale
def scaleInvariant(FourierDesc):
  """Input: FourierDescriptor that has been normalised for translation invariance
  Divide every fourier coefficient with the zero-frequency coefficient (first coefficient)
  The scale is represented in the first coefficient
  Returns a scale invariant Fourier Descriptor
  """
  firstVal = FourierDesc[0] 

  for index, value in enumerate(FourierDesc):
      FourierDesc[index] = value / firstVal

  return FourierDesc

# Make fourier descriptor invariant to rotaition and start point
def rotationInvariant(FourierDesc):
  """
  Input: Fourier Descriptor
  Only keep the magnitude of the Fourier Coefficient 
  """
  for index, value in enumerate(FourierDesc):
        FourierDesc[index] = np.absolute(value)

  return FourierDesc

def final_FD(edge_coordinates):
  FD = get_FD(edge_coordinates)
  FD = scaleInvariant(FD)
  FD = rotationInvariant(FD)
  return FD

def GetSpacedElements(shape_signature, numElems):
  # get 128 equally spaced out elements from the shape signature
    out = shape_signature[np.round(np.linspace(0, len(shape_signature)-1, numElems)).astype(int)]
    return out
