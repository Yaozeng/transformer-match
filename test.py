import xlrd
import pandas as pd
import numpy as np
a=np.array([1])
print(a.shape)
a=a[np.newaxis,:]
print(a.shape)
print(a.squeeze().shape)
print(a.squeeze(0).shape)