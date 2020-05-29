from Element_PI import VariancePersist
from Element_PI import VariancePersistv1
import numpy as np

pixelsx=150
pixelsy=150
spread=.08
Max=2.5

samples = #len of samples

X=np.zeros((samples,pixelsx*pixelsy))

for i in range(1,samples):
    X[i,:]=VariancePersistv1('babel/{}.xyz'.format(i+1), pixelx=pixelsx, pixely=pixelsy, myspread=spread, myspecs={"maxBD": Max, "minBD":-.10}, showplot=False)