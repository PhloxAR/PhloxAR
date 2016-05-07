#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PhloxAR.descriptor import harris
from PIL import Image
from matplotlib.pylab import gray
from numpy import array

img = array(Image.open('../data/empire.jpg').convert('L'))
gray()
harrisimg = harris.compute_harris_response(img)
filtered_coords = harris.get_harris_points(harrisimg, 6)
harris.plot_harris_points(img, filtered_coords)