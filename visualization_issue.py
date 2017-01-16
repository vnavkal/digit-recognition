import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import hog

X,Y = np.mgrid[-256:256,-256:256]
image = np.sqrt(X**2 + Y**2)
_, hog_image = hog(
    image,
    orientations=5,
    pixels_per_cell=(32, 32),
    cells_per_block=(1, 1),
    visualise=True
)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
ax1.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
ax2.imshow(hog_image, cmap=plt.cm.gray, interpolation='nearest')
# plt.savefig('comparison.png')

plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
# plt.savefig('original.png')
plt.close()

plt.imshow(hog_image, cmap=plt.cm.gray, interpolation='nearest')
# plt.savefig('hog_image.png')
plt.close()
