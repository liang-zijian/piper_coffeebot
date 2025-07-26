import cv2
import numpy as np


image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
cv2.imshow("Test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
