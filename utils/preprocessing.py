import cv2
import numpy as np

def image_normalization(src, norm_type, norm_rng=None):
  if norm_type == "minmax":
    if norm_rng:
      alpha, beta = norm_rng
      src[src < alpha] = alpha
      src[src > beta] = beta
      return 255.0 * (src.astype(np.float32) - alpha) / (beta - alpha)
    else:
      minval, maxval, _, _ = cv2.minMaxLoc(src)
      return 255.0 * (src.astype(np.float32) - minval) / (maxval - minval)
  else:
    raise NotImplementedError