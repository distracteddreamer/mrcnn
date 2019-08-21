import numpy as np

def generate_anchors(height, width, strides, scales, areas):
    centres_y = np.arange(0, height, strides) + strides/2
    centres_x = np.arange(0, width, strides) + strides/2
    areas, scales = np.meshgrid(areas, scales)
    heights = np.sqrt(areas / scales) # n_anchors x n_scales
    widths = np.sqrt(areas * scales)  # n_anchors x n_scales

    centres_x=/height
    centres_y=/width
    heights=/height
    widths=/width

    anchors = np.meshgrid(widths, heights, centres_x, centres_y, indexing='ij')
    return np.reshape(anchors, [4, -1])[::-1].T


