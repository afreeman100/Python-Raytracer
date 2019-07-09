import numpy as np


class Directional:
    def __init__(self, direction, ambient_intensity=0.4, local_intensity=0.6, color=None):
        """ Define directional light. Only required parameter is direction, all others will set to default values """
        self.direction = np.array(direction)

        # Normalized vector pointing towards light
        self.vector = (self.direction * -1) / np.linalg.norm(self.direction * -1)

        self.ambient = ambient_intensity
        self.local = local_intensity

        self.color = np.array([1, 1, 1] if color is None else np.array(color))
