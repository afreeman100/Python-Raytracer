import numpy as np
import accelerator as A


class Perlin2D:
    def __init__(self, seed=None):
        # If given, use the specified seed to generate the pseudo-random gradients
        if seed is not None:
            np.random.seed(seed)

        self.grid_size = 100
        self.grid = np.zeros([self.grid_size, self.grid_size, 2])

        print('Creating Perlin noise generator')
        # For each grid node, generate random gradient vector
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.grid[i, j, :] = A.normalize_(np.random.uniform(-1, 1, 2))


    def value_at(self, point):
        point = point % (self.grid_size - 1)

        # Vertices defining square around point
        x0, y0 = np.floor(point)
        vertices = np.array([[x0, y0],
                             [x0 + 1, y0],
                             [x0, y0 + 1],
                             [x0 + 1, y0 + 1]], dtype=int)

        # Get the randomly generated gradients for each of these vertices
        gradients = np.array([self.grid[i, j, :] for i, j in vertices])

        # For each vertex, calculate vector from vertex to centre
        # Then take dot products with gradient vector
        dots = np.zeros(len(vertices))
        for i in range(len(vertices)):
            dots[i] = np.dot(gradients[i, :], (point - vertices[i, :]))


        def smoothing_function(p):
            return 6 * p ** 5 - 15 * p ** 4 + 10 * p ** 3


        # Determine position of point in unit cube around it, then calculate weights of surrounding vertices
        point = point - vertices[0]
        weights = np.apply_along_axis(smoothing_function, axis=-0, arr=point)

        # Interpolation across x axis, using weights calculated for x coordinate
        x_1 = dots[0] + weights[0] * (dots[1] - dots[0])
        x_2 = dots[2] + weights[0] * (dots[3] - dots[2])

        # Interpolate across y
        y_1 = x_1 + weights[1] * (x_2 - x_1)

        return y_1


    def turbulence(self, o, point):
        o = 1 / 2 ** o
        t = 0
        scale = 1
        while scale > o:
            t += abs(self.value_at(point / scale) * scale)
            scale /= 2
        return t


    def marble(self, t, f, point):
        x = np.sin((point[0] + f * self.turbulence(t, point)))
        x = np.sqrt(x + 1)
        return x


class Perlin3D:
    def __init__(self, seed=None):
        # If given, use the specified seed to generate the pseudo-random gradients
        if seed is not None:
            np.random.seed(seed)

        self.grid_size = 100
        self.grid = np.zeros([self.grid_size, self.grid_size, self.grid_size, 3])

        print('Creating Perlin noise generator')
        # For each grid node, generate random gradient vector
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                for k in range(self.grid_size):
                    self.grid[i, j, k, :] = A.normalize_(np.random.uniform(-1, 1, 3))


    def value_at(self, point):
        point = point % (self.grid_size - 1)

        # Vertices defining square around point
        x0, y0, z0 = np.floor(point)
        vertices = np.array([[x0, y0, z0],
                             [x0 + 1, y0, z0],
                             [x0, y0 + 1, z0],
                             [x0 + 1, y0 + 1, z0],
                             [x0, y0, z0 + 1],
                             [x0 + 1, y0, z0 + 1],
                             [x0, y0 + 1, z0 + 1],
                             [x0 + 1, y0 + 1, z0 + 1]], dtype=int)

        # Get the randomly generated gradients for each of these vertices
        gradients = np.array([self.grid[i, j, k, :] for i, j, k in vertices])

        # For each vertex, calculate vector from vertex to centre
        # Then take dot products with gradient vector
        dots = np.zeros(len(vertices))
        for i in range(len(vertices)):
            dots[i] = np.dot(gradients[i, :], (point - vertices[i, :]))


        def smoothing_function(p):
            return 6 * p ** 5 - 15 * p ** 4 + 10 * p ** 3


        # Determine position of point in unit cube around it, then calculate weights of surrounding vertices
        point = point - vertices[0]
        weights = np.apply_along_axis(smoothing_function, axis=-0, arr=point)

        # Interpolation across x axis, using weights calculated for x coordinate
        x_1 = dots[0] + weights[0] * (dots[1] - dots[0])
        x_2 = dots[2] + weights[0] * (dots[3] - dots[2])

        x_3 = dots[4] + weights[0] * (dots[5] - dots[4])
        x_4 = dots[6] + weights[0] * (dots[7] - dots[6])

        # Interpolate across y
        y_1 = x_1 + weights[1] * (x_2 - x_1)
        y_2 = x_3 + weights[1] * (x_4 - x_3)

        # Interpolate across z
        z_1 = y_1 + weights[2] * (y_2 - y_1)

        return z_1


    def turbulence(self, o, point):
        o = 1 / 2 ** o
        t = 0
        scale = 1
        while scale > o:
            t += abs(self.value_at(point / scale) * scale)
            scale /= 2
        return t


    def marble(self, t, f, point):
        x = np.sin((point[0] + f * self.turbulence(t, point)))
        x = np.sqrt(x + 1)
        return x
