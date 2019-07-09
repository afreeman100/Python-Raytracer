import numpy as np
import accelerator as A


class Mesh:
    def __init__(self, v, f_v, color=None, noise=None, diffuse_proportion=0.7, ambient_reflection=1,
                 specular_falloff=40):
        """
        Store arrays of vertices, normals, faces for objects loaded from obj file.
        Sets default color and lighting properties if not specified.
        """
        self.vertices = v
        self.face_vertices = f_v

        # Default color to blue if not specified
        self.color = np.array([0, 0, 1] if color is None else color)
        self.noise = noise

        self.ambient = ambient_reflection
        self.diffuse = self.color * diffuse_proportion
        self.specular = 1 - diffuse_proportion
        self.n = specular_falloff

        # Calculate normals for each face ahead of time
        self.normals = []
        for face in self.face_vertices:
            vector_1 = self.vertices[face[0]] - self.vertices[face[1]]
            vector_2 = self.vertices[face[2]] - self.vertices[face[1]]
            self.normals.append(A.normalize(A.cross(vector_2, vector_1)))

        # Calculate bounding sphere
        v_largest = self.vertices[1]
        v_smallest = self.vertices[1]

        for vertex in self.vertices:
            v_largest = np.maximum(v_largest, vertex)
            v_smallest = np.minimum(v_smallest, vertex)

        self.bounding_centre = (v_largest + v_smallest) / 2
        self.bounding_radius = np.sqrt(sum(np.square(self.bounding_centre - v_smallest)))


    def intersect(self, position, direction):
        """
        Given 2 points, see if vector defined by points intersects the mesh. This is done by
        checking intersection with each of the triangles comprising the mesh. Returns where on
        the line the intersection occurs, in terms of the t parameter, and the normal to the
        point of intersection.
        """

        # Check bounding sphere
        l = position - self.bounding_centre
        a = 1  # Dot product of unit vector with itself is 1. Same as doing direction.dot(direction)
        b = 2 * direction.dot(l)
        c = l.dot(l) - self.bounding_radius ** 2

        determinant = b * b - 4 * a * c
        if determinant < 0:
            return -1, None

        for face, normal in zip(self.face_vertices, self.normals):
            # For each face, get vertices that comprise it. Subtract 1 since arrays start at 0.
            vertex_1 = self.vertices[face[0]]
            vertex_2 = self.vertices[face[1]]
            vertex_3 = self.vertices[face[2]]

            # Parallel
            if normal.dot(direction) == 0:
                # Ray does not intersect current face, check next face
                continue
            # Line within plane
            if (vertex_1 - position).dot(normal) == 0:
                continue

            # Calculate parameter (t) and position (i) where line intersects plane
            t = (vertex_1 - position).dot(normal) / normal.dot(direction)
            i = position + t * direction

            # Behind camera
            if t < 0:
                continue

            # Forming vectors from triangle points
            v1 = vertex_2 - vertex_1
            v2 = vertex_3 - vertex_2
            v3 = vertex_1 - vertex_3

            # Cross product with intersection point, make sure all point inwards using dot product
            c1 = A.cross((i - vertex_1), v1)
            c2 = A.cross((i - vertex_2), v2)
            c3 = A.cross((i - vertex_3), v3)

            dot1 = c1.dot(c2)
            dot2 = c2.dot(c3)

            # Inside triangle - intersection!
            if dot1 > 0 and dot2 > 0:
                return t, normal
            # Outside triangle, check next
            else:
                continue

        # Ray has not intersected any face
        return -1, None


class Sphere:
    def __init__(self, centre, radius, color=None, noise=None, diffuse_proportion=0.8, ambient_reflectivity=1,
                 specular_falloff=40):
        """ Define sphere by centre and radius, with RGB color and lighting parameters """
        self.centre = centre
        self.radius = radius

        self.color = np.array([0, 0, 1] if color is None else color)
        self.noise = noise

        self.ambient = ambient_reflectivity
        self.diffuse = self.color * diffuse_proportion
        self.specular = np.array([1 - diffuse_proportion, 1 - diffuse_proportion, 1 - diffuse_proportion])
        self.n = specular_falloff


    def intersect(self, position, direction):
        """
        Given an initial point and a normalized direction vector, see if line defined by these intersects the sphere.
        Returns where on the line the intersection occurs, in terms of the t parameter, and the normal to the point
        of intersection.
        """
        l = position - self.centre
        a = 1  # Dot product of unit vector with itself is 1. Same as doing direction.dot(direction)
        b = 2 * direction.dot(l)
        c = l.dot(l) - self.radius ** 2

        determinant = b * b - 4 * a * c

        # If intersection / tangent exists, calculate where it occurs on the line in terms of t
        if determinant >= 0:
            t1 = (-b + np.sqrt(determinant)) / (2 * a)
            t2 = (-b - np.sqrt(determinant)) / (2 * a)

            # Smallest t is point that ray collides with first (assuming t1 and t2 are both positive)
            # If one of the points is behind the camera (i.e. negative), use the larger one
            t = np.min([t1, t2]) if (t1 > 0 and t2 > 0) else np.max([t1, t2])

            # Calculate vector from centre to intersection point and normalize
            normal = (position + t * direction) - self.centre
            return t, A.normalize(normal)

        # No intersection
        return -1, None
