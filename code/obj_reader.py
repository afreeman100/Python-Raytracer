import numpy as np
from scene_objects import Mesh


def read_obj_file(file_name, color):
    vertices = []
    face_vertices = []

    file = open(file_name, 'r')
    for l in file:
        line = l.split()

        # If line is empty or comment, skip
        if len(line) == 0 or line[0] == '#':
            continue

        # Convert vertices to floats
        if line[0] == 'v':
            v = [float(i) for i in line[1:]]
            vertices.append(v)
            continue

        # Convert face indices to integers, subtracting 1 so they will properly index elements of the vertices array
        if line[0] == 'f':
            f_v = [int(i)-1 for i in line[1:]]
            face_vertices.append(f_v)
            continue

    # Convert lists to 2D numpy arrays which can have matrix operations performed on them later
    return Mesh(np.array(vertices), np.array(face_vertices), np.array(color))
