import numpy as np
import accelerator as A
from PIL import Image


class Camera:
    def __init__(self):
        # Focal point and length
        self.focal_point = np.array([0, 0, 0])
        self.focal_length = 1

        # Camera direction vectors
        self.right = np.array([1, 0, 0])
        self.up = np.array([0, 1, 0])
        self.forward = np.array([0, 0, 1])

        # Plane width and height
        self.W = 2
        self.H = 2


    def calculate_lighting(self, scene, lights, obj, normal, world_coords, t, direction):
        # Perform lighting calculations at given point
        intensity = 0
        for light in lights:
            ambient = light.ambient * obj.ambient
            diffuse = 0
            specular = 0

            # If facing towards the light, calculate diffuse and specular components (or shadow)
            if np.dot(normal, light.vector) > 0:
                intersection_point = self.focal_point + t * direction

                # See if any objects in the scene intersect the ray from intersection point to light
                in_shadow = False
                for shadow_obj in scene:
                    t_shadow, n_shadow = shadow_obj.intersect(intersection_point, light.vector)
                    # Prevent self-occlusion
                    if t_shadow > 0.0001:
                        in_shadow = True
                        break

                # Calculate diffuse / specular if light ray not blocked
                if not in_shadow:
                    diffuse = light.color * light.local * obj.diffuse * np.dot(normal, light.vector)
                    # Calculate terms for specular lighting and normalize
                    reflected = light.vector - 2 * normal * np.dot(light.vector, normal)
                    r = A.normalize(reflected)
                    viewer = self.focal_length + t * world_coords
                    v = A.normalize(viewer)
                    specular = light.local * obj.specular * np.dot(r, v) ** obj.n

            intensity += ambient + diffuse + specular
        return intensity


    def render(self, canvas_width, canvas_height, scene, lights, file_name='scene'):
        pixel_grid = np.zeros([canvas_height, canvas_width, 3])

        for j in range(canvas_height):
            print('Rendering row ', j + 1, ' of ', canvas_height)
            for i in range(canvas_width):

                # Map pixel indices to world space
                r = self.W * ((i + 0.5) / canvas_width - 0.5)
                b = self.H * ((j + 0.5) / canvas_height - 0.5)
                world_coords = self.focal_point + self.focal_length * self.forward + r * self.right - b * self.up
                # Normalized direction vector from focal point to world_coords
                direction = A.normalize(world_coords - self.focal_point)

                # Check intersections for each object in scene.
                # If object is closer to camera than previous objects (but not behind it), overwrite pixel
                t_smallest = np.inf
                for obj in scene:
                    t, normal = obj.intersect(self.focal_point, direction)
                    if 0 < t < t_smallest:
                        t_smallest = t
                        intensity = self.calculate_lighting(scene, lights, obj, normal, world_coords, t, direction)

                        n = 1
                        if obj.noise is not None:
                            n = obj.noise.marble(6, 4, abs(world_coords*3 + 350))

                        pixel_grid[j, i, :] = (intensity * obj.color * n).clip(min=0, max=1)

        # Convert color data to integer values and save to image file
        im = Image.fromarray(np.uint8(pixel_grid * 255))
        im.save(file_name + '.png')
