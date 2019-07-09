import obj_reader
from scene_objects import Sphere
from lighting import Directional
from camera import Camera
from perlin import Perlin3D

scene = []
lights = []

bunny = obj_reader.read_obj_file('obj/bunny.obj', [0.7, 0.7, 0.65])
bunny.noise = Perlin3D()
scene.append(bunny)

cube = obj_reader.read_obj_file('obj/cube.obj', [0.62, 0.65, 0.6])
cube.noise = Perlin3D(seed=5)
scene.append(cube)

scene.append(Sphere(centre=[-4, -2.5, 7], radius=0.5, color=[0.8, 0.8, 0.75], noise=Perlin3D(seed=6)))
scene.append(Sphere(centre=[-4, -1, 7], radius=1, color=[0.85, 0.85, 0.85], noise=Perlin3D(seed=10)))
scene.append(Sphere(centre=[-4, 0.5, 7], radius=0.5, color=[0.7, 0.73, 0.7], noise=Perlin3D(seed=8)))
# scene.append(Sphere(centre=[0, -2, 7], radius=2.5, color=[0.7, 0.73, 0.7], noise=Perlin3D(seed=11)))

scene.append(obj_reader.read_obj_file('obj/floor.obj', [0.8, 0.8, 0.8]))

light = Directional(direction=[1, -1, 0.3], ambient_intensity=0.5, local_intensity=0.8)
lights.append(light)

camera = Camera()
camera.render(canvas_width=2000, canvas_height=2000, scene=scene, lights=lights, file_name='scene5')
