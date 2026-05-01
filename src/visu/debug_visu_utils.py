import matplotlib.pyplot as plt
import numpy as np
import trimesh


def show_mesh(sdf_mesh):
    if sdf_mesh is None:
        print("[WARN] show_mesh ricevuto None, salto la visualizzazione.")
        return
    if not isinstance(sdf_mesh, trimesh.Trimesh):
        print("[WARN] show_mesh richiede una trimesh.Trimesh, salto la visualizzazione.")
        return
    if sdf_mesh.vertices is None or len(sdf_mesh.vertices) == 0:
        print("[WARN] show_mesh ricevuto una mesh vuota, salto la visualizzazione.")
        return
    scene = trimesh.Scene()
    scene.add_geometry(sdf_mesh)
    scene.show()


def show_mesh_mtplt(mesh, title=""):
    vertices = mesh.vertices
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c="b", marker=".")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.scatter(0, 0, 0, c="g", marker="o")
    ax.quiver(0, 0, 0, 1, 0, 0, color="r")
    ax.quiver(0, 0, 0, 0, 1, 0, color="g")
    ax.quiver(0, 0, 0, 0, 0, 1, color="b")
    ax.set_title(title)
    plt.show()


def show_two_pc_mtplt(pc1, pc2, title=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c="b", marker=".")
    ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], c="r", marker=".")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.scatter(0, 0, 0, c="g", marker="o")
    ax.quiver(0, 0, 0, 1, 0, 0, color="r")
    ax.quiver(0, 0, 0, 0, 1, 0, color="g")
    ax.quiver(0, 0, 0, 0, 0, 1, color="b")
    ax.set_title(title)
    plt.show()


def show_mesh_and_pointcloud(mesh, pointcloud, title=""):
    import pyrender

    mesh = pyrender.Mesh.from_trimesh(mesh)
    red_color = np.array([1.0, 0.0, 0.0])
    colors = np.tile(red_color, (pointcloud.shape[0], 1))
    pointcloud = pyrender.Mesh.from_points(pointcloud, colors=colors)
    scene = pyrender.Scene()
    scene.add(mesh)
    scene.add(pointcloud)
    pyrender.Viewer(scene, use_raymond_lighting=True, point_size=5)


def show_pointcloud_matplot(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="b", marker=".")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
