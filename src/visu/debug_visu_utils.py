import numpy as np
import matplotlib.pyplot as plt
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


def show_sdf_pointcloud(points, sdf_points, filter_interval=0.01):
    import pyrender
    
    colors = np.zeros(points.shape)
    colors[sdf_points < -filter_interval, 2] = 1
    colors[sdf_points > filter_interval, 0] = 1

    cloud = pyrender.Mesh.from_points(points, colors=colors)

    scene = pyrender.Scene()
    scene.add(cloud)

    # Visualizza la scena
    pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)



def relate_meshes(mesh1, mesh2, title=''):
    ver1 = mesh1.vertices
    ver2 = mesh2.vertices

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(ver1[:, 0], ver1[:, 1], ver1[:, 2], c='b', marker='.')
    ax.scatter(ver2[:, 0], ver2[:, 1], ver2[:, 2], c='r', marker='.')
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #set legend
    ax.legend(['mesh1', 'mesh2'])
    #display origin
    ax.scatter(0,0,0, c='g', marker='o')
    #show origin axes
    ax.quiver(0,0,0,1,0,0, color='r')
    ax.quiver(0,0,0,0,1,0, color='g')
    ax.quiver(0,0,0,0,0,1, color='b')
    #add title
    ax.set_title(title)
    plt.show()



def show_mesh_unit_spere(mesh):
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    sphere.visual.vertex_colors = [0.0, 0.0, 1.0, 0.1]
    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    scene.add_geometry(sphere)
    scene.show()

#matplot
def show_mesh_mtplt(mesh, title=''): #pc stands for pointcloud
    vertices = mesh.vertices
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='.')
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #display origin
    ax.scatter(0,0,0, c='g', marker='o')
    #show origin axes
    ax.quiver(0,0,0,1,0,0, color='r')
    ax.quiver(0,0,0,0,1,0, color='g')
    ax.quiver(0,0,0,0,0,1, color='b')
    #add title
    ax.set_title(title)
    plt.show()

def show_two_pc_mtplt(pc1, pc2, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc1[:, 0], pc1[:, 1], pc1[:, 2], c='b', marker='.')
    ax.scatter(pc2[:, 0], pc2[:, 1], pc2[:, 2], c='r', marker='.')
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #display origin
    ax.scatter(0,0,0, c='g', marker='o')
    #show origin axes
    ax.quiver(0,0,0,1,0,0, color='r')
    ax.quiver(0,0,0,0,1,0, color='g')
    ax.quiver(0,0,0,0,0,1, color='b')
    #add title
    ax.set_title(title)
    plt.show()

def show_mesh_with_pyrender(mesh):
    import pyrender
    pyrender_mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(pyrender_mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True)


def show_sdf_pointcloud_with_mesh(points, mesh, sdf_points):
    import pyrender
    colors = np.zeros(points.shape)
    colors[sdf_points < 0, 2] = 1
    colors[sdf_points > 0, 0] = 1
    cloud = pyrender.Mesh.from_points(points, colors=colors)
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene = pyrender.Scene()
    scene.add(cloud)
    scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)

def show_mesh_and_pointcloud(mesh, pointcloud, title=''):
    import pyrender
    mesh = pyrender.Mesh.from_trimesh(mesh)
    # Imposta i punti della pointcloud in rosso
    red_color = np.array([1.0, 0.0, 0.0])  # RGB per rosso
    colors = np.tile(red_color, (pointcloud.shape[0], 1))  # Replica il colore per ogni punto
    # Crea il mesh della pointcloud con colori
    pointcloud = pyrender.Mesh.from_points(pointcloud, colors=colors)
    scene = pyrender.Scene()
    scene.add(mesh)
    scene.add(pointcloud)
    pyrender.Viewer(scene, use_raymond_lighting=True, point_size=5)

def show_pointcloud_matplot(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='.')
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()





# def plot_mvie_ellipsoid(points, A, c):
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # Plot original points
#     ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.2, color='gray')

#     # Plot ellipsoid
#     U, s, _ = np.linalg.svd(A)
#     radii = 1.0 / np.sqrt(s)
#     ellipsoid_transform = U @ np.diag(radii)

#     u = np.linspace(0, 2 * np.pi, 50)
#     v = np.linspace(0, np.pi, 25)
#     x = np.outer(np.cos(u), np.sin(v))
#     y = np.outer(np.sin(u), np.sin(v))
#     z = np.outer(np.ones_like(u), np.cos(v))
#     sphere = np.stack((x, y, z), axis=-1)
#     ellipsoid = sphere @ ellipsoid_transform.T + c

#     # Actual surface
#     ax.plot_surface(
#         ellipsoid[:, :, 0], ellipsoid[:, :, 1], ellipsoid[:, :, 2],
#         rstride=2, cstride=2, color='red', alpha=0.4
#     )

#     # Wireframe reference unit sphere (optional)
#     us = np.linspace(0, 2 * np.pi, 30)
#     vs = np.linspace(0, np.pi, 15)
#     xs = np.outer(np.cos(us), np.sin(vs))
#     ys = np.outer(np.sin(us), np.sin(vs))
#     zs = np.outer(np.ones_like(us), np.cos(vs))
#     ax.plot_wireframe(xs, ys, zs, color='blue', alpha=0.2, linewidth=0.5)

#     # Manual legend
#     legend_elements = [
#         Line2D([0], [0], marker='o', color='w', label='Punti',
#                markerfacecolor='gray', markersize=8, alpha=0.5),
#         Patch(facecolor='red', edgecolor='r', label='MVIE', alpha=0.4)
#     ]
#     ax.legend(handles=legend_elements)

#     ax.set_title("MVIE - Ellissoide di massimo volume dentro il convesso")
#     ax.set_box_aspect([1, 1, 1])
#     plt.tight_layout()
#     plt.show()

def plot_ellipsoid(center, axes_lengths, eigenvectors, points=None):
    """
    Plotta un elissoide definito da centro, assi e autovettori.
    Se forniti, mostra anche i punti originali.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Mostra i punti originali, se presenti
    if points is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

    # Parametri sferici
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = axes_lengths[0] / 2 * np.outer(np.cos(u), np.sin(v))
    y = axes_lengths[1] / 2 * np.outer(np.sin(u), np.sin(v))
    z = axes_lengths[2] / 2 * np.outer(np.ones_like(u), np.cos(v))

    # Rotazione e traslazione
    ellipsoid = np.dot(np.stack([x.flatten(), y.flatten(), z.flatten()]).T, eigenvectors.T)
    x_rot = ellipsoid[:, 0].reshape(x.shape) + center[0]
    y_rot = ellipsoid[:, 1].reshape(y.shape) + center[1]
    z_rot = ellipsoid[:, 2].reshape(z.shape) + center[2]

    # Plotta la superficie dell’elissoide
    ax.plot_surface(x_rot, y_rot, z_rot, rstride=4, cstride=4, color='c', alpha=0.3)

    plt.show()
