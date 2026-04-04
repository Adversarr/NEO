from argparse import ArgumentParser
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = ArgumentParser(description="Visualize tetmesh with colormap.")
    parser.add_argument("mesh", type=str, help="Path to the tetmesh file")
    parser.add_argument("color", type=str, help="Path to the colormap file")
    parser.add_argument("component", type=int, help="Component index to visualize (0 for first component, 1 for second, etc.)")
    parser.add_argument("--show", action='store_true', help="Show the plot instead of saving it")

    args = parser.parse_args()

    mesh_path = Path(args.mesh)
    color_path = Path(args.color)
    component_index = args.component

    mesh = np.load(mesh_path)    # [nPoints, 3]
    colors = np.load(color_path) # [nPoints, C]

    colors = colors[:, component_index]  # Select the specified component

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], c=colors, cmap='viridis', s=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')
    if args.show:
        plt.show()
    else:
        plt.savefig('tetmesh_visualization.png', dpi=300, bbox_inches='tight')