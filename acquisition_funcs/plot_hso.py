import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# Define the points with their coordinates
points = {
    'a': (11, 4, 4),
    'b': (9, 2, 5),
    'c': (5, 6, 7),
    'd': (3, 3, 10)
}

# Define the slicing planes in the x-direction
x_slices = [3, 5, 9, 11]

# Create a figure with subplots for each slice
fig, axs = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={'projection': '3d'})

# Function to create a 3D rectangle (cuboid)
def cuboid_data(o, size=(1,1,1)):
    X = [[0,1,1,0,0,0], [1,1,1,1,1,1], [0,0,0,0,0,1], [0,1,1,0,0,0]]
    Y = [[0,0,1,1,0,0], [0,0,1,1,0,0], [1,1,1,1,1,0], [1,1,1,1,0,0]]
    Z = [[0,0,0,0,0,0], [0,1,1,1,1,1], [0,1,1,0,0,0], [0,0,0,1,1,1]]
    X = np.array(X).T * size[0] + o[0]
    Y = np.array(Y).T * size[1] + o[1]
    Z = np.array(Z).T * size[2] + o[2]
    return X,Y,Z

# Function to plot a cuboid
def plot_cuboid(ax, pos, size, color='blue', alpha=0.1):
    X, Y, Z = cuboid_data(pos, size)
    ax.plot_surface(X, Y, Z, color=color, rstride=1, cstride=1, alpha=alpha, shade=True)
    ax.add_collection3d(Poly3DCollection([list(zip(X.flatten(), Y.flatten(), Z.flatten()))], 
                                          facecolors=color, linewidths=1, edgecolors='r', alpha=alpha))

# Iterate over slices and plot them individually
colors = ['orange', 'blue', 'green', 'purple']
for i, (x, ax) in enumerate(zip(x_slices, axs)):
    ax.set_title(f'Slice {i+1}')
    # Plot the current slice
    plot_cuboid(ax, (x_slices[i], 0, 0), (x_slices[i+1] - x_slices[i] if i+1 < len(x_slices) else 12 - x_slices[i], 12, 12), colors[i])
    
    # Plot the points with their coordinates in the legend
    for label, (px, py, pz) in points.items():
        if px >= x and (i+1 == len(x_slices) or px < x_slices[i+1]):
            ax.scatter(px, py, pz, label=f'Point {label} ({px}, {py}, {pz})', s=100)
            ax.text(px, py, pz, label, fontsize=12, ha='right')

    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([0, 12])
    ax.set_ylim([0, 12])
    ax.set_zlim([0, 12])

    ax.legend()

# Show the plots
plt.tight_layout()
plt.show()
