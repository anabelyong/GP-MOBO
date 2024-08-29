from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np

# New points a, b, c in R^3
points = {
    'a': [2, 2, 3],
    'b': [4, 4, 2],
    'c': [6, 6, 3]
}

# Define colors for points
colors = {
    'a': 'cyan',
    'b': 'purple',
    'c': 'white'
}

# Setting up the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the points
for point, coords in points.items():
    ax.scatter(*coords, label=f"Point {point} {tuple(coords)}", s=100, color=colors[point])

# Reference point r
r = [0, 0, 0]

# Function to draw hyperrectangles
def draw_hyperrectangle(ax, point, r, color, alpha=0.5):
    x = [r[0], point[0], point[0], r[0], r[0], r[0], point[0], point[0]]
    y = [r[1], r[1], point[1], point[1], r[1], point[1], point[1], r[1]]
    z = [r[2], r[2], r[2], r[2], point[2], point[2], point[2], point[2]]
    
    verts = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]], [x[3], y[3], z[3]],
             [x[4], y[4], z[4]], [x[5], y[5], z[5]], [x[6], y[6], z[6]], [x[7], y[7], z[7]]]
    
    faces = [[verts[j] for j in [0, 1, 2, 3]],
             [verts[j] for j in [4, 5, 6, 7]],
             [verts[j] for j in [0, 1, 5, 4]],
             [verts[j] for j in [2, 3, 7, 6]],
             [verts[j] for j in [1, 2, 6, 5]],
             [verts[j] for j in [4, 7, 3, 0]]]
    
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=1, edgecolors='black', alpha=alpha))

# Draw hyperrectangles for points c, b, and a in that order to avoid overlapping colors
draw_hyperrectangle(ax, points['c'], r, colors['c'], alpha=0.5)
draw_hyperrectangle(ax, points['b'], r, colors['b'], alpha=0.5)
draw_hyperrectangle(ax, points['a'], r, colors['a'], alpha=0.5)

# Labeling the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Adjusting the axis limits and centering the axes at the origin
ax.set_xlim(0, 7)
ax.set_ylim(0, 7)
ax.set_zlim(0, 7)

# Adding a legend
ax.legend()

# Set the viewing angle for better clarity
ax.view_init(elev=20, azim=135)

plt.show()
