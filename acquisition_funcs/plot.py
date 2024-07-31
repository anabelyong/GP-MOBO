import matplotlib.pyplot as plt
import numpy as np

# Given points
points = np.array([[8.5, 3.0], [8.5, 3.5], [5.0, 5.0], [9.0, 1.0], [4.0, 5.0]])
reference_point = [0.0, 0.0]

# Sort the points by the first coordinate, and by the second coordinate if the first coordinates are the same
sorted_points = sorted(points, key=lambda x: (x[0], x[1]), reverse=True)

# Plotting the points
plt.figure(figsize=(10, 6))
for point in sorted_points:
    plt.scatter(point[0], point[1], color="blue")
    plt.text(point[0], point[1], f"({point[0]}, {point[1]})", fontsize=12, ha="right")

# Adding the reference point
plt.scatter(reference_point[0], reference_point[1], color="red")
plt.text(
    reference_point[0], reference_point[1], f"({reference_point[0]}, {reference_point[1]})", fontsize=12, ha="right"
)

# Drawing the rectangles
total_area = 0
previous_point = reference_point

for point in sorted_points:
    width = point[0] - reference_point[0]
    height = point[1] - previous_point[1]

    if width > 0 and height > 0:
        area = width * height
        total_area += area

        # Drawing the rectangle
        plt.gca().add_patch(
            plt.Rectangle((reference_point[0], previous_point[1]), width, height, edgecolor="black", facecolor="none")
        )

        previous_point = point

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Manual Calculation of Hypervolume")
plt.grid(True)
plt.xlim(-1, 10)
plt.ylim(-1, 6)

# Showing the plot
plt.show()
# Printing the calculated hypervolume
print(f"Computed Hypervolume: {total_area}")
