import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Link lengths
l1 = 0.3
l2 = 0.25

# Structure Matrix A
A = np.array([[0.05, 0.01, 0.03],
              [0.01, 0.04, 0.02]])

# Muscle Stiffness Matrix Km
Km = np.array([[100, 0, 0],
               [0, 80, 0],
               [0, 0, 120]])

# Joint Space Stiffness Matrix Kq = A * Km * A^T
Kq = A @ Km @ A.T

# Configurations (q1, q2) in radians
configurations = {
    "Reaching Forward": (np.pi/6, -np.pi/6),
    "Reaching Up": (np.pi/2, 0),
    "Blocking Punch": (2*np.pi/3, -np.pi/2),
    "Throwing Punch": (np.pi/9, -np.pi/9),
    "Carrom Strike": (-np.pi/4, np.pi/2)
}

def calculate_matrices(q1, q2, l1, l2, Kq):
    """Calculates Jacobian, Velocity, Force, and Stiffness matrices for a given configuration."""
    # Jacobian Matrix J
    J = np.array([[-l1*np.sin(q1) - l2*np.sin(q1+q2), -l2*np.sin(q1+q2)],
                  [ l1*np.cos(q1) + l2*np.cos(q1+q2),  l2*np.cos(q1+q2)]])

    # Check for singularity
    if np.linalg.det(J) == 0:
        print(f"Warning: Singular configuration at q1={np.degrees(q1):.2f}, q2={np.degrees(q2):.2f}")
        return None, None, None, None

    # Velocity Manipulability Matrix (J*J^T)
    velocity_matrix = J @ J.T

    # Force Manipulability Matrix (J*J^T for plotting F.T * (J*J^T)^-1 * F = 1)
    force_matrix = J @ J.T

    # Task Space Stiffness Matrix Kx = (J^T)^-1 * Kq * J^-1
    J_inv = np.linalg.inv(J)
    J_T_inv = np.linalg.inv(J.T)
    stiffness_matrix_Kx = J_T_inv @ Kq @ J_inv

    return velocity_matrix, force_matrix, stiffness_matrix_Kx, J

# Function to plot an ellipse from a 2x2 matrix M representing x.T * M_plot * x = 1
def plot_ellipse(ax, matrix_for_calc, origin, color, label):
    """Plots an ellipse based on a 2x2 matrix."""

    if label == "Velocity":
        # Velocity ellipse: v.T * (J*J^T)^-1 * v = 1. Plot using J*J^T. Semi-axes = sqrt(eigenvalues(J*J^T)).
        plot_matrix_for_eig = matrix_for_calc # J*J^T
    elif label == "Force":
        # Force ellipse: F.T * (J*J^T)^-1 * F = 1. Plot using (J*J^T)^-1. Semi-axes = sqrt(eigenvalues((J*J^T)^-1)).
        # The input matrix_for_calc is J*J^T, so we need its inverse for eigenvalue calculation.
        if np.linalg.det(matrix_for_calc) == 0:
             print(f"Warning: Cannot plot Force ellipse for singular J*J^T.")
             return
        plot_matrix_for_eig = np.linalg.inv(matrix_for_calc) # (J*J^T)^-1
    elif label == "Stiffness":
        # Stiffness (Compliance) ellipse: dx.T * Kx * dx = constant or dx.T * Kx_inv * dx = 1. Plot using Kx_inv. Semi-axes = sqrt(eigenvalues(Kx_inv)).
        # The input matrix_for_calc is Kx, so we need its inverse for eigenvalue calculation.
        if np.linalg.det(matrix_for_calc) == 0:
             print(f"Warning: Cannot plot Stiffness ellipse for singular Kx.")
             return
        plot_matrix_for_eig = np.linalg.inv(matrix_for_calc) # Kx_inv


    try:
        eigenvalues, eigenvectors = np.linalg.eig(plot_matrix_for_eig)
    except np.linalg.LinAlgError:
        print(f"Warning: Could not calculate eigenvalues for {label} ellipse.")
        return

    # Ensure eigenvalues are positive for ellipse
    if np.any(eigenvalues < 0):
        print(f"Warning: Negative eigenvalues encountered for {label} ellipse.")
        return

    # Sort eigenvalues and corresponding eigenvectors
    idx = eigenvalues.argsort()[::-1] # Sort descending
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Semi-axis lengths are sqrt of eigenvalues for the matrix used in the x.T * M * x = 1 form
    semi_axis_lengths = np.sqrt(eigenvalues)

    # Angle of the first principal axis (in degrees)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # Create and add the ellipse patch
    ellipse = Ellipse(xy=origin, width=2*semi_axis_lengths[0], height=2*semi_axis_lengths[1],
                      angle=angle, edgecolor=color, fc='None', lw=2, label=label)
    ax.add_patch(ellipse)


# Calculate matrices for each configuration
results = {}
for name, (q1, q2) in configurations.items():
    print(f"Calculating for configuration: {name}")
    vel_mat, force_mat, stiff_mat, J_matrix = calculate_matrices(q1, q2, l1, l2, Kq)
    if vel_mat is not None:
        results[name] = {
            "q1": q1,
            "q2": q2,
            "J": J_matrix,
            "Velocity_Matrix": vel_mat, # J*J^T
            "Force_Matrix": force_mat, # J*J^T
            "Stiffness_Matrix": stiff_mat # Kx
        }
    else:
        results[name] = None

# Plotting
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 18))
axes = axes.flatten()
fig.tight_layout(pad=5.0)

row = 0
col = 0

for name, data in results.items():
    ax = axes[row*2 + col]

    if data is None:
        ax.set_title(f"{name}\n(Singular Configuration)")
        ax.text(0.5, 0.5, "Singular", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=14)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)

    else:
        q1, q2 = data["q1"], data["q2"]
        velocity_matrix = data["Velocity_Matrix"]
        force_matrix = data["Force_Matrix"]
        stiffness_matrix = data["Stiffness_Matrix"]

        ax.set_title(f"{name}\n(q1={np.degrees(q1):.1f}°, q2={np.degrees(q2):.1f}°)")
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True)

        # End-effector position for the origin of the ellipse
        x = l1 * np.cos(q1) + l2 * np.cos(q1 + q2)
        y = l1 * np.sin(q1) + l2 * np.sin(q1 + q2)
        origin = (x, y)
        ax.plot(x, y, 'ko') # Plot the end-effector position

        # Plot Velocity Manipulability Ellipse (based on J*J^T)
        plot_ellipse(ax, velocity_matrix, origin, 'b', 'Velocity')

        # Plot Force Manipulability Ellipse (based on (J*J^T)^-1)
        plot_ellipse(ax, force_matrix, origin, 'r', 'Force')

        # Plot Stiffness Ellipse (Compliance Ellipse based on Kx_inv)
        plot_ellipse(ax, stiffness_matrix, origin, 'g', 'Stiffness')

        ax.legend()

        # Auto-set plot limits based on ellipse sizes and origin
        all_ellipse_points = []
        for matrix, color, label in [(velocity_matrix, 'b', 'Velocity'),
                                      (force_matrix, 'r', 'Force'),
                                      (stiffness_matrix, 'g', 'Stiffness')]:
            try:
                if label == "Velocity":
                     plot_mat = matrix
                elif label == "Force":
                     plot_mat = np.linalg.inv(matrix)
                elif label == "Stiffness":
                     plot_mat = np.linalg.inv(matrix)

                if np.linalg.det(plot_mat) != 0 and np.all(np.linalg.eigvals(plot_mat) > 0):
                    eigenvalues, eigenvectors = np.linalg.eig(plot_mat)
                    semi_axes = np.sqrt(eigenvalues)
                    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

                    # Generate points on the unit circle and transform them to ellipse points
                    theta = np.linspace(0, 2*np.pi, 100)
                    unit_circle = np.array([np.cos(theta), np.sin(theta)])
                    scale_matrix = np.diag(semi_axes)
                    rotation_matrix = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                                                [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])
                    ellipse_points = rotation_matrix @ scale_matrix @ unit_circle + np.array(origin).reshape(-1, 1)
                    all_ellipse_points.append(ellipse_points)
            except (np.linalg.LinAlgError, ValueError):
                pass # Handle cases where ellipse cannot be plotted

        if all_ellipse_points:
            all_points = np.concatenate(all_ellipse_points, axis=1)
            min_x, min_y = np.min(all_points, axis=1)
            max_x, max_y = np.max(all_points, axis=1)
            padding_x = (max_x - min_x) * 0.2
            padding_y = (max_y - min_y) * 0.2
            ax.set_xlim(min_x - padding_x, max_x + padding_x)
            ax.set_ylim(min_y - padding_y, max_y + padding_y)
        else:
             # Default limits if no ellipses were plotted successfully
             ax.set_xlim(origin[0] - 0.5, origin[0] + 0.5)
             ax.set_ylim(origin[1] - 0.5, origin[1] + 0.5)


    col += 1
    if col > 1:
        col = 0
        row += 1

# Hide any unused subplots
for i in range(row*2 + col, len(axes)):
    fig.delaxes(axes[i])

plt.show()