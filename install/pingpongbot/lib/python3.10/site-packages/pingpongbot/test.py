import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def calculate_v0_with_debug(p0, pfinal, g, weights):
    """
    Calculate the optimal initial velocity vector (v0) and time of flight (T) with debugging and weighting.
    Args:
    - p0 (array): Initial position [x, y, z].
    - pfinal (array): Final position [x, y, z].
    - g (array): Gravity vector [gx, gy, gz].
    - weights (array): Weights for [x, y, z] components.

    Returns:
    - T_opt (float): Optimal time of flight.
    - v0_opt (array): Optimal initial velocity vector.
    """
    def v0_norm_debug(T):
        """
        Objective function to minimize weighted ||v0||.
        Logs intermediate results for debugging.
        """
        if T <= 0:
            return np.inf  # Avoid division by zero
        # Compute v0 for given T
        v0 = (pfinal - p0 - 0.5 * g * (T**2)) / T
        # Weighted norm
        weighted_norm = np.sqrt(weights[0] * v0[0]**2 + weights[1] * v0[1]**2 + weights[2] * v0[2]**2)
        return weighted_norm

    # Initial guess for T
    T_guess = 1.0

    # Minimize ||v0|| with respect to T
    res = minimize(v0_norm_debug, x0=[T_guess], bounds=[(1e-3, None)])

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    # Optimal time and velocity
    T_opt = res.x[0]
    v0_opt = (pfinal - p0 - 0.5 * g * (T_opt**2)) / T_opt

    # Validate final position using kinematic equation
    p_computed = p0 + v0_opt * T_opt + 0.5 * g * (T_opt**2)
    print("\n=== Results ===")
    print(f"Optimal Time of Flight (T): {T_opt}")
    print(f"Optimal Initial Velocity (v0): {v0_opt}")
    print(f"Minimum Speed ||v0||: {np.linalg.norm(v0_opt)}")
    print(f"Computed Final Position: {p_computed}")
    print(f"Target Final Position: {pfinal}")
    print(f"Position Error: {p_computed - pfinal}")

    return T_opt, v0_opt

def plot_trajectory(p0, v0, g, T_opt, pfinal):
    """
    Plot the trajectory of the ball.
    Args:
    - p0 (array): Initial position [x, y, z].
    - v0 (array): Initial velocity [vx, vy, vz].
    - g (array): Gravity vector [gx, gy, gz].
    - T_opt (float): Optimal time of flight.
    - pfinal (array): Target position [x, y, z].
    """
    # Time points for the trajectory
    t_vals = np.linspace(0, T_opt, num=100)

    # Calculate trajectory
    x_vals = p0[0] + v0[0] * t_vals + 0.5 * g[0] * t_vals**2
    y_vals = p0[1] + v0[1] * t_vals + 0.5 * g[1] * t_vals**2
    z_vals = p0[2] + v0[2] * t_vals + 0.5 * g[2] * t_vals**2

    # Plot 3D trajectory
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_vals, y_vals, z_vals, label='Trajectory', color='blue')
    ax.scatter(p0[0], p0[1], p0[2], color='green', label='Initial Position')
    ax.scatter(pfinal[0], pfinal[1], pfinal[2], color='red', label='Final Position')
    ax.set_title('Trajectory of the Ball')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.legend()
    plt.show()

def test_calculate_v0():
    # Initial and target positions
    p0 = np.array([0.5, 0.5, 0.5])  # Initial ball position
    pfinal = np.array([2.0, 1.0, 0.0])  # Target bowl position
    g = np.array([0.0, 0.0, -9.81])  # Gravity vector

    # Weights for optimization
    weights = np.array([1.0, 1.0, 1.0])  # Equal weight for all components

    # Run the calculation
    T_opt, v0_opt = calculate_v0_with_debug(p0, pfinal, g, weights)

    # Plot the trajectory
    plot_trajectory(p0, v0_opt, g, T_opt, pfinal)

    # Print final results
    print("\nFinal Test Results:")
    print(f"Optimal Time of Flight (T): {T_opt}")
    print(f"Optimal Initial Velocity (v0): {v0_opt}")
    print(f"Minimum Initial Speed ||v0||: {np.linalg.norm(v0_opt)}")

if __name__ == "__main__":
    test_calculate_v0()
