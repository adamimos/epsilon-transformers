# type: ignore
# %%
from epsilon_transformers.comp_mech.processes import (
    mess3,
    nond,
    even_process,
    zero_one_random,
    golden_mean,
    random_random_xor,
)

from epsilon_transformers import create_train_loader

# %%

from epsilon_transformers.comp_mech import (
    generate_sequences,
    mixed_state_tree,
    block_entropy,
    myopic_entropy,
    collect_path_probs_with_paths,
)

import numpy as np


def main():
    mess3_hmm = mess3()
    even_process_hmm = even_process(p=0.25)
    nond_hmm = nond()
    golden_mean_hmm = golden_mean(1, 1, 0.5)
    zero_one_random_hmm = zero_one_random(0.5)
    random_random_xor_hmm = random_random_xor(0.5, 0.5)
    print(mess3_hmm)
    print(even_process_hmm)
    print(nond_hmm)
    print(golden_mean_hmm)
    process = even_process_hmm
    generate_sequences(process, 5, 10000)

    n_ctx = 10
    MSP_tree = mixed_state_tree(process, n_ctx)

    H_mu = block_entropy(MSP_tree)
    H_mu_L = myopic_entropy(MSP_tree)

    from matplotlib import pyplot as plt

    plt.plot(H_mu_L, "o-")
    plt.show()

    data = generate_sequences(process, num_sequences=100, sequence_length=1000)

    train_loader = create_train_loader(data, batch_size=10, n_ctx=n_ctx)

    for batch in train_loader:
        print(batch)
        break

    print(f"the number of batches is {len(train_loader)}")

    path_probs = collect_path_probs_with_paths(MSP_tree, n_ctx)

    print(path_probs)

    seqs = np.array([path[0] for path in path_probs])
    probs = np.array([path[1] for path in path_probs])
    print(seqs)
    print(probs, probs.sum())

    # compute the simplex
    belief_states = MSP_tree.get_belief_states()
    # belief states is a list of np.ndarrays
    # only keep the unique np.ndarrays
    belief_states = np.unique(belief_states, axis=0)
    print(belief_states)

    # plot these belief states in a ternary plot
    # belief states are 3d, the 2d projection is the simplex
    # we can use the 2d projection  to x+y+z=1

    def project_to_simplex(points):
        """Project points onto the 2-simplex (equilateral triangle in 2D)."""
        # Assuming points is a 2D array with shape (n_points, 3)
        x = points[:, 1] + 0.5 * points[:, 2]
        y = (np.sqrt(3) / 2) * points[:, 2]
        return x, y

        projected_belief_states = project_points_onto_xyz_plane(np.array(belief_states))
        projected_belief_states = np.array(projected_belief_states)
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(projected_belief_states[:, 0], projected_belief_states[:, 1], "o")
        plt.show()

    simplex_points = project_to_simplex(np.array(belief_states))

    # add a z-dimension to the simplex points of zero
    simplex_points = np.vstack((simplex_points, np.zeros_like(simplex_points[0])))
    # make simplex points in a pandas df with columns X, Y, Z, then save to csv
    import pandas as pd

    df = pd.DataFrame(simplex_points.T, columns=["X", "Y", "Z"])
    df.to_csv("simplex_points.csv", index=False)

    from matplotlib import pyplot as plt

    plt.rcParams["figure.dpi"] = 300  # Increase the resolution of the plot
    fig, ax = plt.subplots()

    # draw the triangles
    ax.plot(
        [0, 0.5, 1, 0], [0, np.sqrt(3) / 2, 0, 0], "k-", linewidth=0.5
    )  # Adjusted line width for crispness
    ax.plot(
        simplex_points[0], simplex_points[1], "o", markersize=0.5, color="black"
    )  # Adjusted marker size for visibility
    # get rid of x,y axis, frame, etc
    ax.axis("off")

    plt.show()

    def get_optimal_prediction(belief_state, process):
        emit_probs = process.transition_probs.sum(axis=2)  # (emission, from_state)
        prediction = np.einsum("s,es->e", belief_state, emit_probs)
        return prediction

    # for each belief state, get the prediction
    predictions = np.array(
        [
            get_optimal_prediction(belief_state, process)
            for belief_state in belief_states
        ]
    )

    # now i want to plot the simplex colored by the predictions
    # the predictions are 3d

    import plotly.graph_objects as go

    fig = go.Figure()

    # Create the triangle boundary
    fig.add_trace(
        go.Scatter(
            x=[0, 0.5, 1, 0],
            y=[0, np.sqrt(3) / 2, 0, 0],
            mode="lines",
            line=dict(color="Black", width=0.5),
        )
    )

    # Prepare hover text
    hover_texts = ["Prediction: {:.2f}, {:.2f}".format(*pred) for pred in predictions]
    # Add hover info with prediction
    fig.add_trace(
        go.Scatter(
            x=simplex_points[0],
            y=simplex_points[1],
            mode="markers",
            marker=dict(
                size=5,
                color=predictions,
                colorscale="Viridis",
                showscale=True,
                line=dict(width=0.5, color="DarkSlateGrey"),
            ),
            hoverinfo="text",
            text=hover_texts,
        )
    )

    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )

    fig.show()

    # now lets create a grid of possible 3d prob dists
    # we can use this to see how the predictions change as we move through the simplex

    grid_probs = np.array(
        [
            [i, j, 1 - i - j]
            for i in np.linspace(0, 1, 100)
            for j in np.linspace(0, 1, 500)
            if i + j <= 1
        ]
    )
    predictions = np.array(
        [get_optimal_prediction(probs, process) for probs in grid_probs]
    )
    fig = go.Figure()

    # Create the triangle boundary
    fig.add_trace(
        go.Scatter(
            x=[0, 0.5, 1, 0],
            y=[0, np.sqrt(3) / 2, 0, 0],
            mode="lines",
            line=dict(color="Black", width=0.5),
        )
    )

    # Prepare hover text
    hover_texts = [
        "Prediction: {:.2f}, {:.2f}, {:.2f}".format(*pred) for pred in predictions
    ]
    # Add hover info with prediction
    grid_probs_simples = project_to_simplex(grid_probs)

    prediction_colors = [
        "rgb({},{},{})".format(pred[0] * 255, pred[1] * 255, pred[2] * 255)
        for pred in predictions
    ]
    fig.add_trace(
        go.Scatter(
            x=grid_probs_simples[0],
            y=grid_probs_simples[1],
            mode="markers",
            marker=dict(
                size=5, color=prediction_colors, colorscale="Viridis", showscale=True
            ),
            hoverinfo="text",
            text=hover_texts,
        )
    )

    # now in black lets plot the belief states
    belief_states_simples = project_to_simplex(belief_states)
    fig.add_trace(
        go.Scatter(
            x=belief_states_simples[0],
            y=belief_states_simples[1],
            mode="markers",
            marker=dict(size=2, color="Black"),
            hoverinfo="text",
            text=hover_texts,
        )
    )

    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
    )

    # Label the colorbar
    fig.update_layout(coloraxis_colorbar=dict(title="P(0 token)"))

    fig.show()

    plt.rcParams["figure.dpi"] = 300  # Increase the resolution of the plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # draw the first triangle in 3D space
    triangle_vertices = np.array(
        [[0, 0, 0], [0.5, np.sqrt(3) / 2, 0], [1, 0, 0], [0, 0, 0]]
    )
    ax.plot(
        triangle_vertices[:, 0],
        triangle_vertices[:, 1],
        triangle_vertices[:, 2],
        "k-",
        linewidth=0.5,
    )  # Adjusted line width for crispness

    # scatter points in 3D space for the first triangle
    ax.scatter(
        simplex_points[0],
        simplex_points[1],
        np.zeros_like(simplex_points[0]),
        c="black",
        marker="o",
        s=0.5,
    )  # Adjusted marker size for visibility

    # draw the second triangle in 3D space, rotated
    # the rotation is around a new axis that goes halfway through the triangle
    rotation_angle = np.radians(37)  # Convert 45 degrees rotation to radians

    # Calculate the axis of rotation as the midpoint of the base to the opposite vertex
    # Define the line of rotation by two points
    def calculate_rotation_matrix(point_on_line, direction_of_line, rotation_angle):
        # Normalize the direction vector
        direction_of_line_normalized = direction_of_line / np.linalg.norm(
            direction_of_line
        )
        a, b, c = direction_of_line_normalized
        d, e, f = point_on_line
        t = rotation_angle
        st, ct = np.sin(t), np.cos(t)
        return np.array(
            [
                [
                    ct + a**2 * (1 - ct),
                    a * b * (1 - ct) - c * st,
                    a * c * (1 - ct) + b * st,
                    (d * (b**2 + c**2) - a * (e * b + f * c)) * (1 - ct)
                    + (e * c - f * b) * st,
                ],
                [
                    a * b * (1 - ct) + c * st,
                    ct + b**2 * (1 - ct),
                    b * c * (1 - ct) - a * st,
                    (e * (a**2 + c**2) - b * (d * a + f * c)) * (1 - ct)
                    + (f * a - d * c) * st,
                ],
                [
                    a * c * (1 - ct) - b * st,
                    b * c * (1 - ct) + a * st,
                    ct + c**2 * (1 - ct),
                    (f * (a**2 + b**2) - c * (d * a + e * b)) * (1 - ct)
                    + (d * b - e * a) * st,
                ],
                [0, 0, 0, 1],
            ]
        )

    point_on_line = np.array(
        [0.5, np.sqrt(3) / 6, 0]
    )  # This is the midpoint of the triangle base
    direction_of_line = np.array(
        [1, 0, 0]
    )  # This direction makes the line perpendicular to the triangle base
    rotation_matrix = calculate_rotation_matrix(
        point_on_line, direction_of_line, rotation_angle
    )

    # Apply the rotation
    homogenous_triangle_vertices = np.hstack(
        (triangle_vertices, np.ones((triangle_vertices.shape[0], 1)))
    )
    # Correcting the subtraction operation to ensure broadcasting compatibility
    rotated_triangle_vertices_homogenous = np.dot(
        homogenous_triangle_vertices - np.hstack((point_on_line, [1])),
        rotation_matrix.T,
    ) + np.hstack((point_on_line, [1]))
    rotated_triangle_vertices = rotated_triangle_vertices_homogenous[:, :3]
    ax.plot(
        rotated_triangle_vertices[:, 0],
        rotated_triangle_vertices[:, 1],
        rotated_triangle_vertices[:, 2],
        "k-",
        linewidth=0.5,
    )
    # scatter points in 3D space for the second triangle, rotated
    # Correcting the variable used for translation to ensure compatibility
    rotated_simplex_points = np.dot(
        np.vstack(
            (
                simplex_points[0],
                simplex_points[1],
                np.zeros_like(simplex_points[0]),
                np.ones_like(simplex_points[0]),
            )
        ).T
        - np.hstack((point_on_line, [1])),
        rotation_matrix.T,
    ) + np.hstack((point_on_line, [1]))
    rotated_simplex_points = rotated_simplex_points[
        :, :3
    ]  # Discarding the homogenous coordinate
    ax.scatter(
        rotated_simplex_points[:, 0],
        rotated_simplex_points[:, 1],
        rotated_simplex_points[:, 2],
        c="black",
        marker="o",
        s=0.5,
    )
    # add an xyz axis
    ax.quiver(0, 0, 0, 1, 0, 0, color="black")
    ax.quiver(0, 0, 0, 0, 1, 0, color="black")
    ax.quiver(0, 0, 0, 0, 0, 1, color="black")

    # label the quivers
    ax.text(1.1, 0, 0, "X", color="black")
    ax.text(0, 1.1, 0, "Y", color="black")
    ax.text(0, 0, 1.1, "Z", color="black")

    # get rid of the axes
    ax.axis("off")

    # rotate the view
    ax.view_init(90, -90)

    plt.show()
    import pyvista as pv

    # Initialize plotter
    plotter = pv.Plotter()
    plotter.add_light(pv.Light(position=(-1, -2, 3), intensity=0.55))

    # Triangle vertices
    triangle_vertices = np.array(
        [[0, 0, 0], [0.5, np.sqrt(3) / 2, 0], [1, 0, 0], [0, 0, 0]]
    )

    # Define triangles (the cells array)
    cells = np.array([3, 0, 1, 2])  # The triangle is defined by these vertices

    # Create a PolyData object for the first triangle
    rotation_angle1 = np.radians(-47)  # Convert -37 degrees rotation to radians
    rotation_matrix1 = calculate_rotation_matrix(
        point_on_line, direction_of_line, rotation_angle1
    )

    rotated_triangle1_vertices = np.dot(
        np.hstack((triangle_vertices[:, :3], np.ones((triangle_vertices.shape[0], 1)))),
        rotation_matrix1.T,
    )
    triangle1 = pv.PolyData(
        rotated_triangle1_vertices[:, :3], np.hstack([[len(cells)], cells])
    )

    if isinstance(simplex_points, tuple):
        simplex_points = np.array(
            simplex_points
        )  # Assuming simplex_points is structured as (x_points, y_points)

    # Scatter points for the first triangle
    # Assuming simplex_points is a 2xN numpy array for XY points, we need to convert it
    simplex_points_3d = np.zeros((simplex_points.shape[1], 3))
    simplex_points_3d[:, :2] = simplex_points.T
    # Correcting the calculation of rotated_simplex_points1 similar to the method used above
    rotated_simplex_points1 = np.dot(
        np.hstack((simplex_points_3d, np.ones((simplex_points_3d.shape[0], 1)))),
        rotation_matrix1.T,
    )  # Removed unnecessary addition/subtraction for translation
    rotated_simplex_points1 = rotated_simplex_points1[
        :, :3
    ]  # Discarding the homogenous coordinate

    # Add the first triangle and scatter points to the plot
    plotter.add_mesh(
        triangle1,
        color="#5F9EA0",
        show_edges=True,
        line_width=1,
        opacity=1,
        specular=0.5,
        specular_power=15,
        smooth_shading=True,
    )
    plotter.add_points(rotated_simplex_points1, color="black", point_size=2, opacity=1)

    # For the rotated triangle, apply the rotation you described to triangle_vertices
    # Assuming the rotation logic is correctly implemented elsewhere

    # Rotation matrix calculation (assuming you've calculated this correctly)
    # This example will skip the matrix calculation for brevity

    # Apply rotation and translation to get the new vertices for the rotated triangle
    rotated_triangle_vertices = np.dot(
        np.hstack((triangle_vertices[:, :3], np.ones((triangle_vertices.shape[0], 1)))),
        rotation_matrix.T,
    )

    # Create a PolyData object for the rotated triangle
    triangle2 = pv.PolyData(
        rotated_triangle_vertices[:, :3], np.hstack([[len(cells)], cells])
    )

    # Add the rotated triangle to the plot with a more reddish shade
    plotter.add_mesh(
        triangle2,
        color="#D2691E",
        show_edges=True,
        line_width=1,
        opacity=1,
        specular=0.5,
        specular_power=15,
        smooth_shading=True,
    )
    # Assuming rotated_simplex_points is correctly calculated
    plotter.add_points(rotated_simplex_points, color="black", point_size=2, opacity=1)

    # Adjust camera position, focal point, and view up for better visualization
    plotter.view_xy()
    # rotate the camera a bit in every direction
    camera_position = plotter.camera_position
    # Camera_position is a tuple: (camera position, focal point, view up vector)

    # To rotate slightly, adjust the camera position while keeping the focal point and view up vector the same
    new_camera_position = (
        camera_position[0][0] + 1,
        camera_position[0][1] + 1,
        camera_position[0][2] + 1,
    )
    plotter.camera_position = (
        new_camera_position,
        camera_position[1],
        camera_position[2],
    )
    # plotter.add_light(pv.Light(position=(2, 2, 2), intensity=.5))
    # plotter.add_light(pv.Light(position=(0, 0, 5), intensity=0.37))
    # Display the plot
    plotter.show()


# %%
if __name__ == "__main__":
    main()


# %%
