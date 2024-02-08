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
    collect_path_probs_with_paths
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
    process = mess3_hmm
    generate_sequences(process, 5, 10000)

    MSP_tree = mixed_state_tree(process, 7)

    H_mu = block_entropy(MSP_tree)
    H_mu_L = myopic_entropy(MSP_tree)

    from matplotlib import pyplot as plt

    plt.plot(H_mu_L, "o-")
    plt.show()

    data = generate_sequences(process, num_sequences=100, sequence_length=1000)

    train_loader = create_train_loader(data, batch_size=10, n_ctx=10)

    for batch in train_loader:
        print(batch)
        break

    print(f"the number of batches is {len(train_loader)}")

    path_probs = collect_path_probs_with_paths(MSP_tree, 10)

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


    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    
    # draw the triangles
    ax.plot([0, 0.5, 1, 0], [0, np.sqrt(3)/2, 0, 0], "k-")
    ax.plot(simplex_points[0], simplex_points[1], ".")
    plt.show()




# %%
if __name__ == "__main__":
    main()


# %%
