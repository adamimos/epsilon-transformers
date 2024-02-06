# %%
from epsilon_transformers.comp_mech.processes import (
    mess3,
    nond,
    even_process,
    zero_one_random,
    golden_mean,
    random_random_xor,
)

# %%

from epsilon_transformers.comp_mech import (
    generate_sequences,
    mixed_state_tree,
    block_entropy,
    myopic_entropy,
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
    process = random_random_xor_hmm
    generate_sequences(process, 5, 10000, True)

    MSP_tree = mixed_state_tree(process, 11)

    H_mu = block_entropy(MSP_tree)
    H_mu_L = myopic_entropy(MSP_tree)

    from matplotlib import pyplot as plt

    plt.plot(H_mu_L, "o-")
    plt.show()


# %%
if __name__ == "__main__":
    main()


# %%
