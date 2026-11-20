#!/usr/bin/env python3

import pandas as pd
import numpy as np


def main():
    # Load species and strains matrices
    mat_sp = (
        pd.read_csv("mat_common_species.tsv", sep="\t", index_col=1, header=1)
        .drop("sample_id")
        .drop("sample_id", axis=1)
    )

    mat_st = (
        pd.read_csv("mat_strains.tsv", sep="\t", index_col=1, header=1)
        .drop("sample_id")
        .drop("sample_id", axis=1)
    )

    # Convert to long format
    mat_sp_long = mat_sp.stack().reset_index()
    mat_sp_long.columns = ["sample_id_1", "sample_id_2", "n_shared_species"]

    mat_st_long = mat_st.stack().reset_index()
    mat_st_long.columns = ["sample_id_1", "sample_id_2", "n_shared_strains"]

    # Merge species and strains
    mat = pd.merge(
        mat_sp_long, mat_st_long, how="left", on=["sample_id_1", "sample_id_2"]
    )

    # Compute strain sharing rate
    mat["strain_sharing_rate"] = np.divide(
        mat["n_shared_strains"], mat["n_shared_species"]
    )

    # Keep only unique (unordered) pairs
    mat = mat[
        pd.DataFrame(np.sort(mat[["sample_id_1", "sample_id_2"]].values, 1)).duplicated()
    ]

    # Save output
    mat.to_csv("pairwise_sharing_allRaxml.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
