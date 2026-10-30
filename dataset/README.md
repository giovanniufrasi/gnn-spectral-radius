# CuriosAI graph dataset

Place the original graph JSON files directly in this directory before running Section 2 of `gin_vs_gcn.ipynb`. The notebook reads `dataset/*.json` relative to the repository root; it does not scan subdirectories or open ZIP files.

The original experiment is documented as loading 313,597 simple, undirected, pairwise non-isomorphic graphs with 2–50 nodes, then selecting a shuffled subset of 50,000. Dataset attribution and its role in the experiment are described in the thesis.

## Expected structure

Each JSON file must contain a top-level `graph_list` array. Each graph record must supply:

| Key | Expected content |
|---|---|
| `nodes` | Node collection; its length determines the number of nodes |
| `edges` | List of endpoint pairs, using integer node indices from 0 to the number of nodes minus 1 |
| `graph_features` | Feature list with at least seven entries; index 6 is used as the adjacency spectral radius target |

In the original loader, each undirected edge appears once in the source JSON. `to_undirected` adds reverse edges for message passing. Graph annotations other than the target are not fed into the network.

## Preserve the original sample

- Keep the original JSON filenames, including their numeric components: the loader sorts files by the first number in each filename.
- Keep the order of records inside each `graph_list` and the contents of all files unchanged.
- Keep every original source file used for the documented experiment. A smaller or different dataset changes the selected sample and can change the reported results.
- Keep documentation in Markdown files, not additional `.json` files: every file matching `*.json` is treated as graph data.

The loader fixes the shuffle and split seeds but does not independently check graph isomorphism or recompute eigenvalues.

## Distribution

Use the source dataset's existing attribution and distribution terms when adding the original files. If distributing the data as a separate downloadable archive, extract it so that the JSON files end up directly in `dataset/`, and provide the actual download location in the main README.
