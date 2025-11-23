**EOGConv (Edge-Only Graph Convolution)** is a custom neural network layer designed for graphs where all useful information is stored in the edges, not the nodes.

Most GNN architectures assume node features are the primary carriers of information, with edges serving as optional context.  
EOGConv flips that assumption: it performs message passing entirely based on **edge features**, making it suitable for any graph structure where edges encode relationships, interactions, or signals that dominate the learning task.
Many real-world graphs can naturally encode information in edges:
- **Chemical graphs:** resonance, bond types, and electronic features
- **Electrical circuits:** component types, resistances, capacitatances, and inductances  
- **Transportation networks:** connections encode distances, speeds, flows  
- **Any scenario where nodes are blank placeholders, but edges carry structure**

Traditional GCNs struggle when node features are empty or uninformative. EOGConv addresses this by designing the entire message-passing mechanism around edge attributes.

**How to Import**
Place "EOGConv.py" in your project, then call:
"from EOGConv import EOGConv"
This layer requires that Pytorch is already installed in the python environment.

**Layer Parameters**
in_channels : int -> Number of input channels per edge.
out_channels : int -> Number of output channels per edge.
aggr : {"sum", "mean"} (default "mean") -> Aggregation method for edge neighbors.
directed : bool (default False) -> If True, messages can only flow from the node left of the edge. Right weights are set to zero.
symmetric : bool (default True) -> If True and not directed, left/right weights become identical.

**Input Format**
EOGConv is an **edge-centric**, anisotropic message-passing layer. It requires two tensors, and (optionally) a third tensor for batching.

1. edge_x: Per-endpoint edge features
Shape = [E, 2, C]
E= number of edges
2 endpoints per edge:
edge_x[e, 0, :] = features for endpoint attached to node edge_index[0, e]
edge_x[e, 1, :] = features for endpoint attached to node edge_index[1, e]
C = number of channels (user-defined)
If the edge has no directional/polar information, simply duplicate features: edge_x[e, 0, :] = edge_x[e, 1, :]

2. edge_index: Edge connectivity
Shape = [2, E]
edge_index[0, e] = node index at endpoint 0
edge_index[1, e] = node index at endpoint 1
Node indices must be integers in [0, num_nodes - 1].
This is identical to PyTorch Geometric format.

3. (optional) edge_batch: Graph assignment per edge
Shape = [E]
edge_batch[e] = k means: edge e belongs to graph k.
It is required whwn passing batched inputs.

**How to use in a custom pytorch model**
Define it in  "__init__(self)":
self.conv1 = EOGConv(, 16, aggr, directed=False, symmetric=True)







**How to Cite**
If you use this layer in academic work, please cite the repository:
Ghiasi, F. (2025). EOGConv: Edge-Only Graph Convolution Layer.
GitHub: https://github.com/fatimaghiasi/EOGConv

@misc{ghiasi2025eogconv,
  author       = {Fatima Ghiasi},
  title        = {EOGConv: Edge-Only Graph Convolution Layer},
  year         = {2025},
  howpublished = {\url{https://github.com/fatimaghiasi/EOGConv}}
}
