**EOGConv (Edge-Only Graph Convolution)** is a custom neural network layer designed for graphs where all useful information is stored in the edges, not the nodes.

Most GNN architectures assume node features are the primary carriers of information, with edges serving as optional context.  
EOGConv flips that assumption: it performs message passing entirely based on **edge features**, making it suitable for any graph structure where edges encode relationships, interactions, or signals that dominate the learning task.
Many real-world graphs can naturally encode information in edges:
- **Chemical graphs:** resonance, bond types, and electronic features
- **Electrical circuits:** component types, resistances, capacitatances, and inductances  
- **Transportation networks:** connections encode distances, speeds, flows  
- **Any scenario where nodes are blank placeholders, but edges carry structure**

Traditional GCNs struggle when node features are empty or uninformative. EOGConv addresses this by designing the entire message-passing mechanism around edge attributes.

How to Cite
If you use this layer in academic work, please cite the repository:
Ghiasi, F. (2025). EOGConv: Edge-Only Graph Convolution Layer.
GitHub: https://github.com/fatimaghiasi/EOGConv

@misc{ghiasi2025eogconv,
  author       = {Fatima Ghiasi},
  title        = {EOGConv: Edge-Only Graph Convolution Layer},
  year         = {2025},
  howpublished = {\url{https://github.com/fatimaghiasi/EOGConv}}
}
