A variant of FluidGPT, my miniature transformer model, is trained on chess games via UCI token predictions. It has about 33.4 million trainable parameters. Athough I have coded the FluidGPT and ChessGPT myself, I did use AI tools for debugging and prebuilt snippets for convenience. The model is still under development with training refinements.

Model Overview:

| Parameter         | Value                                |
| ----------------- | ------------------------------------ |
| Total Parameters  | ≈ 33.4 million                       |
| Layers            | 6                                    |
| Heads             | 8                                    |
| Embedding Dim     | 512                                  |
| Feedforward Depth | 2048                                 |
| Block Size        | 80 tokens                            |
| Dropout           | 0.2                                  |
| Weight Decay      | 0.01                                 |

The dataset consists of 10,000+ PGN games from Magnus Carlsen and elite tournaments.
