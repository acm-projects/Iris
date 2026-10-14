"""A small recurrent classifier over pose keypoint sequences.

Kept deliberately small: with ~55 training clips, a big model just memorizes.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PoseGRU(nn.Module):
    """Reads a clip frame by frame and outputs a score for each word.

    Pipeline for one clip:
      (32 frames, 110 numbers each)
        -> embed: squeeze each frame's 110 numbers into 64 learned features
        -> GRU:   read the frames in order, building up a summary of the motion
        -> mean:  average the GRU's output over all frames
        -> head:  turn that summary into one score per word
    """

    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.3):
        """
        input_size:  numbers per frame (55 joints x 2 coordinates = 110)
        num_classes: how many words we're choosing between (2 for book/drink)
        hidden_size: size of the internal feature vectors; bigger = more capacity,
                     but also easier to overfit with little data
        num_layers:  how many GRU layers to stack
        dropout:     fraction of features randomly zeroed during training, which
                     stops the model leaning too hard on any single feature
        """
        super().__init__()

        # Per-frame feature extractor. Applied to every frame independently.
        self.embed = nn.Sequential(
            nn.Linear(input_size, hidden_size),  # 110 joint coordinates -> 64 features
            nn.LayerNorm(hidden_size),           # keeps the features in a stable range, helps training
            nn.ReLU(),                           # non-linearity, so it can learn more than straight lines
            nn.Dropout(dropout),
        )

        # GRU = Gated Recurrent Unit, a network designed for sequences. It walks
        # through the frames one at a time, carrying a memory of what it has
        # seen so far. bidirectional=True runs a second GRU backwards through
        # the clip as well, so every frame "knows" what came before and after.
        # batch_first=True means inputs are shaped (batch, frames, features).
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)

        # Final classifier. The bidirectional GRU outputs 2 x hidden_size
        # features (forward + backward), mapped to one score per word.
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden_size, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, frames, features) -> logits (batch, num_classes)."""
        # h has one output vector per frame: (batch, frames, 2 * hidden_size).
        # The second return value (the GRU's final memory state) isn't needed.
        h, _ = self.gru(self.embed(x))

        # Average over the frame dimension to get one vector per clip, then
        # score it. The outputs are "logits": raw scores where higher means
        # more likely. softmax() turns them into probabilities if needed.
        return self.head(h.mean(dim=1))
