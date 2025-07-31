- first missing stockfish dataset: thinking of why we need to use the training data- because it might interfere with activations if moves are too out of distribution
- if in-distribution moves are what we need, our stockfish game play data should be sufficient.
- 

---

## Deeper Dive: The Full Investigation into 100% Probe Accuracy

### 1. The Observation: An Unbelievable Result

Our initial experiments yielded a startling result: when training linear probes on our Stockfish-trained models (`small-16` and `small-36`), we observed near-100% accuracy in predicting board state across **all layers**. This was deeply counter-intuitive, as it suggested even the first layer of the transformer had a near-perfect internal model of the chessboard, contradicting previous findings on models trained on more diverse datasets like Lichess, where accuracy grew with model depth.

### 2. Initial Hypotheses: Why So Accurate?

We formulated several hypotheses to explain this phenomenon:

*   **Model Capacity Liberation:** Since our dataset contained no ELO ratings, perhaps the model didn't need to waste capacity learning to model different skill levels. This freed-up capacity could have been dedicated to achieving a much stronger internal representation of the board state.
*   **Stockfish Determinism:** The games were generated with Stockfish using a fixed number of nodes per move. Could this have made the moves deterministic given a board state, leading to highly predictable game trajectories that were easier for the model to learn?
*   **Human Data Complexity:** Perhaps the Lichess dataset is simply far more erratic and complex. Modeling the vast range of human playstyles, including blunders, traps, and inconsistencies, might be a much harder task that consumes a significant portion of the model's learning capacity, leaving less for a perfect world model.
*   **Superior Training Data:** Conversely, perhaps our Stockfish data was simply *better*. By training exclusively on games between a 3200 ELO engine (White) and various strong Stockfish levels (Black), the model was exposed to a consistent, high-level of play. This could have enabled it to develop a more robust and generalized understanding of chess, leading to a stronger world model than one trained on the noisier, lower-ELO human games.

### 3. A Strategy for Systematic Testing

To test these ideas, we devised a clear plan:

1.  **Cross-Model/Data Testing:**
    *   Train probes for our Stockfish model on the Lichess dataset.
    *   Train probes for the original paper's Lichess-trained model on our Stockfish dataset.
2.  **The Control Experiment:** Train probes on a model with **randomly initialized weights** using our Stockfish data. This would be the ultimate test: if the high accuracy was due to the model's learned representations, this setup should fail completely.

### 4. The Shocking Results

The experiments yielded a series of crucial clues:

*   Training on the Lichess data produced the expected result: accuracy was modest in early layers and grew with depth. This suggested the probing framework itself was sound.
*   However, training on our Stockfish data with a **randomly initialized model** still produced ~94% accuracy.

This was the smoking gun. The problem was not our model's superior training; the high accuracy was achievable *without any learned representations at all*. The issue had to be in the data itself or how we were processing it.

### 5. The "Aha!" Moment: It's the Tokenization

We compared the data formats and found minor differences (e.g., `;;;` prefix, spaces after move numbers), but fixing these didn't solve the problem. The openings were very repetitive, but that alone couldn't explain how a linear probe on random activations could succeed.

The final breakthrough came from inspecting the probe-training code itself. How did it select which activations to train on from the sequence of moves?

We found the answer in the `custom_indexing_function`. The default setting was `find_dots_indices`, meaning the probe was being trained to predict the board state using the activations from the model at the precise character position of every `.` in the PGN string (e.g., `1.e4` -> probe at index of `.`).

Because our Stockfish dataset had highly repetitive openings, the character sequence leading up to the first, second, and third dots was almost always the same. The probe wasn't learning chess; it was learning a trivial lookup:
*   "At position of the first `.`, the board is always in the starting state."
*   "At position of the second `.`, the board is almost always in state X."
*   "At position of the third `.`, the board is almost always in state Y."

The linear probe could easily solve this task even on random activations because the activations at these fixed, early positions would have very low variance across the dataset, creating a simple, learnable mapping.

### 6. Reflections: Why Did This Trivial Task Work So Well?

The 100% accuracy wasn't a result of the model hallucinating or getting confused. It was the opposite: the task was made so simple that even a linear classifier could solve it perfectly.

By always probing at the same syntactically determined points (`.`) in a dataset with very little variation in the opening sequences, we weren't evaluating the model's ability to understand a dynamic board state. We were evaluating its ability to recognize a fixed character position within a string. The model's internal state at character #3 (the first dot) is determined by the input `";;;1"`. Since this is constant across thousands of games, the resulting activation vector is also nearly constant, making it trivial for a probe to map it to the starting board state. This created an inadvertent data leak where positional information, not semantic understanding, was the primary signal.
