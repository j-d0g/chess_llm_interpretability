# Results

### Data Visualisation: Stockfish Games (Now)

> Wrap-up data visuals, especially resolving stockfish analysis ones, to a useable standard.
> 
- [ ]  Stockfish level score: breakdown?
- [ ]  Turn dashboard into pdf format? Combine the data visuals.
- [ ]  Sample 100 games, prove uniqueness compared to training data. (Muyao)

### Linear Probing

> Walk-through interpretability repository
> 
- [ ]  Assess probes trained on randomly initialised weights first.
- [ ]  Train all models’ probes, prioritise small-36 and large-16. Then waterfall down to small-24 and medium-16. Do this for board-state first.
- [ ]  Generate results tables, plots, heat-maps, replicating the same metrics Adam does. Layer-by-layer performance is crucial.
- [ ]  Perform intervention on a single model to verify.

# Thesis

### Plan & Initial Draft (Tonight)

- [x]  Generate initial outline of headers, rough contents, integrating past and present research.
- [ ]  Expand outline points to be more verbose, contextual, directed with a good idea of the talking points I intend. Back-and-forth with AI on the decision-making here.
    - [ ]  NotebookLM: Initial Rubric/Example Cross-Check Validation.
    - [ ]  Claude: Write-Up Focusing on Meat from Above.
    - [ ]  Perplexity: Expand on above with LLM grounded in Real World Context.
- [ ]  Do the above for the objective parts initially:
    - [ ]  Introduction, Background, Methodology, Experiments/Techniques/Results. Avoid asserting too much detail into insights; keep them as bullet points for expanding.
    - [ ]  Insert questions / holes fillable by results and research. If any questions, refer back to the visualisations to understand the findings.
    - [ ]  **Identify strengths** of research: but also **limitations**. **Acknowledge these!!!**
- [ ]  **Figures, Tables, Data and Diagrams:** Now, layer in the right data, tables, visualisations, figures and diagrams- grounding your work and correcting figures and values. Hone the explanations to fit these, and vice versa, ensuring correctness.
    - [ ]  Use cursor to run scripts that analyse useful statistics and generate README files if needed to support LLM context for later on these parts! Include them here!

### Penultimate Draft (Wednesday)

> With the structure, narrative and skeleton in place, with sufficient data, figures, talking points and direction: we can move through these one-by-one with Claude AI.
> 
- [ ]  **Prep for review:** Use NotebookLM, Perplexity and Gemini to review the Final Plan Outline for thoughts, improvement, grounding in sources, tagging, talking points for conclusion/reflection/abstract etc. before moving onto Claude for Write-Up and Perplexity for Review.
- [ ]  **Write-up:** Move through each part with Claude for write-up.
    - [ ]  One Claude AI for writing.
    - [ ]  One Perplexity & NotebookLM AI for cross-checking, and strengthening links between literature/sources/my blog story.
    - [ ]  One Claude AI / Gemini for reviewing, planning next steps, and finding links.
    - [ ]  One Cursor for formatting into Markdown / LaTeX.
    - [ ]  One Cursor for formatting into LaTeX.
- [ ]  Pass Markdown File to someone to reconstruct in LaTeX.
- [ ]  Pass Markdown Draft to Terry *tomorrow night.*

# Screencast

### (Thursday Morning-Afternoon)

- [ ]  Finish Full-Stack! (120)
    - [ ]  Softmax Visuals. (Must)
    - [ ]  Board Probe Reconstruction Visuals. (Ext.)
    - [ ]  Game History, Tournament & Leaderboard (Ext.)
- [ ]  Presentation! (120)

### (Thursday Evening)

- [ ]  Review feedback, make final adjustments, iterate ASAP.

I will submit:

- Code, 23.59.
- Screencast, 23.59.
- Thesis, EoD Friday…?

# Data Visualisation — Chess Benchmarks

> We should choose a few focuses now, and dedicate a full page of different transformations targeted at understanding them.
> 

### **Illegal Moves**

> *When, Where, How often, and Why does the model make illegal moves?*
> 
- How do scaling laws affect move legality, and what does that tell us about how models are training?
- Where do illegal moves occur the most frequently, and what does that tell us?
    - By model size: scaling laws? bigger = less illegality.
    - By game stage: move number? early.
    - By stockfish levels: no obvious correlation.
    - By advantage: when board-state is winning or losing?
    - By possibilities: when there are more or less legal moves to sample from?
    - By complexity: when there are fewer good moves to choose from?
    - By opponent’s move quality: when the opponent blunders/best?
    - By illegal move type:
        - Can the piece move in that direction generally? Basic Error.
        - IF it can, why can’t it? Notation Error (Lack of Full-Move Notation in Ambiguous Scenario), Board-State Issue (Check, Captures, Piece Blocking).
- Extract all instances of illegal moves (take the pgn sequence) and represent them by the features above and identify trends. <1% of 100,000 should be 1,000 illegal move sequences.
    - Visualise? SVM? Probe above states and construct internal representation of board-state to see if mapped correctly?
- Finally, why do I suspect illegal moves occur?

### **Gameplay Quality**

> *Beyond win-rates, how good are moves on average? When does the model play well, vs. when does it blunder- are there trends or patterns that indicate this? How does this relate to the training process, or LLM architecture / machine learning decisions?*
> 

**Win-rate vs. Stockfish**

- The obvious baseline benchmark: how often does each model win across stockfish levels?
- How does this distribute across features like move bins?

**Move Quality**

- How does scaling laws affect model performance? Layers, Hidden-Size, Total Parameters.
    - Win-rate: more parameters = Better performance. Insufficient evidence regarding Depth vs Width, as no equivalent parameter comparisons.
    - Move-quality: does it affect the average quality of move, blunder rate, the spread of move quality across a game? So far we notice that larger models perform better for longer.
    - Illegal-moves: yes it affects, but do layers/width affect?
    - Probing: does layers/hidden-size affect how early/late latent features are learned and/or represented? does it affect how many layers maintain high performance?

**Board-State Advantage**

- When does the advantage typically drop-off for each model / stockfish opponent?
    - By move number?
    - By complexity?
    - By possibilities?
- How well does the model perform heuristically despite win/loss?
- Does the model dominate, hold it’s ground until a fatal blunder, or does it generally make series of bad moves that cannot be excused in leading to its downfall?
- What would the win-rates be if we denote a win by average advantage score across a game, or average number of winning positions in a game? total number of winning positions vs losing position scores etc. across different max move limits (idea is including games that pass the 60, 70, 80 move mark might skew results, as there’s an abnormally large number of games which last that long due to our artificial cut-off. observing intervals of max-move limits from 30, 40, 50… 90 will give us an indication into when and why the model loses.).
- Does average move quality correlate with win-rate? How often did the model lose

For each of the above, we start at the highest level of visualisation:

- For each of our models, how do they stack up?
    - Illegal move rate.
    - Average centi-pawn loss.
    - Average #winning positions.
    - Average #
- The above can be broken down to heat maps visualised against:
    - 10 Stockfish Opponents
    - Move Number/Bins.
- Not a bad idea to radix chart the above features once they’re well defined: radix chart for above (both at sub-level and top level).

### Entropy

> Are there common sub-sequences of games that lead to repeated loss? Does the model tend to play the same kinds of games all the time?
> 

**Puzzle Performance**

Performance of my models across a set of 1,000 puzzles?

- n-move puzzles
- glicko difficulty
- checkmate
- tactics
- sacrifices

How

- Download puzzles dataset.
- Use Lichess API to fetch pgn from game-links.
- Sample from NanoGPT & record success rate, number of moves / n correct.
- Table of stats.

### Game

- [ ]  Wrap-up stockfish visuals, validation, and answering the questions we want answered:
    - [ ]  What kind of illegal moves are being made? What kinds of scenarios are they being made most frequently- early game? Why?
    - [ ]  Currently most graphs are proportions of games within a move distribution. But can we break each of these into wins/losses too? What are the winning and losing trends of our models across stockfish levels?

### Analysis

- [ ]  Solve the stockfish segmentation issue: where is stockfish 0-9 id’d?
- [ ]  Advantage graphs: not by move centi-pawn loss, but by overall board-state!
    - [ ]  Who’s winning at what points most of the time???
    - [ ]  Where does this drop / change in favour the most often?
    - [ ]  At the points do they tend to drop? Are there any patterns once they drop?
        - [ ]  Analyse a sample of some games one-by-one first, see if there are any trends, across models / stockfish levels, then
- [ ]  Ext. Illegal moves: do they usually occur when the model is winning or losing? Can we combine these datasets?