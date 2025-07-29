# Chess LLM Mechanistic Interpretability Setup Guide

This guide will help you run **mechanistic interpretability analysis** on custom chess models using linear probes, interventions, and granular diagnostics. This repository is designed for systematic investigation of how chess-playing LLMs develop internal world models.

## 🎯 **Project Context**

This setup supports a **foundational benchmark study** investigating:
- **Scaling laws** in domain-specific performance (depth vs. width efficiency)
- **Internal world model representations** through linear probing
- **Granular chess diagnostics** beyond surface-level ELO metrics
- **Mechanistic understanding** of how LLMs acquire chess abilities

You have **7 Stockfish-trained chess models** that need comprehensive interpretability analysis to understand their internal representations and compare architectural choices.

## 📋 **Prerequisites**

- **Python 3.8+** with PyTorch
- **7 chess models** trained on Stockfish games (located in `/models` directory)
- **Model specifications**: layers, hidden dimensions, attention heads, parameter counts
- **Computational resources**: GPU recommended for probe training (10-15 min per model)

## 🚀 **Step-by-Step Setup**

### **Step 1: Environment Setup**

```bash
# Install dependencies
pip install -r requirements.txt

# Verify CUDA availability (optional but recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### **Step 2: Download Baseline Model & Verify Setup**

```bash
python model_setup.py
```

**What this accomplishes:**
- Downloads author's pre-trained 8-layer baseline model (102MB)
- Converts from nanoGPT → TransformerLens format for interpretability
- Runs verification test to ensure conversion integrity
- **Expected output**: `tensor([[True, True, True...]])` confirming successful setup

### **Step 3: Prepare Stockfish Evaluation Dataset**

**⚠️ CRITICAL:** Your models were trained on **Stockfish games**, so you **must use Stockfish evaluation data** to avoid domain mismatch.

```bash
jupyter notebook utils/chess_gpt_eval_data_filtering.ipynb
```

**Why Stockfish data matters:**
- **Domain alignment**: Engine games match your training distribution
- **Skill classification**: Uses `player_two` column (engine skill levels) vs `WhiteEloBinIndex` (human ELO)
- **Fair evaluation**: Avoids underestimating interpretability due to human/engine game pattern differences

**Expected output**: `data/stockfish_train.csv` and `data/stockfish_test.csv`

### **Step 4: Configure Training Script**

Before running probes, modify the training script for Stockfish data:

```python
# Edit train_test_chess.py line 905
# Change from:
dataset_prefix = "lichess_"
# To:
dataset_prefix = "stockfish_"
```

This ensures the repository uses the correct skill classification column for your engine-trained models.

## 🔬 **The Probing Journey: Step-by-Step Analysis**

### **Phase 1: Single Model Exploration**

Start with **one model** to understand the interpretability pipeline:

#### **1.1 Board State Probes**
```bash
python train_test_chess.py --probe piece --model_name your_first_model
```

**What you'll observe:**
- **Training progress**: Probe accuracy improving from ~75% → ~98%
- **Layer-by-layer performance**: How board state representations develop across depths
- **Emergent patterns**: Which layers capture spatial chess understanding

#### **1.2 Skill Estimation Probes**
```bash
python train_test_chess.py --probe skill --model_name your_first_model
```

**What this reveals:**
- **Player strength modeling**: How well the model internally estimates skill levels
- **Distribution matching**: Evidence of models suppressing optimal moves to match training distribution
- **Layer specialization**: Which depths encode strategic understanding

#### **1.3 Analyze Initial Results**

Check generated files:
- `results/piece_probe_results.csv` - Board state accuracy by layer
- `results/skill_probe_results.csv` - ELO prediction performance
- `models/probes/` - Trained probe weights for interventions

### **Phase 2: Systematic Model Comparison**

#### **2.1 Architecture-Performance Mapping**

For each of your 7 models, document:
- **Architecture**: layers, hidden dimensions, total parameters
- **Training details**: dataset size, computational resources
- **Hypotheses**: expected scaling law behaviors

#### **2.2 Probe All Models**

Run the same probe analysis on each model:

```bash
# For each model, run both probe types
python train_test_chess.py --probe piece --model_name model_2
python train_test_chess.py --probe skill --model_name model_2
# ... repeat for all 7 models
```

#### **2.3 Comparative Analysis**

**Key research questions to investigate:**
- **Depth vs. Width**: Which architectural choice improves interpretability?
- **Parameter efficiency**: Do deeper models achieve better representations with fewer parameters?
- **Layer specialization**: How do different architectures distribute chess knowledge across layers?

### **Phase 3: Advanced Interpretability**

#### **3.1 Board State Interventions**

Test causal relationships between internal representations and behavior:

```bash
python board_state_interventions.py --model_name your_best_model
```

**What this validates:**
- **Causal world models**: Do internal board representations actually influence move generation?
- **Intervention effectiveness**: How robust are the learned representations?
- **Mechanistic understanding**: Can we causally manipulate chess understanding?

#### **3.2 Granular Chess Diagnostics**

Beyond probe accuracy, analyze chess-specific behaviors:

**Move Legality Analysis:**
- Track illegal move patterns across models
- Correlate with game length (state tracking errors)
- Identify architectural factors affecting legal move generation

**Move Quality Distribution:**
- Categorize moves: Brilliant, Best, Good, Okay, Mistake, Blunder
- Analyze performance vs. Stockfish across game phases
- Understand when and why models make poor decisions

#### **3.3 Custom Probe Development**

Extend beyond basic probes to investigate:
- **Tactical motif detection**: Forks, pins, skewers
- **Positional evaluation**: Material advantage, piece activity
- **Game phase recognition**: Opening, middlegame, endgame transitions

## 📊 **Understanding Your Results**

### **Probe Performance Interpretation**

**High Board State Accuracy (>95%):**
- Model has robust internal spatial representation
- Good candidate for interventional studies
- Likely demonstrates genuine world model learning

**Layer-by-Layer Patterns:**
- **Early layers**: Basic pattern recognition
- **Middle layers**: Peak board state accuracy (spatial reasoning)
- **Later layers**: Strategic/tactical integration

**Architecture Insights:**
- **Depth efficiency**: Deeper models may achieve better representations with fewer parameters
- **Specialization**: Different layers encode different aspects of chess understanding

### **Research Contributions**

Your analysis will contribute to understanding:
1. **Domain-specific scaling laws** in structured environments
2. **Distribution matching vs. task optimization** in LLM training
3. **Mechanistic basis** of emergent world models
4. **Architectural choices** affecting interpretability

## 🔍 **Troubleshooting**

### **Model Loading Issues**
If your models need format conversion:
```python
# Modify model_setup.py for your specific format
def load_custom_model(model_path):
    checkpoint = torch.load(model_path)
    # Add your conversion logic here
    return converted_model
```

### **Memory Management**
For large models or limited GPU memory:
- Reduce batch size in `chess_utils.py`
- Process models sequentially rather than in parallel
- Use gradient checkpointing if available

### **Data Format Validation**
Ensure your Stockfish data has correct columns:
- `player_two` for skill classification
- Proper PGN format with semicolon delimiters
- Consistent game length (365 characters recommended)

## 🎯 **Research Timeline**

**Phase 1 (Single Model)**: 2-3 hours
- Setup and baseline probes
- Understanding the pipeline
- Initial result interpretation

**Phase 2 (All 7 Models)**: 4-6 hours
- Systematic probe training
- Comparative analysis
- Scaling law investigation

**Phase 3 (Advanced Analysis)**: 3-4 hours
- Interventions and diagnostics
- Custom probe development
- Research synthesis

**Total Estimated Time**: 10-15 hours for comprehensive analysis

## 📈 **Expected Discoveries**

Based on the research framework, you may discover:
- **Architectural biases** in learning structured domains
- **Evidence for depth efficiency** in parameter scaling
- **Mechanistic understanding** of illegal move generation
- **Distribution matching behaviors** in skill modeling
- **Layer specialization patterns** across different architectures

## 🚀 **Next Steps After Analysis**

1. **Synthesize findings** across all 7 models
2. **Identify best-performing architectures** for interpretability
3. **Develop hypotheses** about scaling laws in chess domains
4. **Plan interventional studies** on top-performing models
5. **Document insights** for foundational benchmark paper

This systematic approach will provide comprehensive mechanistic understanding of your chess models and contribute to the broader interpretability research landscape.

Good luck with your mechanistic interpretability journey! 🏆♟️ 