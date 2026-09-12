# slm-hybrid-prompting-thesis

**Honors Thesis — The Pennsylvania State University, Schreyer Honors College**  
**Department of Computer Science and Engineering**

> Optimizing Quantitative Reasoning in Small Language Models Through Hybrid Prompting  
> Siddhardha Basu · Fall 2025

📄 **[Read the full thesis](https://honors.libraries.psu.edu/catalog/9882syb5570)**

---

## Abstract

This thesis investigates how **structured prompting strategies** improve reasoning accuracy in **small language models** under computational constraints. Four prompting methods — **Direct**, **Self-Ask**, **Program-of-Thought (PoT)**, and a **Hybrid** combining Self-Ask with PoT — were evaluated on two quantitative reasoning datasets: **GSM8K** for mathematical problems and **FinQA** for financial reasoning. Experiments were conducted using four instruction-tuned models between **4 and 8 billion parameters** on an RTX 3080 GPU, measuring exact match accuracy, relative numeric error, execution success, and latency.

Results show that structured prompting significantly enhances performance compared to unstructured inputs. Self-Ask improved reasoning coherence but exhibited arithmetic errors, while PoT achieved higher numeric precision through **executable computation**, albeit with occasional runtime issues. The Hybrid approach produced the **highest overall accuracy** in extraction-heavy contexts, maintaining stable performance across both domains. Error analysis revealed Hybrid substantially reduced both **extraction and arithmetic errors** by addressing complementary failure modes through its **two-phase structure**: decomposition followed by code execution.

However, important trade-offs emerged. Hybrid introduced execution failures, added latency, and consumed significantly more tokens than baseline prompting. In GSM8K, where problems were clearly stated, Hybrid showed inconsistent benefits, suggesting **context-dependent effectiveness**. The approach proved most valuable when both extraction complexity and computational precision were required, but added unjustified overhead when problem statements were already clear or arithmetic was simple.

These findings demonstrate that **well-designed prompting structures can offset reasoning limitations of smaller models**, improving precision and interpretability without additional fine-tuning. Overall, the study highlights that hybrid structured prompting provides an effective pathway to strengthen reasoning reliability in **small-scale models suitable for local or resource-constrained applications**.

---

## Key Results

| Metric | Result |
|---|---|
| Accuracy improvement over Direct (FinQA) | **2–4×** |
| Extraction error reduction (Hybrid vs. Direct) | **71%** |
| Arithmetic error reduction (Hybrid vs. Direct) | **77%** |
| Models evaluated | **4 (4B–8B parameters)** |
| Datasets | **GSM8K, FinQA** |
| Hardware | **NVIDIA RTX 3080, 10GB VRAM** |

---

## Prompting Methods

- **Direct** — Baseline: question + minimal formatting, no intermediate scaffolding
- **Self-Ask** — Decomposes problems into sub-questions answered step by step
- **Program-of-Thought (PoT)** — Generates executable Python code for deterministic arithmetic
- **Hybrid** *(thesis contribution)* — Two-phase: Self-Ask decomposition followed by PoT code execution

---

## Methodology

### Experimental Framework
The study compared **four prompting strategies** across **two quantitative reasoning datasets** under consistent experimental conditions, isolating the effects of prompt structure while holding model parameters, datasets, and hardware constant. All models were run at **full precision** on a local runtime to eliminate external API latency and ensure reproducible throughput.

### Hardware & Configuration

| Setting | Value |
|---|---|
| **GPU** | NVIDIA RTX 3080, 10GB VRAM |
| **Context Length** | 4,096 tokens |
| **Max New Tokens** | 512 |
| **Temperature** | 0.2, fixed seed |
| **Code Execution** | Python 3.11 runtime |

### Datasets
Two datasets were selected to represent complementary quantitative reasoning demands:

- **FinQA** — Financial reasoning grounded in semi-structured context and tables. Requires selecting relevant quantities from dense financial documents, performing arithmetic or ratio operations, and producing a precise numeric answer. Emphasizes **arithmetic fidelity and logical grounding**.
- **GSM8K** — Grade-school math word problems requiring **multi-step symbolic reasoning** over abstract language, without external evidence tables.

### Prompt Design
All four prompts enforced consistent output formatting and explicit final answer syntax:

- **Direct** — Question, context, and output format only. Measured raw reasoning ability with no guided decomposition.
- **Self-Ask** — Instructed the model to list sub-questions, answer them, and combine into a final result. Enforced **modular stepwise reasoning**.
- **Program-of-Thought** — Instructed the model to write a single executable Python block that prints the final numeric answer. Delegated arithmetic to the **Python interpreter for deterministic computation**.
- **Hybrid** *(core thesis contribution)* — Two-phase structure combining the above:
  - **Phase 1:** Self-Ask decomposition — list and answer sub-questions, identify relevant quantities
  - **Phase 2:** Program-of-Thought execution — translate Phase 1 results into executable Python code

### Evaluation Metrics
Three metric categories were used:

**Accuracy**
- **Exact Match (EM@τ)** — Percentage of predictions within tolerance τ ∈ {0.1%, 1%, 5%}
- **Average Relative Numeric Error (RNE)** — Mean relative deviation from correct value
- **Accuracy per Token / per Second** — Efficiency-normalized accuracy measures

**Efficiency**
- **LLM Latency** — Mean text generation time
- **Program Latency** — Python execution time (PoT and Hybrid only)
- **Program Error Rate** — Fraction of runs producing runtime or syntax errors

**Error Attribution**
- **Error Location** — Distribution across extraction, arithmetic, formatting, and execution failures
- **First-Error Position** — Location of first deviation in reasoning trace
- **Sign Errors, Missing Operations, Rounding Drift** — Fine-grained failure pattern analysis

---

## Models Evaluated

| Model | Parameters | Type |
|---|---|---|
| `granite3.3:8b` | 8B | General-purpose structured reasoning |
| `gemma3:4b` | 4B | Compact efficiency-focused |
| `llama3.1:8b` | 8B | General-purpose baseline |
| `mistral:7b-instruct` | 7B | Instruction-tuned |

---

## Repository Structure

```
slm-hybrid-prompting-thesis/
├── prompts/          # Prompt templates for all four methods
├── evaluation/       # Scoring and error attribution scripts
├── results/          # Output data and performance summaries
├── notebooks/        # Analysis and visualization notebooks
└── thesis/           # Final thesis PDF
```

---

## Thesis Details

| | |
|---|---|
| **Author** | Siddhardha Basu |
| **Institution** | The Pennsylvania State University, Schreyer Honors College |
| **Department** | Computer Science and Engineering |
| **Degree** | Bachelor of Science in Computer Science, with Honors |
| **Term** | Fall 2025 |
| **Thesis Supervisor** | Dr. Shagufta Mehnaz |
| **Honors Advisor** | Dr. Martin Fürer |
| **Access** | Open Access |

---

## Keywords

`Small Language Models` · `Large Language Models` · `Prompting` · `Quantitative Reasoning` · `Program-of-Thought` · `Self-Ask` · `Hybrid Prompting` · `Machine Learning` · `Artificial Intelligence` · `FinQA` · `GSM8K`

---

## Citation

```
Basu, Siddhardha. "Optimizing Quantitative Reasoning in Small Language Models 
Through Hybrid Prompting." Honors Thesis, The Pennsylvania State University, 
Schreyer Honors College, Fall 2025.
https://honors.libraries.psu.edu/catalog/9882syb5570
```