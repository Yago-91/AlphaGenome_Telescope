# AlphaGenome_Telescope 🔭


A high-performance, GPU-accelerated tool for performing **Saturation Mutagenesis** and **Deep Learning-based Regulatory Analysis** using the AlphaGenome/Enformer model.

Designed for the **NVIDIA RTX 5090** (and scalable to A100/H100 clusters), this tool allows researchers to:

1. **Scan** megabase-scale genomic windows to identify distal enhancers.
2. **Inspect** specific regions for transcription factor binding (TF) and chromatin states.
3. **Map** the regulatory landscape of any target gene in any tissue (e.g., Spleen, Liver, Brain).

## 🚀 Key Features

* **Turbo-Mode Inference:** Uses JAX JIT compilation to process >10 variants per second on consumer hardware.
* **Tissue-Specific Filtering:** Automatically targets specific biological contexts (e.g., "Spleen RNA-seq" or "Monocyte ATAC-seq").
* **Smart Gene Targeting:** Auto-fetches coordinates for target genes (e.g., *TFEB*) and handles strand orientation automatically.
* **Modular Architecture:** Clean separation of Scanning (Impact Calculation) and Inspection (Mechanism of Action).
* **Multi-Format Output:** Generates CSVs for data analysis and `.bedgraph` tracks for visualization in IGV/UCSC Genome Browser.

## 🛠️ Installation

### Prerequisites

* Python 3.9+
* CUDA-enabled GPU (Minimum 24GB VRAM recommended for default settings).
* [JAX](https://github.com/google/jax) with CUDA support.

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/AlphaGenome-Telescope.git
cd AlphaGenome-Telescope

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Set up the AlphaGenome model weights
# Ensure your 'alphagenome_research' package is in the PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/path/to/alphagenome_research

```

## 📖 Usage

The tool is run via the `main.py` entry point. It supports two primary modes of operation which run sequentially by default.

### 1. The Standard Scan (Discovery Mode)

Find enhancers for a specific gene in a specific tissue.

```bash
python main.py \
  --gene TFEB \
  --tissue Spleen \
  --assay RNA \
  --out_dir ./results/TFEB_Spleen

```

**Output:**

* `TFEB_Spleen_RNA_scan.csv`: Ranked list of all mutations and their impact scores.
* `TFEB_Spleen_RNA_scan.bedgraph`: Visual track of enhancer locations.

### 2. The Deep Dive (Inspector Mode)

If you want to understand *why* a region is an enhancer, add the `--inspect_tracks` flag.

```bash
python main.py \
  --gene TFEB \
  --tissue Spleen \
  --inspect_tracks "PU.1,CTCF,H3K27ac" \
  --top_hits_inspect 10

```

This will generate an additional **Inspection Report** detailing the specific transcription factor binding probabilities for your top 10 hits.

### 3. Global/Manual Region Scan

If you are not focusing on a specific gene, you can scan arbitrary genomic coordinates.

```bash
python main.py \
  --region chr6:41700000-41800000 \
  --tissue Liver \
  --window_size 100000

```

## ⚙️ Configuration Options

| Flag | Description | Default |
| --- | --- | --- |
| `--gene` | Target Gene Symbol (Auto-fetches coords). | `None` |
| `--region` | Manual coordinates (`chr:start-end`). | `None` |
| `--tissue` | Tissue filter for output tracks. | `Spleen` |
| `--assay` | Assay type (`RNA`, `ATAC`, `DNASE`, `CHIP`). | `RNA` |
| `--window_size` | Context window size (bp). | `196608` |
| `--mutation_size` | Size of the deletion/mask (bp). | `2000` |
| `--step_size` | Sliding window step (bp). | `1000` |
| `--batch_size` | GPU Batch Size (Adjust for VRAM). | `1` |
| `--exclude_gene_body` | Skip mutations inside the gene itself. | `True` |

## 📊 Interpreting Results

### Impact Scores

* **Positive Score (+):** The mutation *decreased* expression. (Likely an Enhancer).
* **Negative Score (-):** The mutation *increased* expression. (Likely a Repressor/Silencer).
* **Magnitude:** Represented in log-scale. A score of `0.002` typically represents a ~10% change in total regional expression.

### Visualization

Drag and drop the generated `.bedgraph` file into [IGV](https://software.broadinstitute.org/software/igv/) to see the "Impact Landscape." Peaks indicate the location of critical regulatory elements.

## 🤝 Contributing

Contributions are welcome! Please open an issue if you encounter bugs with specific GPU configurations or model weights.

## 📄 License

MIT License.

