# AlphaGenome Telescope 🔭

A high-performance, GPU-accelerated tool for performing **Saturation Mutagenesis** and **Unbiased Regulatory Discovery** using the AlphaGenome/Enformer model.

Designed for the **NVIDIA RTX 5090** (scalable to A100/H100 clusters), this tool acts as a "Genomic Telescope": it scans vast distances to find distal enhancers and automatically determines which Transcription Factors (TFs) are driving them.

## 🚀 Key Features

* **Turbo-Mode Inference:** JAX JIT compilation processes >10 variants per second.
* **Unbiased TF Discovery:** Automatically scans **5,313 genomic tracks** to identify which proteins (TFs) and histone marks are enriched at your hits—no prior guessing required.
* **Tissue-Specific Filtering:** Targets specific biological contexts (e.g., "Spleen RNA-seq") for impact calculation.
* **Smart Gene Targeting:** Auto-fetches coordinates for genes (e.g., *TFEB*) and handles strand orientation.
* **Modular Architecture:** Clean separation of Scanning (Impact) and Inspection (Mechanism).

## 🛠️ Installation

### Prerequisites

* Python 3.9+
* CUDA-enabled GPU (Minimum 24GB VRAM recommended).
* [JAX](https://github.com/google/jax) with CUDA support.

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/AlphaGenome-Telescope.git
cd AlphaGenome-Telescope

# 2. Install dependencies
pip install -r requirements.txt

```

## 📖 Usage

The tool runs in two phases automatically: **Scan** (Find the location) -> **Inspect** (Find the mechanism).

### 1. The Standard Discovery Run

Find enhancers for a gene and identify the top TFs binding there.

```bash
python main.py \
  --gene TFEB \
  --tissue Spleen \
  --assay RNA \
  --out_dir ./results/TFEB_Spleen

```

**What happens:**

1. **Scan:** The tool mutates the 1Mb region around *TFEB* to find sequences that drop expression in the Spleen.
2. **Inspect:** It takes the Top 5 hits and scans **all 5,313 tracks** in the model.
3. **Report:** It generates a CSV listing the top-ranked TFs for each enhancer (e.g., *"PU.1 (0.95); H3K27ac (0.99)"*).

### 2. Manual Region Scan

Scan an arbitrary genomic window without focusing on a specific gene.

```bash
python main.py \
  --region chr6:41700000-41800000 \
  --tissue Liver \
  --top_hits_inspect 10

```

### 3. Advanced: Force-Include Tracks

If you want to ensure a specific track is checked even if it's low-ranking:

```bash
python main.py \
  --gene TFEB \
  --inspect_tracks "CTCF,Rad21"

```

## ⚙️ Configuration Options

| Flag | Description | Default |
| --- | --- | --- |
| `--gene` | Target Gene Symbol (Auto-fetches coords). | `None` |
| `--region` | Manual coordinates (`chr:start-end`). | `None` |
| `--tissue` | Tissue filter for Impact calculation. | `Spleen` |
| `--assay` | Assay type for Impact (`RNA`, `ATAC`). | `RNA` |
| `--window_size` | Context window size (bp). | `196608` |
| `--exclude_gene_body` | Skip mutations inside the gene itself. | `True` |
| `--top_hits_inspect` | Number of enhancers to deep-scan for TFs. | `5` |
| `--batch_size` | GPU Batch Size (Adjust for VRAM). | `1` |

## 📊 Output Files

1. **`_scan.csv`**: The raw mutagenesis data.
* *Columns:* `chrom`, `start`, `end`, `impact_score`, `distance_to_tss`


2. **`_scan.bedgraph`**: Visualization track.
* *Usage:* Drag and drop into IGV to see the "Impact Peaks."


3. **`_inspection_report.csv`**: The mechanism report.
* *Columns:* `location`, `type` (Promoter/Distal), `top_regulatory_signals` (The ranked list of TFs).



## 🤝 Contributing

Contributions are welcome! Please open an issue for bug reports or feature requests.

## 📄 License

MIT License.