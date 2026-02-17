### The "All-Explicit" Test Command

We use `--region` instead of `--gene` here to strictly define the 20kb window around the TSS (Transcription Start Site) and bypass the "Gene Body Exclusion" safety feature, ensuring we purposefully destroy the promoter to see the signal drop.

```bash
python main.py \
  --region chr6:41694353-41714353 \
  --tissue "Spleen" \
  --assay "RNA" \
  --mutation_size 2000 \
  --step_size 1000 \
  --window_size 196608 \
  --batch_size 1 \
  --agg_mode "mean" \
  --top_hits_scan 10 \
  --top_hits_inspect 0 \
  --out_dir ./test_results_promoter \
  --no_inspect

```

### Breakdown of Arguments (What you are telling the tool)

| Argument | Value | What it does |
| --- | --- | --- |
| **`--region`** | `chr6:41694353...` | **The Target.** We are scanning exactly 10kb upstream and 10kb downstream of the TFEB start site. |
| **`--tissue`** | `"Spleen"` | **The Filter.** We only care about expression impact in the Spleen. |
| **`--assay`** | `"RNA"` | **The Readout.** We are measuring gene expression (RNA-seq), not accessibility (ATAC). |
| **`--mutation_size`** | `2000` | **The Hammer.** We are deleting 2,000 base pairs at a time. |
| **`--step_size`** | `1000` | **The Slide.** We move the deletion window 1,000 bp forward each step (50% overlap). |
| **`--window_size`** | `196608` | **The Context.** Even though we scan 20kb, the model sees this much surrounding DNA to understand the sequence. |
| **`--batch_size`** | `1` | **The Safety.** Processes 1 mutation at a time to prevent VRAM overflow. |
| **`--agg_mode`** | `"mean"` | **The Consensus.** If there are multiple Spleen RNA tracks, we average them. |
| **`--top_hits_scan`** | `10` | **The Report.** The final CSV will list the Top 10 biggest drops. |
| **`--no_inspect`** | `(Flag)` | **The Skip.** **This is what you asked for.** It tells the tool: "Do not run the TF Inspector. Just give me the scan results." |

### What to expect

1. **Speed:** Since we are only scanning 20kb (about 20 variants), this should finish in **under 30 seconds**.
2. **Output:** You will get a single CSV file in `./test_results_promoter/`.
3. **The Result:** Look at the `impact` column. You should see a **massive positive value** (e.g., `0.04` or higher) right at the center coordinates. This confirms that deleting the promoter kills the gene.

Run this, and if you see that big drop, the tool is alive and working correctly.