"""
Phase 3: Combine all per-protein HDXResult pickles into summary.csv.
"""
import pickle
from pathlib import Path

from hdxrate.io import write_results_csv

output_path = Path(snakemake.output.summary)  # noqa: F821
output_path.parent.mkdir(parents=True, exist_ok=True)

results = {}
for pkl_path in snakemake.input.results:  # noqa: F821
    with open(pkl_path, 'rb') as f:
        result = pickle.load(f)
    results[result.protein_id] = result

write_results_csv(results, output_path)
print(f"Wrote {len(results)} proteins to {output_path}")
