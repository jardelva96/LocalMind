param(
  [string]$out = "data/processed/dataset.csv",
  [int]$samples = 1000,
  [int]$features = 32,
  [int]$informative = 16,
  [int]$classes = 4,
  [int]$seed = 42
)
$py = ".\.venv\Scripts\python.exe"
& $py .\scripts\make_dataset_csv.py --out $out --samples $samples --features $features --informative $informative --classes $classes --seed $seed