param(
  [string]$csv = "data/processed/dataset.csv",
  [string]$target = "target",
  [string]$out = "models/export/preprocess.json",
  [Nullable[int]]$inputDim
)
$py = ".\.venv\Scripts\python.exe"
$argsList = @("-m","iac_core.export_preprocess","--csv",$csv,"--target-col",$target,"--out",$out)
if ($PSBoundParameters.ContainsKey("inputDim")) { $argsList += @("--input-dim",$inputDim) }
& $py @argsList