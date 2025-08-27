param(
  [string]$ckpt,
  [string]$csv = "data/processed/dataset.csv",
  [string]$target = "target",
  [string]$outdir = "logs/eval",
  [Nullable[int]]$inputDim,
  [Nullable[int]]$numClasses
)
$py = ".\.venv\Scripts\python.exe"
$argsList = @("-m","iac_core.evaluate","--checkpoint",$ckpt,"--csv",$csv,"--target-col",$target,"--outdir",$outdir)
if ($PSBoundParameters.ContainsKey("inputDim")) { $argsList += @("--input-dim",$inputDim) }
if ($PSBoundParameters.ContainsKey("numClasses")) { $argsList += @("--num-classes",$numClasses) }
& $py @argsList