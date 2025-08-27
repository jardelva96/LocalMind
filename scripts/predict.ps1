param(
  [string]$ckpt = "",
  [string]$csv  = "data/processed/dataset.csv",
  [Nullable[int]]$inputDim,
  [Nullable[int]]$numClasses,
  [string]$out = "predicoes.csv"
)
$py = ".\.venv\Scripts\python.exe"
$argsList = @("-m","iac_core.predict","--csv",$csv,"--out",$out)
if ($ckpt) { $argsList += @("--checkpoint",$ckpt) }
if ($PSBoundParameters.ContainsKey("inputDim")) { $argsList += @("--input-dim",$inputDim) }
if ($PSBoundParameters.ContainsKey("numClasses")) { $argsList += @("--num-classes",$numClasses) }
& $py @argsList