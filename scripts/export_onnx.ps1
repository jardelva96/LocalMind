param(
  [string]$ckpt,
  [string]$out = "models/export/model.onnx",
  [Nullable[int]]$inputDim,
  [Nullable[int]]$numClasses
)
$py = ".\.venv\Scripts\python.exe"
$argsList = @("-m","iac_core.export_onnx","--checkpoint",$ckpt,"--out",$out)
if ($PSBoundParameters.ContainsKey("inputDim")) { $argsList += @("--input-dim",$inputDim) }
if ($PSBoundParameters.ContainsKey("numClasses")) { $argsList += @("--num-classes",$numClasses) }
& $py @argsList