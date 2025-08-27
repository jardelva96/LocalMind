param(
  [string]$model = "models/export/iazero.onnx",
  [string]$preprocess = "models/export/preprocess.json",
  [string]$bindHost = "127.0.0.1",
  [int]$port = 8000
)
$py = ".\.venv\Scripts\python.exe"
& $py -m iac_core.serve_onnx --model $model --preprocess $preprocess --host $bindHost --port $port