# Roda treino com overrides opcionais do Hydra
param(
  [int]$epochs = 5,
  [int]$bs = 64
)
$py = ".\.venv\Scripts\python.exe"
& $py -m iac_core.train train.max_epochs=$epochs dataset.batch_size=$bs