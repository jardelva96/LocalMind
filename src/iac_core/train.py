from __future__ import annotations
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from hydra import main as hydra_main
from omegaconf import DictConfig, OmegaConf

from iac_core.utils.seed import set_seed
from iac_core.data.loader import SyntheticConfig, SyntheticClassifDM, CSVConfig, CSVClassifDM
from iac_core.models.mlp import MLPClassifier

@hydra_main(version_base=None, config_path="../../configs", config_name="config")
def run(cfg: DictConfig) -> None:
    print("== CONFIG ==")
    print(OmegaConf.to_yaml(cfg, resolve=False))

    set_seed(int(cfg.seed))

    # --- escolhe dataset e instancia DM ---
    ds_name = str(getattr(cfg.dataset, "name", "synthetic_classif"))
    ds_dict = OmegaConf.to_container(cfg.dataset, resolve=True)
    assert isinstance(ds_dict, dict)

    if ds_name == "csv_classif":
        ds_dict.pop("name", None)
        dcfg = CSVConfig(**ds_dict)
        dm = CSVClassifDM(dcfg)
    else:
        ds_dict.pop("name", None)
        dcfg = SyntheticConfig(**ds_dict)
        dm = SyntheticClassifDM(dcfg)

    # --- prepara para descobrir dimensões dinamicamente ---
    dm.setup("fit")
    n_features = getattr(dm, "n_features", None)
    n_classes  = getattr(dm, "n_classes",  None)

    # fallback (ex.: sintético antes de setar atributos)
    if n_features is None or n_classes is None:
        ds_cont = OmegaConf.to_container(cfg.dataset, resolve=True)
        if isinstance(ds_cont, dict):
            n_features = n_features or ds_cont.get("n_features")
            n_classes  = n_classes  or ds_cont.get("n_classes")

    if n_features is None or n_classes is None:
        raise ValueError("Não foi possível inferir input_dim/num_classes. Verifique o dataset ou informe via config.")

    # --- modelo (não lê cfg.model.input_dim/num_classes para evitar interpolação) ---
    mcfg = cfg.model
    model = MLPClassifier(
        input_dim=int(n_features),
        hidden_dims=[int(h) for h in list(mcfg.hidden_dims)],
        num_classes=int(n_classes),
        lr=float(mcfg.lr),
        weight_decay=float(mcfg.weight_decay),
        dropout=float(getattr(mcfg, "dropout", 0.0)),
    )

    # --- loggers e callbacks ---
    log_dir = Path.cwd() / "logs"
    logger = CSVLogger(save_dir=str(log_dir), name="iazero", flush_logs_every_n_steps=50)
    ckpt_cb = ModelCheckpoint(
        monitor="val/acc", mode="max",
        save_top_k=1, filename="best-epoch{epoch:02d}-valacc{val/acc:.3f}",
        auto_insert_metric_name=False
    )
    early_cb = EarlyStopping(monitor="val/acc", mode="max", patience=5)

    # --- treinador ---
    tcfg = cfg.train
    trainer = pl.Trainer(
        max_epochs=int(tcfg.max_epochs),
        accelerator=str(tcfg.accelerator),
        devices=int(tcfg.devices),
        precision=str(tcfg.precision),
        log_every_n_steps=int(tcfg.log_every_n_steps),
        enable_progress_bar=bool(tcfg.enable_progress_bar),
        default_root_dir=str(Path.cwd()),
        logger=logger,
        callbacks=[ckpt_cb, early_cb],
    )

    trainer.fit(model, datamodule=dm)
    test_metrics = trainer.test(model, datamodule=dm)
    print("\\nMelhor checkpoint:", ckpt_cb.best_model_path or "(não salvo)")
    if test_metrics:
        print("Métricas de teste:", test_metrics[0])

if __name__ == "__main__":
    run()