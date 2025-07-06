from lightning.pytorch.callbacks import ModelCheckpoint
from ml_template.models.module import BinaryClassifier
from ml_template.data.datamodule import DataModule
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from huggingface_hub import HfApi
from dotenv import load_dotenv
from typing import Optional

import lightning as L
import hydra
import os


@hydra.main(config_path="./conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    print(
        "############################### CONFIGURATION ###############################"
    )
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print(
        "#############################################################################"
    )
    print(
        f"Hydra Runtime Output Directory: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )
    print(
        "#############################################################################"
    )
    cfg = cfg.experiment
    checkpoints_dir = os.path.join(".", "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    seed: Optional[int] = cfg.get("seed")

    if seed:
        L.seed_everything(seed, workers=True)

    datamodule = DataModule(**cfg.datamodule)
    # TODO: use your custom LightningModule
    module = BinaryClassifier(**cfg.module)
    trainer: L.Trainer = instantiate(cfg.trainer)

    if cfg.get("run_fit", True):
        trainer.fit(model=module, datamodule=datamodule)
    if cfg.get("run_test", True):
        ckpt_path: Optional[str] = None
        if cfg.get("test_ckpt_path", None):
            ckpt_path = cfg.test_ckpt_path
        if cfg.get("run_fit", True):
            # UNCOMMENT THIS FOR THE HUGGING FACE INTEGRATION
            # primary_checkpoint_cb = trainer.checkpoint_callback
            # if isinstance(primary_checkpoint_cb, ModelCheckpoint):
            #     if hasattr(primary_checkpoint_cb, "best_model_path"):
            #         ckpt_path = primary_checkpoint_cb.best_model_path
            #         load_dotenv()
            #         token = os.getenv("HF_TOKEN")
            #         api = HfApi(token=token)
            #         repo_id = f"{cfg.dev_org_name}/{cfg.project_name}"
            #         base_repo_path = f"checkpoints/{cfg.experiment_name}.ckpt"
            #         api.upload_file(
            #             path_or_fileobj=ckpt_path,
            #             path_in_repo=base_repo_path,
            #             repo_id=repo_id,
            #             repo_type="model",
            #         )
            ckpt_path = "best"
        if ckpt_path is None:
            print("TEST SKIPPED: no training was done and no checkpoint was provided.")
        else:
            test_results = trainer.test(
                model=module, datamodule=datamodule, ckpt_path=ckpt_path
            )
            monitor_metric = cfg.monitor_metric
            optimized_metric = test_results[0].get(monitor_metric)
            return optimized_metric


if __name__ == "__main__":
    train()
