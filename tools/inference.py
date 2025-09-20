import hydra
from omegaconf import DictConfig, OmegaConf

from wildlifemapper.inference import InferenceRunner


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Initialize and run inference
    runner = InferenceRunner(cfg)
    test_stats, coco_evaluator = runner.run_inference()


if __name__ == "__main__":
    main()