from modules.benchmark_modules import LlamaOpenRewriteBenchmark
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate, call


@hydra.main(config_path="config", config_name="main")
def main(cfg: DictConfig):
    print(cfg)
    benchmark: LlamaOpenRewriteBenchmark = instantiate(cfg.llama3_open_rewrite_benchmark)
    to_benchmark = benchmark.predict(**cfg.predict)
    benchmark.evaluate(to_benchmark)


if __name__ == "__main__":
    main()