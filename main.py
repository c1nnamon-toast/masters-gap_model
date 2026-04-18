import sys
import gc
import logging

import torch

from logger_helper import logger_setup
from experiments.fisheye.main_fisheye \
    import main as run_fisheye
from experiments.rubsheet_square.main_rubsheet_square \
    import main as run_rubsheet_square
from experiments.rubsheet_3to1.main_rubsheet_3to1 \
    import main as run_rubsheet_3to1

logger = logging.getLogger(__name__)


def cleanup_gpu():
    """Clear GPU memory between experiments."""

    gc.collect()                    # Free Python objects (model, tensors, dataloaders)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()    # Release cached CUDA blocks
        torch.cuda.reset_peak_memory_stats()  # Reset memory stats

def run_all_experiments():
    """Run all experiments"""

    experiments = [
        ("Fisheye", run_fisheye),
        ("Rubsheet 3-to-1", run_rubsheet_3to1),
        ("Rubsheet Square", run_rubsheet_square),
    ]

    logger.info("-" * 80)
    logger.info("Starting all experiments")
    logger.info("-" * 80)

    results = {}
    for i, (exp_name, exp_main) in enumerate(experiments):
        logger.info(f'{"-" * 80}')
        logger.info(f"Experiment {i}: {exp_name}")
        logger.info("-" * 80)

        try:
            exp_main()
            results[exp_name] = "SUCCESS"
            logger.info(f"Experiment {i} ({exp_name}): completed successfully")
        except Exception as e:
            results[exp_name] = f"FAILED: {str(e)}"
            logger.error(f"Experiment {i} ({exp_name}): failed with error: {str(e)}")
            logger.exception(e)
        finally:
            # GPU clean up between experiments
            cleanup_gpu()
            logger.info(f"GPU memory released after {exp_name}")

    # Summary
    logger.info("")
    logger.info("-" * 80)
    logger.info("main Summary")
    logger.info("-" * 80)
    for exp_name, status in results.items():
        logger.info(f"{exp_name:25}: {status}")
    logger.info("-" * 80)

if __name__ == "__main__":
    logger_setup()  

    if not torch.cuda.is_available():
        logger.error("no GPU found")
        sys.exit(1)

    try:
        run_all_experiments()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.exception(e)
        sys.exit(1)
