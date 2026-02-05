import sys

import click
from loguru import logger

from imm.extractors._helper import EXTRACTORS_REGISTRY
from imm.matchers._helper import MATCHERS_REGISTRY
from imm.registry.factory import download_model_weights


@click.command()
@click.option("--name", type=str, required=False, help="Model name to download")
@click.option("--all", "download_all", is_flag=True, help="Download all models")
@click.option("--path", type=str, default="hub", help="Directory to save weights (default: hub)")
@click.help_option("--help", "-h")
def download(name, download_all, path):
    """Download pretrained model weights for IMM extractors and matchers."""

    try:
        if download_all:
            all_models = EXTRACTORS_REGISTRY.list_models + MATCHERS_REGISTRY.list_models
            logger.info(f"Downloading {len(all_models)} models...")
            success = 0
            failed = []

            for model_name in all_models:
                try:
                    if EXTRACTORS_REGISTRY.is_model(model_name):
                        cfg = EXTRACTORS_REGISTRY.get_defaultmerge_config(
                            model_name)
                    else:
                        cfg = MATCHERS_REGISTRY.get_defaultmerge_config(
                            model_name)
                    download_model_weights(model_name, cfg, path)
                    success += 1
                except Exception as e:
                    failed.append(model_name)
                    logger.error(f"Failed to download {model_name}: {e}")

            logger.success(
                f"Downloaded {success}/{len(all_models)} models to {path}")
            if failed:
                logger.warning(f"Failed: {failed}")

        else:
            if not name:
                logger.error("Please provide --name or --all")
                sys.exit(1)

            if EXTRACTORS_REGISTRY.is_model(name):
                cfg = EXTRACTORS_REGISTRY.get_defaultmerge_config(name)
            elif MATCHERS_REGISTRY.is_model(name):
                cfg = MATCHERS_REGISTRY.get_defaultmerge_config(name)
            else:
                logger.error(f"Model not found: {name}")
                sys.exit(1)

            download_model_weights(name, cfg, path)
            logger.success(f"Downloaded {name}")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)
