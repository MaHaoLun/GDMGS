import logging
import sys
from huggingface_hub import snapshot_download


def download_matrixcity(target_dir: str = "/ssddata/lun/data/matrixcity") -> None:
    """
    下载 BoDai/MatrixCity 数据集中的指定文件到目标目录。
    
    当前配置为仅下载 small_city/street/train/small_city_road_down.tar 文件。

    Args:
        target_dir (str): 下载目标目录，默认为 /ssddata/lun/data/matrixcity。

    Raises:
        RuntimeError: 下载失败时抛出。

    Example:
        >>> download_matrixcity("/ssddata/lun/data/matrixcity")
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    try:
        logging.info("开始下载 small_city/street/train/small_city_road_down.tar ...")
        snapshot_download(
            repo_id="BoDai/MatrixCity",
            repo_type="dataset",
            allow_patterns=["small_city/street/train/small_city_road_down.tar"],
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        logging.info("small_city_road_down.tar 下载完成。")
    except Exception as e:
        logging.error(f"下载失败: {e}")
        raise RuntimeError(f"下载 small_city_road_down.tar 文件失败: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="下载 BoDai/MatrixCity 的 small_city/street/train/small_city_road_down.tar 文件")
    parser.add_argument(
        "--target_dir",
        type=str,
        default="/ssddata/lun/data/matrixcity",
        help="下载目标目录 (默认: /ssddata/lun/data/matrixcity)"
    )
    args = parser.parse_args()
    download_matrixcity(args.target_dir) 
