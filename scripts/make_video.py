from ml_logger import logger
from pathlib import Path
from glob import glob
import os

logger.configure(root=str(os.path.dirname(os.path.abspath(__file__))), prefix=".")
frames = sorted(glob("results/*.png"))
with logger.Prefix("."):
    logger.make_video(frames, key="video.mp4", fps=10)
    
