from __future__ import annotations

import os

import numpy as np

OUT_DIR = os.path.dirname(__file__)


def open_area(size: int = 20) -> np.ndarray:
    return np.zeros((size, size), dtype=np.uint8)


def corridor(width: int = 20, height: int = 20) -> np.ndarray:
    grid = np.zeros((height, width), dtype=np.uint8)
    grid[5:15, 8] = 1  # left wall
    grid[5:15, 12] = 1  # right wall
    grid[10, 8:13] = 0  # opening
    return grid


def with_block(size: int = 20) -> np.ndarray:
    grid = np.zeros((size, size), dtype=np.uint8)
    grid[0:6, 8:12] = 1
    return grid


def main() -> None:
    np.save(os.path.join(OUT_DIR, "open_area.npy"), open_area())
    np.save(os.path.join(OUT_DIR, "corridor.npy"), corridor())
    np.save(os.path.join(OUT_DIR, "with_block.npy"), with_block())
    print("Saved: open_area.npy, corridor.npy, with_block.npy in", OUT_DIR)


if __name__ == "__main__":
    main()
