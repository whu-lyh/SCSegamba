
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layernorm import RMSNorm
from mmcv.cnn.bricks.transformer import build_dropout
from pyzorder import ZOrderIndexer
from torch.distributions.normal import Normal

from mmcls.SAVSS_dev.models.SAVSS.moe_layer import (FeedForward, SwitchGate,
                                                    SwitchGate_Conv)
from models.GBC import GBC, BottConv
from models.PAF import PAF
from util import hilbert


class SerializationStrategies_base(nn.Module):
    def __init__(self):
        super().__init__()

    def Parallel_snake_horizontal(self, hw_shape):
        """
        Generate a horizontal snake-like traversal order for an H×W grid.

        This function constructs a “snake” (zig-zag) scanning order over a 2D grid,
        starting from the bottom row. The traversal direction alternates between
        left-to-right and right-to-left for each row, producing a continuous
        one-dimensional index sequence. It also computes the inverse mapping from
        grid index to its position in the snake order.

        Args:
            hw_shape (tuple): A tuple (H, W) specifying the grid height and width.

        Returns:
            tuple:
                o1 (list[int]): A list of length H*W containing the linear
                    indices of the grid visited in snake order.
                o1_inverse (list[int]): A list of length H*W where each entry
                    gives the position of the corresponding grid index in `o1`.
                    For index k in the grid, `o1_inverse[k]` is its order in the
                    snake traversal.
        """
        H, W = hw_shape
        L = H * W
        o1 = []
        d1 = []
        o1_inverse = [-1 for _ in range(L)]

        if H % 2 == 1:
            i, j = H - 1, W - 1
            j_d = "left"
        else:
            i, j = H - 1, 0
            j_d = "right"

        while i > -1:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W - 1:
                    j = j + 1
                    d1.append(1)
                else:
                    i = i - 1
                    d1.append(3)
                    j_d = "left"
            else:
                if j > 0:
                    j = j - 1
                    d1.append(2)
                else:
                    i = i - 1
                    d1.append(3)
                    j_d = "right"
        d1 = [0] + d1[:-1]

        return o1, o1_inverse

    def Parallel_snake_horizontal2(self, hw_shape):
        """
        Generate a horizontal serpentine (snake-like) scan order starting from
        the top-left corner of the grid.

        This method performs a row-wise traversal beginning at (0, 0). For each row,
        the scanning direction alternates: even-indexed rows move left-to-right,
        while odd-indexed rows move right-to-left. The traversal continues until all
        H × W grid elements are visited. Direction codes are internally recorded
        but not returned, except for the forward scan order.

        Args:
            hw_shape (tuple[int, int]):
                A tuple (H, W) specifying the grid height and width.

        Returns:
            tuple[list[int], list[int]]:
                - o1: Forward scan order following the horizontal serpentine traversal.
                    Each element corresponds to a flattened grid index i * W + j.
                - o1_inverse: Inverse mapping array where o1_inverse[idx] gives the
                            position of index `idx` in the serialized forward path.
                            This enables O(1) lookup from grid index to traversal order.
        """
        H, W = hw_shape
        L = H * W

        o1 = []
        d1 = []
        o1_inverse = [-1 for _ in range(L)]
        i, j = 0, 0
        j_d = "right"
        while i < H:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W-1:
                    j = j + 1
                    d1.append(1)
                else:
                    i = i + 1
                    d1.append(4)
                    j_d = "left"
            else:
                if j > 0:
                    j = j - 1
                    d1.append(2)
                else:
                    i = i + 1
                    d1.append(4)
                    j_d = "right"
        d1 = [0] + d1[:-1]

        return o1, o1_inverse

    def Parallel_snake_vertical(self, hw_shape):
        """
        Generate a vertical snake-like traversal order for an H×W grid.

        This function constructs a vertical “snake” (zig-zag) scanning order over
        a 2D grid. The traversal proceeds column by column: in even-indexed
        columns it scans from top to bottom, and in odd-indexed columns it scans
        bottom to top. This produces a continuous 1D index sequence. The function
        also returns the inverse mapping from each grid index to its position
        in the snake traversal.

        Args:
            hw_shape (tuple):
                A tuple (H, W) specifying the grid height and width.

        Returns:
            tuple:
                o1 (list[int]):
                    A list of length H*W containing the linear indices of all
                    grid cells visited in vertical snake order.
                o1_inverse (list[int]):
                    A list of length H*W that maps each grid index to its order
                    in the snake traversal. For any index k, `o1_inverse[k]`
                    gives its position within `o1`.
        """
        H, W = hw_shape
        L = H * W
        o1 = []
        d1 = []
        o1_inverse = [-1 for _ in range(L)]

        i, j = 0, 0
        i_d = "down"
        while j < W:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                    d1.append(4)
                else:
                    j = j + 1
                    d1.append(1)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d1.append(3)
                else:
                    j = j + 1
                    d1.append(1)
                    i_d = "down"
        d1 = [0] + d1[:-1]

        return o1, o1_inverse

    def Diagonal_snake_left(self, hw_shape):
        """
        Generate a diagonal snake-like traversal order for an H×W grid (left-aligned).

        This function constructs a diagonal zig-zag traversal over a 2D grid along
        all anti-diagonals (i.e., where i + j = constant). The traversal direction
        alternates between diagonals: even-indexed diagonals iterate in one order
        (i-first), and odd-indexed diagonals iterate in the opposite order
        (j-first). This produces a continuous snake-like sequence without
        horizontal mirroring (i.e., left-aligned). The function also computes
        the inverse mapping that records the position of each grid index in the
        traversal sequence.

        Args:
            hw_shape (tuple):
                A tuple (H, W) specifying the grid height (H) and width (W).

        Returns:
            tuple:
                o1 (list[int]):
                    A list of length H*W containing the linear indices of all grid
                    elements visited in diagonal snake order.
                o1_inverse (list[int]):
                    A list of length H*W where each entry gives the position
                    of the corresponding grid index in `o1`. For index k,
                    `o1_inverse[k]` returns its traversal order.
        """
        H, W = hw_shape
        L = H * W
        o1 = []
        d1 = []
        o1_inverse = [-1 for _ in range(L)]

        # Diagonal route
        for diag in range(H + W - 1):
            if diag % 2 == 0:
                # Even diagonal: iterate by i first
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + j
                        o1.append(idx)
                        o1_inverse[idx] = len(o1) - 1
                        d1.append(1 if j == diag else 4)
            else:
                # Odd diagonal: iterate by j first
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + j
                        o1.append(idx)
                        o1_inverse[idx] = len(o1) - 1
                        d1.append(4 if i == diag else 1)
        d1 = [0] + d1[:-1]

        return o1, o1_inverse

    def Diagonal_snake_right(self, hw_shape):
        """
        Generate a diagonal snake-like traversal order for an H×W grid (right-aligned).

        This function constructs a diagonal “snake” scanning order over a 2D grid.
        The traversal proceeds along all anti-diagonals of the grid (i.e., lines
        where i + j = constant). For each diagonal, the direction alternates:
        even-indexed diagonals are visited in one orientation, and odd-indexed
        diagonals in the opposite orientation. The traversal is additionally
        mirrored horizontally (right-aligned), meaning that the column index is
        transformed as (W - j - 1). This produces a continuous 1D index sequence.
        The function also computes the inverse mapping from each grid index to
        its position in this diagonal snake traversal.

        Args:
            hw_shape (tuple):
                A tuple (H, W) specifying the grid height and width.

        Returns:
            tuple:
                o1 (list[int]):
                    A list of length H*W containing the linear grid indices in
                    the diagonal snake traversal order.
                o1_inverse (list[int]):
                    A list of length H*W mapping each grid index to its
                    occurrence position within `o1`.
                    For index k, `o1_inverse[k]` gives the traversal rank.
        """
        H, W = hw_shape
        L = H * W
        o1 = []
        d1 = []
        o1_inverse = [-1 for _ in range(L)]

        for diag in range(H + W - 1):
            if diag % 2 == 0:
                # Even diagonals: iterate i first
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + (W - j - 1)
                        o1.append(idx)
                        o1_inverse[idx] = len(o1) - 1
                        d1.append(1 if j == diag else 4)
            else:
                # Odd diagonals: iterate j first
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + (W - j - 1)
                        o1.append(idx)
                        o1_inverse[idx] = len(o1) - 1
                        d1.append(4 if i == diag else 1)
        d1 = [0] + d1[:-1]

        return o1, o1_inverse

    def zigzag(self, hw_shape):
        """
        Generate a zigzag traversal order for an H×W grid (classic JPEG-style).

        This function computes a zigzag (diagonal sweep) ordering over a 2D grid.
        The traversal follows the conventional pattern used in JPEG block
        processing: the grid is visited along diagonals of length varying between
        1 and min(H, W), and the direction alternates between each diagonal.
        This produces a continuous sequence that preserves local spatial
        relationships more effectively than simple raster scanning.

        Args:
            hw_shape (tuple):
                A tuple (H, W) specifying the grid height and width.

        Returns:
            tuple:
                o1 (list[int]):
                    A list of length H*W containing the linear grid indices
                    visited in zigzag order.
                o1_inverse (list[int]):
                    A list of length H*W where each element gives the position
                    of the corresponding linear index in the zigzag traversal.
                    For index k, `o1_inverse[k]` returns its zigzag rank.
        """
        H, W = hw_shape
        o1 = []

        # Diagonal zigzag traversal
        # Total number of diagonals = H + W - 1
        for diag in range(H + W - 1):
            # Compute valid row/col ranges on this diagonal
            row_start = max(0, diag - (W - 1))
            row_end   = min(diag, H - 1)
            if diag % 2 == 0:
                # Even diag → traverse from high row to low row:
                # (r decreases, c increases)
                for r in range(row_end, row_start - 1, -1):
                    c = diag - r
                    o1.append(r * W + c)
            else:
                # Odd diag → traverse from low row to high row:
                # (r increases, c decreases)
                for r in range(row_start, row_end + 1):
                    c = diag - r
                    o1.append(r * W + c)

        o1 = np.array(o1)
        o1_inverse = np.argsort(o1)

        return o1.tolist(), o1_inverse.tolist()

    def zorder(self, hw_shape):
        """
        Generate a Z-order (Morton order) traversal sequence for an H×W grid.

        This function computes the Morton (Z-order) curve indexing for a 2D grid.
        Z-order is a space-filling curve that interleaves the bit representations
        of row and column coordinates to preserve spatial locality. Given a grid
        of shape H×W, the function returns both the forward Z-order sequence and
        its inverse mapping.

        Args:
            hw_shape (tuple):
                A tuple (H, W) specifying the grid height and width.

        Returns:
            tuple:
                o1 (list[int]):
                    A list of length H*W containing the Morton-order linear
                    indices. Each entry corresponds to a grid cell visited in
                    Z-order.
                o1_inverse (list[int]):
                    A list of length H*W where each element gives the position
                    of the corresponding linear index in the Z-order sequence.
                    For index k, `o1_inverse[k]` returns its Morton traversal rank.
        """
        H, W = hw_shape
        indexes = np.arange(H * W)
        zi = ZOrderIndexer((0, H - 1), (0, W - 1))
        o1 = []

        for z in indexes:
            r, c = zi.rc(int(z))
            o1.append(r * H + c)

        o1 = np.array(o1)
        o1_inverse = np.argsort(o1)

        return o1.tolist(), o1_inverse.tolist()

    def scan(self, hw_shape):
        """
        Generate a scan-line traversal order with alternating row directions.

        This serialization method performs a horizontal scan over the grid but
        reverses every second row to create a snake-like left-to-right then
        right-to-left pattern. Unlike `parallel_snake_horizontal`, which may start
        scanning from the bottom or top depending on grid parity, this method always
        starts from the top-left corner and alternates direction strictly based on
        row index parity.

        Args:
            hw_shape (tuple[int, int]):
                A tuple (H, W) representing the grid height and width.

        Returns:
            tuple[list[int], list[int]]:
                - o1: A list of indices representing the forward scan traversal
                    following the alternating-direction pattern.
                - o1_inverse: A list where each position stores the inverse mapping,
                            such that `o1_inverse[o1[k]] = k`, enabling fast lookup
                            of the position of each grid index in the serialized order.
        """
        H, W = hw_shape
        indexes = np.arange(H * W)
        indexes = indexes.reshape(H, W)
        for i in np.arange(1, H, step=2):
            indexes[i, :] = indexes[i, :][::-1]
        o1 = indexes.reshape(-1)

        o1 = np.array(o1)
        o1_inverse = np.argsort(o1)

        return o1.tolist(), o1_inverse.tolist()

    def hilbert(self, hw_shape):
        H, W = hw_shape
        indexes = np.arange(H * W)
        bit = int(math.log2(H))
        locs = hilbert.decode(torch.tensor(indexes), 2, bit)
        ret = []
        l = 2 ** bit
        for i in range(len(locs)):
            loc = locs[i]
            loc_flat = 0
            for j in range(2):
                loc_flat += loc[j] * (l ** j)
            ret.append(loc_flat)

        o1 = np.array(ret).astype(np.uint64)
        o1_inverse = np.argsort(o1)

        return o1.tolist(), o1_inverse.tolist()
    
    def forward(self, hw_shape):
        # o used to get corresponding serialization index
        # o_inverse used to transform the output sequences back to the original order
        o, o_inverse = [], []
        for ss in [self.Parallel_snake_horizontal2(hw_shape),
                    self.Parallel_snake_vertical(hw_shape), 
                   self.Diagonal_snake_left(hw_shape), 
                   self.zigzag(hw_shape)]:
            o_seq, o_inv_seq = ss
            o.append(o_seq)
            o_inverse.append(o_inv_seq)
        return o, o_inverse


@torch.no_grad()
class SerializationStrategies:
    """Collection of serialization functions.

        Each function accepts a tensor of shape [B, seq_len, C]
        and returns (serialized_tensor, metadata) where metadata contains information
        needed to reverse the serialization (original lengths / shapes / indices).
    """

    @staticmethod
    def Parallel_snake_horizontal(hw_shape):
        """
        Generate a horizontal snake-like traversal order for an H×W grid.

        This function constructs a “snake” (zig-zag) scanning order over a 2D grid,
        starting from the bottom row. The traversal direction alternates between
        left-to-right and right-to-left for each row, producing a continuous
        one-dimensional index sequence. It also computes the inverse mapping from
        grid index to its position in the snake order.

        Args:
            hw_shape (tuple): A tuple (H, W) specifying the grid height and width.

        Returns:
            tuple:
                o1 (list[int]): A list of length H*W containing the linear
                    indices of the grid visited in snake order.
                o1_inverse (list[int]): A list of length H*W where each entry
                    gives the position of the corresponding grid index in `o1`.
                    For index k in the grid, `o1_inverse[k]` is its order in the
                    snake traversal.
        """
        H, W = hw_shape
        L = H * W
        o1 = []
        d1 = []
        o1_inverse = [-1 for _ in range(L)]

        if H % 2 == 1:
            i, j = H - 1, W - 1
            j_d = "left"
        else:
            i, j = H - 1, 0
            j_d = "right"

        while i > -1:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W - 1:
                    j = j + 1
                    d1.append(1)
                else:
                    i = i - 1
                    d1.append(3)
                    j_d = "left"
            else:
                if j > 0:
                    j = j - 1
                    d1.append(2)
                else:
                    i = i - 1
                    d1.append(3)
                    j_d = "right"
        d1 = [0] + d1[:-1]

        return o1, o1_inverse

    @staticmethod
    def Parallel_snake_horizontal2(hw_shape):
        """
        Generate a horizontal serpentine (snake-like) scan order starting from
        the top-left corner of the grid.

        This method performs a row-wise traversal beginning at (0, 0). For each row,
        the scanning direction alternates: even-indexed rows move left-to-right,
        while odd-indexed rows move right-to-left. The traversal continues until all
        H × W grid elements are visited. Direction codes are internally recorded
        but not returned, except for the forward scan order.

        Args:
            hw_shape (tuple[int, int]):
                A tuple (H, W) specifying the grid height and width.

        Returns:
            tuple[list[int], list[int]]:
                - o1: Forward scan order following the horizontal serpentine traversal.
                    Each element corresponds to a flattened grid index i * W + j.
                - o1_inverse: Inverse mapping array where o1_inverse[idx] gives the
                            position of index `idx` in the serialized forward path.
                            This enables O(1) lookup from grid index to traversal order.
        """
        H, W = hw_shape
        L = H * W

        o1 = []
        d1 = []
        o1_inverse = [-1 for _ in range(L)]
        i, j = 0, 0
        j_d = "right"
        while i < H:
            assert j_d in ["right", "left"]
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < W-1:
                    j = j + 1
                    d1.append(1)
                else:
                    i = i + 1
                    d1.append(4)
                    j_d = "left"
            else:
                if j > 0:
                    j = j - 1
                    d1.append(2)
                else:
                    i = i + 1
                    d1.append(4)
                    j_d = "right"
        d1 = [0] + d1[:-1]

        return o1, o1_inverse

    @staticmethod
    def Parallel_snake_vertical(hw_shape):
        """
        Generate a vertical snake-like traversal order for an H×W grid.

        This function constructs a vertical “snake” (zig-zag) scanning order over
        a 2D grid. The traversal proceeds column by column: in even-indexed
        columns it scans from top to bottom, and in odd-indexed columns it scans
        bottom to top. This produces a continuous 1D index sequence. The function
        also returns the inverse mapping from each grid index to its position
        in the snake traversal.

        Args:
            hw_shape (tuple):
                A tuple (H, W) specifying the grid height and width.

        Returns:
            tuple:
                o1 (list[int]):
                    A list of length H*W containing the linear indices of all
                    grid cells visited in vertical snake order.
                o1_inverse (list[int]):
                    A list of length H*W that maps each grid index to its order
                    in the snake traversal. For any index k, `o1_inverse[k]`
                    gives its position within `o1`.
        """
        H, W = hw_shape
        L = H * W
        o1 = []
        d1 = []
        o1_inverse = [-1 for _ in range(L)]

        i, j = 0, 0
        i_d = "down"
        while j < W:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if i_d == "down":
                if i < H - 1:
                    i = i + 1
                    d1.append(4)
                else:
                    j = j + 1
                    d1.append(1)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d1.append(3)
                else:
                    j = j + 1
                    d1.append(1)
                    i_d = "down"
        d1 = [0] + d1[:-1]

        return o1, o1_inverse
    
    @staticmethod
    def Diagonal_snake_left(hw_shape):
        """
        Generate a diagonal snake-like traversal order for an H×W grid (left-aligned).

        This function constructs a diagonal zig-zag traversal over a 2D grid along
        all anti-diagonals (i.e., where i + j = constant). The traversal direction
        alternates between diagonals: even-indexed diagonals iterate in one order
        (i-first), and odd-indexed diagonals iterate in the opposite order
        (j-first). This produces a continuous snake-like sequence without
        horizontal mirroring (i.e., left-aligned). The function also computes
        the inverse mapping that records the position of each grid index in the
        traversal sequence.

        Args:
            hw_shape (tuple):
                A tuple (H, W) specifying the grid height (H) and width (W).

        Returns:
            tuple:
                o1 (list[int]):
                    A list of length H*W containing the linear indices of all grid
                    elements visited in diagonal snake order.
                o1_inverse (list[int]):
                    A list of length H*W where each entry gives the position
                    of the corresponding grid index in `o1`. For index k,
                    `o1_inverse[k]` returns its traversal order.
        """
        H, W = hw_shape
        L = H * W
        o1 = []
        d1 = []
        o1_inverse = [-1 for _ in range(L)]

        # Diagonal route
        for diag in range(H + W - 1):
            if diag % 2 == 0:
                # Even diagonal: iterate by i first
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + j
                        o1.append(idx)
                        o1_inverse[idx] = len(o1) - 1
                        d1.append(1 if j == diag else 4)
            else:
                # Odd diagonal: iterate by j first
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + j
                        o1.append(idx)
                        o1_inverse[idx] = len(o1) - 1
                        d1.append(4 if i == diag else 1)
        d1 = [0] + d1[:-1]

        return o1, o1_inverse
    
    @staticmethod
    def Diagonal_snake_right(hw_shape):
        """
        Generate a diagonal snake-like traversal order for an H×W grid (right-aligned).

        This function constructs a diagonal “snake” scanning order over a 2D grid.
        The traversal proceeds along all anti-diagonals of the grid (i.e., lines
        where i + j = constant). For each diagonal, the direction alternates:
        even-indexed diagonals are visited in one orientation, and odd-indexed
        diagonals in the opposite orientation. The traversal is additionally
        mirrored horizontally (right-aligned), meaning that the column index is
        transformed as (W - j - 1). This produces a continuous 1D index sequence.
        The function also computes the inverse mapping from each grid index to
        its position in this diagonal snake traversal.

        Args:
            hw_shape (tuple):
                A tuple (H, W) specifying the grid height and width.

        Returns:
            tuple:
                o1 (list[int]):
                    A list of length H*W containing the linear grid indices in
                    the diagonal snake traversal order.
                o1_inverse (list[int]):
                    A list of length H*W mapping each grid index to its
                    occurrence position within `o1`.
                    For index k, `o1_inverse[k]` gives the traversal rank.
        """
        H, W = hw_shape
        L = H * W
        o1 = []
        d1 = []
        o1_inverse = [-1 for _ in range(L)]

        for diag in range(H + W - 1):
            if diag % 2 == 0:
                # Even diagonals: iterate i first
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + (W - j - 1)
                        o1.append(idx)
                        o1_inverse[idx] = len(o1) - 1
                        d1.append(1 if j == diag else 4)
            else:
                # Odd diagonals: iterate j first
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + (W - j - 1)
                        o1.append(idx)
                        o1_inverse[idx] = len(o1) - 1
                        d1.append(4 if i == diag else 1)
        d1 = [0] + d1[:-1]

        return o1, o1_inverse

    @staticmethod
    def zigzag(hw_shape):
        """
        Generate a zigzag traversal order for an H×W grid (classic JPEG-style).

        This function computes a zigzag (diagonal sweep) ordering over a 2D grid.
        The traversal follows the conventional pattern used in JPEG block
        processing: the grid is visited along diagonals of length varying between
        1 and min(H, W), and the direction alternates between each diagonal.
        This produces a continuous sequence that preserves local spatial
        relationships more effectively than simple raster scanning.

        Args:
            hw_shape (tuple):
                A tuple (H, W) specifying the grid height and width.

        Returns:
            tuple:
                o1 (list[int]):
                    A list of length H*W containing the linear grid indices
                    visited in zigzag order.
                o1_inverse (list[int]):
                    A list of length H*W where each element gives the position
                    of the corresponding linear index in the zigzag traversal.
                    For index k, `o1_inverse[k]` returns its zigzag rank.
        """
        H, W = hw_shape
        o1 = []

        # Diagonal zigzag traversal
        # Total number of diagonals = H + W - 1
        for diag in range(H + W - 1):
            # Compute valid row/col ranges on this diagonal
            row_start = max(0, diag - (W - 1))
            row_end   = min(diag, H - 1)
            if diag % 2 == 0:
                # Even diag → traverse from high row to low row:
                # (r decreases, c increases)
                for r in range(row_end, row_start - 1, -1):
                    c = diag - r
                    o1.append(r * W + c)
            else:
                # Odd diag → traverse from low row to high row:
                # (r increases, c decreases)
                for r in range(row_start, row_end + 1):
                    c = diag - r
                    o1.append(r * W + c)

        o1 = np.array(o1)
        o1_inverse = np.argsort(o1)

        return o1.tolist(), o1_inverse.tolist()

    @staticmethod
    def zorder(hw_shape):
        """
        Generate a Z-order (Morton order) traversal sequence for an H×W grid.

        This function computes the Morton (Z-order) curve indexing for a 2D grid.
        Z-order is a space-filling curve that interleaves the bit representations
        of row and column coordinates to preserve spatial locality. Given a grid
        of shape H×W, the function returns both the forward Z-order sequence and
        its inverse mapping.

        Args:
            hw_shape (tuple):
                A tuple (H, W) specifying the grid height and width.

        Returns:
            tuple:
                o1 (list[int]):
                    A list of length H*W containing the Morton-order linear
                    indices. Each entry corresponds to a grid cell visited in
                    Z-order.
                o1_inverse (list[int]):
                    A list of length H*W where each element gives the position
                    of the corresponding linear index in the Z-order sequence.
                    For index k, `o1_inverse[k]` returns its Morton traversal rank.
        """
        H, W = hw_shape
        indexes = np.arange(H * W)
        zi = ZOrderIndexer((0, H - 1), (0, W - 1))
        o1 = []

        for z in indexes:
            r, c = zi.rc(int(z))
            o1.append(r * H + c)

        o1 = np.array(o1)
        o1_inverse = np.argsort(o1)

        return o1.tolist(), o1_inverse.tolist()

    @staticmethod
    def scan(hw_shape):
        """
        Generate a scan-line traversal order with alternating row directions.

        This serialization method performs a horizontal scan over the grid but
        reverses every second row to create a snake-like left-to-right then
        right-to-left pattern. Unlike `parallel_snake_horizontal`, which may start
        scanning from the bottom or top depending on grid parity, this method always
        starts from the top-left corner and alternates direction strictly based on
        row index parity.

        Args:
            hw_shape (tuple[int, int]):
                A tuple (H, W) representing the grid height and width.

        Returns:
            tuple[list[int], list[int]]:
                - o1: A list of indices representing the forward scan traversal
                    following the alternating-direction pattern.
                - o1_inverse: A list where each position stores the inverse mapping,
                            such that `o1_inverse[o1[k]] = k`, enabling fast lookup
                            of the position of each grid index in the serialized order.
        """
        H, W = hw_shape
        indexes = np.arange(H * W)
        indexes = indexes.reshape(H, W)
        for i in np.arange(1, H, step=2):
            indexes[i, :] = indexes[i, :][::-1]
        o1 = indexes.reshape(-1)

        o1 = np.array(o1)
        o1_inverse = np.argsort(o1)

        return o1.tolist(), o1_inverse.tolist()

    @staticmethod
    def hilbert(hw_shape):
        H, W = hw_shape
        indexes = np.arange(H * W)
        bit = int(math.log2(H))
        locs = hilbert.decode(torch.tensor(indexes), 2, bit)
        ret = []
        l = 2 ** bit
        for i in range(len(locs)):
            loc = locs[i]
            loc_flat = 0
            for j in range(2):
                loc_flat += loc[j] * (l ** j)
            ret.append(loc_flat)

        o1 = np.array(ret).astype(np.uint64)
        o1_inverse = np.argsort(o1)

        return o1.tolist(), o1_inverse.tolist()


class S6_2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_size=7,
            bias=False, 
            init_layer_scale=None,
            default_hw_shape=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.default_hw_shape = default_hw_shape
        self.default_permute_order = None
        self.default_permute_order_inverse = None

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        assert conv_size % 2 == 1
        self.conv2d = BottConv(in_channels=self.d_inner, out_channels=self.d_inner, mid_channels=self.d_inner // 16, kernel_size=3, padding=1, stride=1)
        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True
        )

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4 real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, x, hw_shape, order, inv_order):
        batch_size, L, _ = x.shape
        H, W = hw_shape
        E = self.d_inner

        ssm_state = None
        xz = self.in_proj(x) # [B, L, 2 * d_inner(8 * d_model)] a more efficient manner to process the input
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)

        x, z = xz.chunk(2, dim=-1) # split into two parts, each [B, L, d_inner(8 * d_model)]
        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.act(self.conv2d(x_2d))
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)
        # construct dt, B, C
        x_dbl = self.x_proj(x_conv) # (B, L, dt_rank + d_state * 2)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        dt = dt.permute(0, 2, 1).contiguous() # [B, d_innter, L]
        B = B.permute(0, 2, 1).contiguous() # [B, d_state, L]
        C = C.permute(0, 2, 1).contiguous() # [B, d_state, L]

        assert self.activation in ["silu", "swish"]

        # S6 block
        expanded_order = order.unsqueeze(-1).expand(-1, -1, x.size(-1)).to(x_conv.device)
        x_conv = torch.gather(x_conv, dim=1, index=expanded_order)
        # y_scan: a list
        y_scan = selective_scan_fn(
                x_conv.permute(0, 2, 1).contiguous(), # the input sequence should be BDL
                dt, # selective factor
                A,
                B,
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            ).permute(0, 2, 1)
        expanded_inv_order = inv_order.unsqueeze(-1).expand(-1, -1, x_conv.size(-1)).to(x_conv.device)
        x_conv = torch.gather(x_conv, dim=1, index=expanded_inv_order)

        # y_scan = selective_scan_fn(
        #         x_conv[:, order, :].permute(0, 2, 1).contiguous(), # the input sequence should be BDL
        #         dt, # selective factor
        #         A,
        #         B,
        #         C,
        #         self.D.float(),
        #         z=None,
        #         delta_bias=self.dt_proj.bias.float(),
        #         delta_softplus=True,
        #         return_last_state=ssm_state is not None,
        #     ).permute(0, 2, 1)[:, inv_order, :] # permute back to original order, and the [:, inv_order, :] operations will transform the output sequences back to the original x order

        # cause the y_scan's token order is consistent to the original order (position consistent one-by-one), we can directly sum them up
        y = y_scan * self.act(z) # sum 4 sequences([B, L, d_inner]) and * z([B, L, d_inner])
        out = self.out_proj(y)
        if self.init_layer_scale is not None:
            out = out * self.gamma

        return out


class S6_2D_HS(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_size=7,
            bias=False,
            init_layer_scale=None,
            default_hw_shape=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.default_hw_shape = default_hw_shape
        self.default_permute_order = None
        self.default_permute_order_inverse = None

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)

        assert conv_size % 2 == 1
        self.conv2d = BottConv(in_channels=self.d_inner, out_channels=self.d_inner, mid_channels=self.d_inner // 16, kernel_size=3, padding=1, stride=1)
        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False,
        )
        self.dt_proj = nn.Linear(
            self.dt_rank, self.d_inner, bias=True
        )

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4 real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, x, hw_shape, orders, inverse_orders):
        batch_size, L, _ = x.shape
        H, W = hw_shape
        E = self.d_inner

        conv_state, ssm_state = None, None
        xz = self.in_proj(x) # [B, L, 2 * d_inner(8 * d_model)] a more efficient manner to process the input
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)

        x, z = xz.chunk(2, dim=-1) # split into two parts, each [B, L, d_inner(8 * d_model)]
        x_2d = x.reshape(batch_size, H, W, E).permute(0, 3, 1, 2)
        x_2d = self.act(self.conv2d(x_2d))
        x_conv = x_2d.permute(0, 2, 3, 1).reshape(batch_size, L, E)
        # construct dt, B, C
        x_dbl = self.x_proj(x_conv) # (B, L, dt_rank + d_state * 2)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        dt = dt.permute(0, 2, 1).contiguous() # [B, d_innter, L]
        B = B.permute(0, 2, 1).contiguous() # [B, d_state, L]
        C = C.permute(0, 2, 1).contiguous() # [B, d_state, L]

        assert self.activation in ["silu", "swish"]

        # S6 block
        # y_scan: a list
        y_scan = [
            selective_scan_fn(
                x_conv[:, order, :].permute(0, 2, 1).contiguous(), # the input sequence should be BDL
                dt, # selective factor
                A,
                B,
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            ).permute(0, 2, 1)[:, inv_order, :] # permute back to original order, and the [:, inv_order, :] operations will transform the output sequences back to the original x order
            for order, inv_order in zip(orders, inverse_orders)
        ] # 4 scan sequences, each [B, L, d_inner(expand*d_model)]

        # TODO: try other aggregation methods
        # cause the y_scan's token order is consistent to the original order (position consistent one-by-one), we can directly sum them up
        y = sum(y_scan) * self.act(z) # sum 4 sequences([B, L, d_inner]) and * z([B, L, d_inner])
        # y = y_scan * self.act(z) # sum 4 sequences([B, L, d_inner]) and * z([B, L, d_inner])
        out = self.out_proj(y)
        if self.init_layer_scale is not None:
            out = out * self.gamma

        return out


class SerializationExpert(nn.Module):
    def __init__(self, S6_block, strategy_fn):
        super().__init__()
        self.S6_block = S6_block
        self.serialize = strategy_fn

    def forward(self, x, hw_shape):
        order, inv_order = self.serialize(hw_shape)
        out = self.S6_block(x, hw_shape, order, inv_order)
        return out


class SwapInput(nn.Module):
    def __init__(self, dim: int, strategy_fn):
        super().__init__()
        self.mlp = FeedForward(dim=dim)
        self.serialize = strategy_fn

    def forward(self, x, hw_shape):
        order, inv_order = self.serialize(hw_shape)
        x = x[:, order, :]
        x = self.mlp(x)
        x = x[:, inv_order, :]
        return x


class SwitchMoE_HS_base(nn.Module):
    """
    A module that implements the Switched Mixture of Experts (MoE) architecture. 

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float, optional): The capacity factor that controls the capacity of the MoE. Defaults to 1.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float): The capacity factor that controls the capacity of the MoE.
        experts (nn.ModuleList): The list of feedforward networks representing the experts.
        gate (SwitchGate): The switch gate module.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        output_dim: int,
        mamba_cfg,
        capacity_factor: float = 1.0,
        use_aux_loss: bool = False,
        use_conv_gate: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.capacity_factor = capacity_factor
        self.use_aux_loss = use_aux_loss

        # -------- Experts -----------
        serial = SerializationStrategies()
        strategies = [
            # lambda hw_shape: serial.Parallel_snake_horizontal(hw_shape),
            # lambda hw_shape: serial.Parallel_snake_horizontal2(hw_shape),
            # lambda hw_shape: serial.Parallel_snake_vertical(hw_shape),
            lambda hw_shape: serial.Diagonal_snake_left(hw_shape),
            lambda hw_shape: serial.Diagonal_snake_right(hw_shape),
            # lambda hw_shape: serial.zorder(hw_shape),
            # lambda hw_shape: serial.zigzag(hw_shape),
            # lambda hw_shape: serial.hilbert(hw_shape),
        ]
        self.num_experts = len(strategies)

        self.experts = nn.ModuleList()
        for i, strat in enumerate(strategies):
            expert = SerializationExpert(S6_2D(**mamba_cfg), strategy_fn=strat)
            self.experts.append(expert)

        # -------- Gating -----------
        self.use_conv_gate = use_conv_gate
        if self.use_conv_gate:
            self.gate = SwitchGate_Conv(
                self.dim,
                self.num_experts,
                self.capacity_factor,
            )
        else:
            self.gate = SwitchGate(
                self.dim,
                self.num_experts,
                self.capacity_factor,
            )

    def forward(self, x: torch.Tensor, hw_shape):
        """
        Forward pass of the SwitchMoE_HS_base module.

        Args:
            x (Tensor): The input tensor. shape is BLC

        Returns:
            Tensor: The output tensor of the MoE.
            Loss: Auxiliary loss from the gating mechanism.

        """
        # if self.use_conv_gate:
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # (batch_size, seq_len, num_experts)
        gate_scores, loss = self.gate(x, use_aux_loss=self.use_aux_loss)
        
        # if self.use_conv_gate:
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # Dispatch to experts
        expert_outputs = [expert(x, hw_shape) for expert in self.experts] # each element (batch_size, seq_len, output_dim)

        # Check if any gate scores are nan and handle
        if torch.isnan(gate_scores).any():
            print("NaN in gate scores")
            gate_scores[torch.isnan(gate_scores)] = 0

        # Stack and weight outputs
        stacked_expert_outputs = torch.stack(expert_outputs, dim=-1) # (batch_size, seq_len, output_dim, num_experts)

        # Check if any expert outputs are nan and handle
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[torch.isnan(stacked_expert_outputs)] = 0

        # Combine expert outputs and gating scores
        # print(gate_scores.unsqueeze(-2).shape)
        # print(stacked_expert_outputs.shape)
        moe_output = torch.sum(
            gate_scores.unsqueeze(-2).unsqueeze(-2) * stacked_expert_outputs, dim=-1
        )
        return moe_output, loss


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
        The purpose of this class is to create input minibatches for the
        experts and to combine the results of the experts to form a unified
        output tensor.
        There are two functions:
        dispatch - take an input Tensor and create input Tensors for each expert.
        combine - take output Tensors from each expert and form a combined output
        Tensor.  Outputs from different experts for the same batch element are
        summed together, weighted by the provided "gates".
        The class is initialized with a "gates" Tensor, which specifies which
        batch elements go to which experts, and the weights to use when combining
        the outputs. Batch element b is sent to expert e if gates[b, e] != 0.
        The inputs and outputs are all two-dimensional [batch, depth].
        Caller is responsible for collapsing additional dimensions prior to
        calling this class and reshaping the output to the original shape.
        See common_layers.reshape_like().
        Example use:
        gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
        inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
        experts: a list of length `num_experts` containing sub-networks.
        dispatcher = SparseDispatcher(num_experts, gates)
        expert_inputs = dispatcher.dispatch(inputs)
        expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
        outputs = dispatcher.combine(expert_outputs)
        The preceding code sets the output for a particular example b to:
        output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
        This class takes advantage of sparsity in the gate matrix by including in the
        `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.

        Args:
            inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
            a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.

        Args:
            expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
            multiply_by_gates: a boolean
        Returns:
            ca `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = torch.einsum("ijkh,ik -> ijkh", stitched, self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3),
                            requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()

    def expert_to_gates(self):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.

        Args:
            expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
            multiply_by_gates: a boolean
        Returns:
            a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class SwitchMoE_HS(nn.Module):
    """
    A module that implements the Switched Mixture of Experts (MoE) architecture. 

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float, optional): The capacity factor that controls the capacity of the MoE. Defaults to 1.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float): The capacity factor that controls the capacity of the MoE.
        experts (nn.ModuleList): The list of feedforward networks representing the experts.
        gate (SwitchGate): The switch gate module.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        output_dim: int,
        mamba_cfg,
        top_k: int = 1,
        capacity_factor: float = 1.0,
        use_conv_gate: bool = False,
        use_noisy_gate: bool = True,
        use_residual_connection: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # -------- Experts -----------
        serial = SerializationStrategies()
        strategies = [
            lambda hw_shape: serial.Parallel_snake_horizontal(hw_shape),
            lambda hw_shape: serial.Parallel_snake_horizontal2(hw_shape),
            lambda hw_shape: serial.Parallel_snake_vertical(hw_shape),
            lambda hw_shape: serial.Diagonal_snake_left(hw_shape),
            lambda hw_shape: serial.Diagonal_snake_right(hw_shape),
            lambda hw_shape: serial.zorder(hw_shape),
            lambda hw_shape: serial.zigzag(hw_shape),
            lambda hw_shape: serial.hilbert(hw_shape),
        ]
        self.num_experts = len(strategies)
        assert (self.top_k <= self.num_experts)

        self.experts = nn.ModuleList()
        for i, strat in enumerate(strategies):
            expert = SerializationExpert(S6_2D(**mamba_cfg), strategy_fn=strat)
            self.experts.append(expert)

        # -------- Gating -----------
        self.use_conv_gate = use_conv_gate
        self.use_noisy_gate = use_noisy_gate
        if self.use_conv_gate:
            self.gate = SwitchGate_Conv(
                self.dim,
                self.num_experts,
                self.capacity_factor,
            )
        else:
            # self.gate = SwitchGate(
            #     self.dim,
            #     self.num_experts,
            #     self.capacity_factor,
            # )
            self.gate = nn.Linear(self.dim, self.num_experts)
        self.gate_norm = nn.LayerNorm(self.num_experts)

        self.noise = nn.Linear(self.dim, self.num_experts)
        self.noise_norm = nn.LayerNorm(self.num_experts)
        
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(-1)

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.use_residual_connection = use_residual_connection

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.

        Args:
            x: a `Tensor`.
        Returns:
            a `Scalar`.
        """
        eps = 1e-10
        # if only num_expert = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        cv_sq = x.float().var() / (x.float().mean() ** 2 + eps)
        return torch.clamp(cv_sq, max=1e6)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.

        Args:
            gates: a `Tensor` of shape [batch_size, n]
        Returns:
            a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating. # TODO: dig into this
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.

        Args:
            clean_values: a `Tensor` of shape [batch_size, num_experts].
            noisy_values: a `Tensor` of shape [batch_size, num_experts].  Equal to clean values plus
            normally distributed noise with standard deviation noise_stddev.
            noise_stddev: a `Tensor` of shape [batch_size, num_experts], or None
            noisy_top_values: a `Tensor` of shape [batch_size, m].
            "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
            a `Tensor` of shape [batch_size, n].
        """
        # print("noisy_top_values.shape: ", noisy_top_values.shape)
        batch = clean_values.size(0)
        m = noisy_top_values.size(-1)
        top_values_flat = noisy_top_values.flatten()
        # print("top_values_flat.shape: ", top_values_flat.shape)

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        # print("threshold_positions_if_in: ", threshold_positions_if_in)
        # print("threshold_positions_if_in.shape: ", threshold_positions_if_in.shape)
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        # print("threshold_if_in: ", threshold_if_in)
        is_in = torch.gt(noisy_values, threshold_if_in)
        # print("is_in: ", is_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)

        scaled_if_in = (clean_values - threshold_if_in) / (noise_stddev + 1e-8)
        scaled_if_out = (clean_values - threshold_if_out) / (noise_stddev + 1e-8)

        scaled_if_in = torch.clamp(scaled_if_in, min=-10.0, max=10.0)
        scaled_if_out = torch.clamp(scaled_if_out, min=-10.0, max=10.0)

        prob_if_in = normal.cdf(scaled_if_in)
        prob_if_out = normal.cdf(scaled_if_out)

        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
        See paper: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.

        Args:
            x: input Tensor with shape [batch_size, seq_len, feat_dim]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
        Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.gate_norm(self.gate(x.mean(1))) # get clean_logits shape [batch_size, seq_len, num_experts], if add mean(1), got shape [batch_size, num_experts]
        if self.use_noisy_gate and train:
            raw_noise_stddev = self.noise_norm(self.noise(x.mean(1)))
            noise_stddev = (self.softplus(raw_noise_stddev) + noise_epsilon).clamp(max=1.0)
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev # get noisy_logits shape [batch_size, seq_len, num_experts], if add mean(1), got shape [batch_size, num_experts]
            logits = noisy_logits
        else:  
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates, and why? # TODO
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.num_experts), dim=-1) # both shape [batch_size, num_experts]
        top_logits = top_logits - top_logits.max(dim=-1, keepdim=True).values
        top_k_logits = top_logits[:, :self.top_k] # shape [batch_size, self.top_k]
        top_k_indices = top_indices[:, :self.top_k] # shape [batch_size, self.top_k]
        top_k_gates = self.softmax(top_k_logits)
        # print("top_k_gates.shape: ", top_k_gates.shape)
        # print("top_k_gates: ", top_k_gates)
        # print("top_k_indices.shape: ", top_k_indices.shape)
        # print("top_k_indices: ", top_k_indices)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates) # shape [batch_size, num_experts]
        # print("gates.shape: ", gates.shape)
        # print("gates: ", gates)

        if self.use_noisy_gate and self.top_k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0) # shape [num_experts]
        else:
            load = self._gates_to_load(gates) # shape [num_experts]
        # print("load.shape: ", load.shape)
        # print("load: ", load)
        return gates, load

    def forward(self, x, hw_shape, loss_coef=1e-2):
        """
        Args:
            x: tensor shape [batch_size, seq_len, feat_dim]
            train: a boolean scalar. Inside member forward, we can use self.training directly
            loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
            y: a tensor with shape [batch_size, seq_len, output_dim].
            extra_training_loss: a scalar.  This should be added into the overall
            training loss of the model.  The backpropagation of this loss
            encourages all experts to be approximately equally used across a batch.
        """
        residual = x
        gates, load = self.noisy_top_k_gating(x, self.training) # gate shape [batch_size, num_experts], load shape [num_experts]
        # calculate importance loss
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef
        # print("Importance:", importance)
        # print("Load:", load)
        # print("balance_loss:", balance_loss)

        # Dispatch to experts
        expert_outputs = [expert(x, hw_shape) for expert in self.experts] # each element (batch_size, seq_len, output_dim)

        # Check if any gate scores are nan and handle
        if torch.isnan(gates).any():
            print("NaN in gate scores")
            gates[torch.isnan(gates)] = 0

        # Stack and weight outputs
        stacked_expert_outputs = torch.stack(expert_outputs, dim=-1) # (batch_size, seq_len, output_dim, num_experts)

        # Check if any expert outputs are nan and handle
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[torch.isnan(stacked_expert_outputs)] = 0

        # Combine expert outputs and gating scores
        moe_output = torch.sum(gates.unsqueeze(1).unsqueeze(2) * stacked_expert_outputs, dim=-1) # (batch_size, seq_len, output_dim)
        # Or the einsum operator, get the same result
        # moe_output = torch.einsum('blhn, bn->blh', stacked_expert_outputs, gates)

        if self.use_residual_connection:
            moe_output = moe_output + residual

        return moe_output, balance_loss


class SwitchMoE_HS_adaptor(nn.Module):
    """
    A module that implements the Switched Mixture of Experts (MoE) architecture. 

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float, optional): The capacity factor that controls the capacity of the MoE. Defaults to 1.0.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float): The capacity factor that controls the capacity of the MoE.
        experts (nn.ModuleList): The list of feedforward networks representing the experts.
        gate (SwitchGate): The switch gate module.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        output_dim: int,
        mamba_cfg,
        top_k: int = 1,
        capacity_factor: float = 1.0,
        use_conv_gate: bool = False,
        use_noisy_gate: bool = True,
        use_residual_connection: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.ssm = S6_2D(**mamba_cfg)

        # -------- Experts -----------
        serial = SerializationStrategies()
        self.strategies = [
            lambda hw_shape: serial.Parallel_snake_horizontal(hw_shape),
            lambda hw_shape: serial.Parallel_snake_horizontal2(hw_shape),
            lambda hw_shape: serial.Parallel_snake_vertical(hw_shape),
            lambda hw_shape: serial.Diagonal_snake_left(hw_shape),
            lambda hw_shape: serial.Diagonal_snake_right(hw_shape),
            lambda hw_shape: serial.zorder(hw_shape),
            lambda hw_shape: serial.zigzag(hw_shape),
            lambda hw_shape: serial.hilbert(hw_shape),
        ]
        self.num_experts = len(self.strategies)
        assert (self.top_k <= self.num_experts)

        self.experts = nn.ModuleList()
        for i, strat in enumerate(self.strategies):
            expert = SwapInput(dim=self.dim, strategy_fn=strat)
            self.experts.append(expert)

        # -------- Gating -----------
        self.use_conv_gate = use_conv_gate
        self.use_noisy_gate = use_noisy_gate
        if self.use_conv_gate:
            self.gate = SwitchGate_Conv(
                self.dim,
                self.num_experts,
                self.capacity_factor,
            )
        else:
            # self.gate = SwitchGate(
            #     self.dim,
            #     self.num_experts,
            #     self.capacity_factor,
            # )
            self.gate = nn.Linear(self.dim, self.num_experts)
        self.gate_norm = nn.LayerNorm(self.num_experts)

        self.noise = nn.Linear(self.dim, self.num_experts)
        self.noise_norm = nn.LayerNorm(self.num_experts)
        
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(-1)

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        self.use_residual_connection = use_residual_connection

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.

        Args:
            x: a `Tensor`.
        Returns:
            a `Scalar`.
        """
        eps = 1e-10
        # if only num_expert = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        cv_sq = x.float().var() / (x.float().mean() ** 2 + eps)
        return torch.clamp(cv_sq, max=1e6)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.

        Args:
            gates: a `Tensor` of shape [batch_size, n]
        Returns:
            a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating. # TODO: dig into this
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.

        Args:
            clean_values: a `Tensor` of shape [batch_size, num_experts].
            noisy_values: a `Tensor` of shape [batch_size, num_experts].  Equal to clean values plus
            normally distributed noise with standard deviation noise_stddev.
            noise_stddev: a `Tensor` of shape [batch_size, num_experts], or None
            noisy_top_values: a `Tensor` of shape [batch_size, m].
            "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
            a `Tensor` of shape [batch_size, n].
        """
        # print("noisy_top_values.shape: ", noisy_top_values.shape)
        batch = clean_values.size(0)
        m = noisy_top_values.size(-1)
        top_values_flat = noisy_top_values.flatten()
        # print("top_values_flat.shape: ", top_values_flat.shape)

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.top_k
        # print("threshold_positions_if_in: ", threshold_positions_if_in)
        # print("threshold_positions_if_in.shape: ", threshold_positions_if_in.shape)
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        # print("threshold_if_in: ", threshold_if_in)
        is_in = torch.gt(noisy_values, threshold_if_in)
        # print("is_in: ", is_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)

        scaled_if_in = (clean_values - threshold_if_in) / (noise_stddev + 1e-8)
        scaled_if_out = (clean_values - threshold_if_out) / (noise_stddev + 1e-8)

        scaled_if_in = torch.clamp(scaled_if_in, min=-10.0, max=10.0)
        scaled_if_out = torch.clamp(scaled_if_out, min=-10.0, max=10.0)

        prob_if_in = normal.cdf(scaled_if_in)
        prob_if_out = normal.cdf(scaled_if_out)

        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
        See paper: Outrageously large neural networks: The sparsely-gated mixture-of-experts layer.

        Args:
            x: input Tensor with shape [batch_size, seq_len, feat_dim]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
        Returns:
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """
        clean_logits = self.gate_norm(self.gate(x.mean(1))) # get clean_logits shape [batch_size, seq_len, num_experts], if add mean(1), got shape [batch_size, num_experts]
        if self.use_noisy_gate and train:
            raw_noise_stddev = self.noise_norm(self.noise(x.mean(1)))
            noise_stddev = (self.softplus(raw_noise_stddev) + noise_epsilon).clamp(max=1.0)
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev # get noisy_logits shape [batch_size, seq_len, num_experts], if add mean(1), got shape [batch_size, num_experts]
            logits = noisy_logits
        else:  
            logits = clean_logits
        # calculate topk + 1 that will be needed for the noisy gates, and why? # TODO
        top_logits, top_indices = logits.topk(min(self.top_k + 1, self.num_experts), dim=-1) # both shape [batch_size, num_experts]
        top_logits = top_logits - top_logits.max(dim=-1, keepdim=True).values
        top_k_logits = top_logits[:, :self.top_k] # shape [batch_size, self.top_k]
        top_k_indices = top_indices[:, :self.top_k] # shape [batch_size, self.top_k]
        top_k_gates = self.softmax(top_k_logits)
        # print("top_k_gates.shape: ", top_k_gates.shape)
        # print("top_k_gates: ", top_k_gates)
        # print("top_k_indices.shape: ", top_k_indices.shape)
        # print("top_k_indices: ", top_k_indices)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates) # shape [batch_size, num_experts]
        # print("gates.shape: ", gates.shape)
        # print("gates: ", gates)

        if self.use_noisy_gate and self.top_k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0) # shape [num_experts]
        else:
            load = self._gates_to_load(gates) # shape [num_experts]
        # print("load.shape: ", load.shape)
        # print("load: ", load)
        return gates, load

    def forward(self, x, hw_shape, loss_coef=1e-2):
        """
        Args:
            x: tensor shape [batch_size, seq_len, feat_dim]
            train: a boolean scalar. Inside member forward, we can use self.training directly
            loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
            y: a tensor with shape [batch_size, seq_len, output_dim].
            extra_training_loss: a scalar.  This should be added into the overall
            training loss of the model.  The backpropagation of this loss
            encourages all experts to be approximately equally used across a batch.
        """
        residual = x
        gates, load = self.noisy_top_k_gating(x, self.training) # gate shape [batch_size, num_experts], load shape [num_experts]
        # calculate importance loss
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef
        # print("Importance:", importance)
        # print("Load:", load)
        # print("balance_loss:", balance_loss)

        # Dispatch to experts
        expert_outputs = [expert(x, hw_shape) for expert in self.experts] # each element (batch_size, seq_len, output_dim)

        # Check if any gate scores are nan and handle
        if torch.isnan(gates).any():
            print("NaN in gate scores")
            gates[torch.isnan(gates)] = 0

        # Stack and weight outputs
        stacked_expert_outputs = torch.stack(expert_outputs, dim=-1) # (batch_size, seq_len, output_dim, num_experts)

        # Check if any expert outputs are nan and handle
        if torch.isnan(stacked_expert_outputs).any():
            stacked_expert_outputs[torch.isnan(stacked_expert_outputs)] = 0

        # Combine expert outputs and gating scores
        output = torch.sum(gates.unsqueeze(1).unsqueeze(2) * stacked_expert_outputs, dim=-1) # (batch_size, seq_len, output_dim)
        # Or the einsum operator, get the same result
        # output = torch.einsum('blhn, bn->blh', stacked_expert_outputs, gates)

        selected_idx = gates.argmax(dim=-1).long()
        orders = []
        inv_orders = []
        for idx in range(selected_idx.size(0)):
            order, inv_order = torch.tensor(self.strategies[selected_idx[idx]](hw_shape)).long()
            orders.append(order)
            inv_orders.append(inv_order)
        orders = torch.stack(orders) # (batch_size, L)
        inv_orders = torch.stack(inv_orders) # (batch_size, L)

        ssms = self.ssm(x + output, hw_shape, orders, inv_orders)

        if self.use_residual_connection:
            output = ssms + residual

        return output, balance_loss


class SpatialRouter(nn.Module):
    def __init__(self, d_model, num_scans):
        super().__init__()
        self.conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(d_model, num_scans)

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.conv(x)
        feat = self.pool(feat).view(B, C)
        scan_weights = self.fc(feat) # (B, num_scans)
        return F.gumbel_softmax(scan_weights, tau=1.0, hard=True)
        # return torch.softmax(scan_weights, dim=-1)


class MLPRouter(nn.Module):
    def __init__(self, d_model, num_scans):
        super().__init__()
        self.MLP = FeedForward(dim=d_model)
        self.fc = nn.Linear(d_model, num_scans)

    def forward(self, x):
        # BLC = x.shape
        feat = self.MLP(x)
        scan_weights = self.fc(feat) # (B, L, num_scans)
        scan_weights = scan_weights.mean(dim=1) # (B, num_scans)
        return F.gumbel_softmax(scan_weights, tau=1.0, hard=True)
        # return torch.softmax(scan_weights, dim=-1)


class SASS(nn.Module):
    """
    A module that implements the Scan-aware Serialization Selection architecture. 

    Args:
        in_dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        output_dim: int,
        mamba_cfg,
        use_conv_gate: bool = False,
        use_residual_connection: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.ssm = S6_2D(**mamba_cfg)

        serial = SerializationStrategies()
        self.strategies = [
            lambda hw_shape: serial.Parallel_snake_horizontal(hw_shape),
            lambda hw_shape: serial.Parallel_snake_horizontal2(hw_shape),
            lambda hw_shape: serial.Parallel_snake_vertical(hw_shape),
            lambda hw_shape: serial.Diagonal_snake_left(hw_shape),
            lambda hw_shape: serial.Diagonal_snake_right(hw_shape),
            lambda hw_shape: serial.zorder(hw_shape),
            lambda hw_shape: serial.zigzag(hw_shape),
            lambda hw_shape: serial.hilbert(hw_shape),
        ]

        self.use_conv_gate = use_conv_gate
        if self.use_conv_gate:
            self.router = SpatialRouter(d_model=self.in_dim, num_scans=len(self.strategies))
        else:
            self.router = MLPRouter(d_model=self.in_dim, num_scans=len(self.strategies))

        self.use_residual_connection = use_residual_connection

    def forward(self, x, hw_shape):
        """
        Args:
            x: tensor shape [batch_size, seq_len, feat_dim]
        Returns:
            y: a tensor indicates the serialization strategy weights with shape [batch_size, seq_len, feat_dim].
        """
        residual = x
        if self.use_conv_gate:
            B, L, C = x.shape
            H = W = int(math.sqrt(L)) # only support square input for now
            x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        gate = self.router(x)
        selected_idx = gate.argmax(dim=-1).long()
        orders = []
        inv_orders = []
        for idx in range(selected_idx.size(0)):
            order, inv_order = torch.tensor(self.strategies[selected_idx[idx]](hw_shape)).long()
            orders.append(order)
            inv_orders.append(inv_order)
        orders = torch.stack(orders) # (batch_size, L)
        inv_orders = torch.stack(inv_orders) # (batch_size, L)

        if self.use_conv_gate:
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        out = self.ssm(x, hw_shape, orders, inv_orders)

        if self.use_residual_connection:
            out = out + residual
        return out


class HSMM_layer(nn.Module):
    def __init__(
            self,
            embed_dims,
            use_rms_norm,
            with_dwconv,
            drop_path_rate,
            mamba_cfg,
            use_conv_gate: bool = False,
            use_noisy_gate: bool = True,
            use_residual_connection: bool = True,
    ):
        super(HSMM_layer, self).__init__()
        mamba_cfg.update({'d_model': embed_dims})
        if use_rms_norm:
            self.norm = RMSNorm(embed_dims)
        else:
            self.norm = nn.LayerNorm(embed_dims)

        # ablation module switcher
        self.with_dwconv = with_dwconv
        if self.with_dwconv:
            self.dw = nn.Sequential(
                nn.Conv2d(
                    embed_dims,
                    embed_dims,
                    kernel_size=(3, 3),
                    padding=(1, 1),
                    bias=False,
                    groups=embed_dims
                ),
                nn.BatchNorm2d(embed_dims),
                nn.GELU(),
            )
        # version base
        # self.serialization_strategies = SerializationStrategies_base()  # default_hw_shape
        # self.HSMM = S6_2D_HS(**mamba_cfg)

        # version1
        # self.HSMM = SASS(in_dim=embed_dims, hidden_dim=embed_dims, output_dim=embed_dims, 
        #                  mamba_cfg=mamba_cfg, 
        #                  use_residual_connection=use_residual_connection)

        # version2
        self.use_noisy_gate = use_noisy_gate
        # self.HSMM = SwitchMoE_HS(dim=embed_dims, hidden_dim=embed_dims, output_dim=embed_dims, 
        #                          mamba_cfg=mamba_cfg,
        #                          use_conv_gate=use_conv_gate, use_noisy_gate=use_noisy_gate, use_residual_connection=use_residual_connection)

        # version3
        self.HSMM = SwitchMoE_HS_adaptor(dim=embed_dims, hidden_dim=embed_dims, output_dim=embed_dims, 
                                 mamba_cfg=mamba_cfg,
                                 use_conv_gate=use_conv_gate, use_noisy_gate=use_noisy_gate, use_residual_connection=use_residual_connection)

        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        self.linear_256 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.GN_256 = nn.GroupNorm(num_channels=256, num_groups=16)
        self.GBC_C = GBC(embed_dims)
        self.PAF_256 = PAF(embed_dims, embed_dims // 2)

    def forward(self, x, hw_shape):
        """
            x: tensor shape [batch_size, seq_len, feat_dim]
        """
        # version base: same as SAVSS but more scanning strategies unified
        # B, L, C = x.shape
        # H = W = int(math.sqrt(L)) # only support square input for now
        # x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # for i in range(2):
        #     x = self.GBC_C(x)

        # x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # orders, inverse_orders = self.serialization_strategies(hw_shape)
        # mixed_x = self.drop_path(self.HSMM(self.norm(x), hw_shape, orders, inverse_orders))

        # version1: get serialization first based on gate alone, return top-1 serialization results
        # mixed_x = self.drop_path(self.HSMM(self.norm(x), hw_shape))

        load_balance_loss = None
        # version2: inject SSM into experts, x shape conversion is set inside HSMM
        # version3: set an adaptor for each scanning strategies and set as experts then add with original x before fed into SSM
        # NOTE that version2 and version3 share the same HSMM class call manner
        (mixed_x, load_balance_loss) = self.HSMM(self.norm(x), hw_shape)
        mixed_x = self.drop_path(mixed_x)

        b, l, c = mixed_x.shape
        h = w = int(math.sqrt(l))
        mixed_x = self.PAF_256(x.permute(0, 2, 1).reshape(b, c, h, w),
                               mixed_x.permute(0, 2, 1).reshape(b, c, h, w))
        mixed_x = self.GN_256(mixed_x).reshape(b, c, h * w).permute(0, 2, 1)

        if self.with_dwconv:
            b, l, c = mixed_x.shape
            h, w = hw_shape
            mixed_x = mixed_x.reshape(b, h, w, c).permute(0, 3, 1, 2)
            mixed_x = self.GBC_C(mixed_x)
            mixed_x = mixed_x.reshape(b, c, h * w).permute(0, 2, 1)

        mixed_x_res = self.linear_256(self.GN_256(mixed_x.permute(0, 2, 1)).permute(0, 2, 1))
        
        if self.use_noisy_gate:
            return mixed_x + mixed_x_res, load_balance_loss
        return mixed_x + mixed_x_res
