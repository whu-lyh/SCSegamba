
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.ops.triton.layernorm import RMSNorm
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from pyzorder import ZOrderIndexer

from mmcls.SAVSS_dev.models.SAVSS.moe_layer import FeedForward, SwitchGate_Conv
from models.GBC import GBC, BottConv
from models.PAF import PAF
from util import hilbert


class SerializationStrategies_base(nn.Module):
    def __init__(self):
        super().__init__()
        self.H = 0
        self.W = 0
        self.L = self.H * self.W
    
    def sass(self):
        o1, o2, o3, o4 = [], [], [], []
        d1, d2, d3, d4 = [], [], [], []
        o1_inverse = [-1 for _ in range(self.L)]
        o2_inverse = [-1 for _ in range(self.L)]
        o3_inverse = [-1 for _ in range(self.L)]
        o4_inverse = [-1 for _ in range(self.L)]

        if self.H % 2 == 1:
            i, j = self.H - 1, self.W - 1
            j_d = "left"
        else:
            i, j = self.H - 1, 0
            j_d = "right"

        while i > -1:
            assert j_d in ["right", "left"]
            idx = i * self.W + j
            o1_inverse[idx] = len(o1)
            o1.append(idx)
            if j_d == "right":
                if j < self.W - 1:
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

        i, j = 0, 0
        i_d = "down"
        while j < self.W:
            assert i_d in ["down", "up"]
            idx = i * self.W + j
            o2_inverse[idx] = len(o2)
            o2.append(idx)
            if i_d == "down":
                if i < self.H - 1:
                    i = i + 1
                    d2.append(4)
                else:
                    j = j + 1
                    d2.append(1)
                    i_d = "up"
            else:
                if i > 0:
                    i = i - 1
                    d2.append(3)
                else:
                    j = j + 1
                    d2.append(1)
                    i_d = "down"
        d2 = [0] + d2[:-1]

        # Diagonal route
        for diag in range(self.H + self.W - 1):
            if diag % 2 == 0:
                for i in range(min(diag + 1, self.H)):
                    j = diag - i
                    if j < self.W:
                        idx = i * self.W + j
                        o3.append(idx)
                        o3_inverse[idx] = len(o1) - 1
                        d3.append(1 if j == diag else 4)
            else:
                for j in range(min(diag + 1, self.W)):
                    i = diag - j
                    if i < self.H:
                        idx = i * self.W + j
                        o3.append(idx)
                        o3_inverse[idx] = len(o1) - 1
                        d3.append(4 if i == diag else 1)
        d3 = [0] + d3[:-1]

        for diag in range(self.H + self.W - 1):
            if diag % 2 == 0:
                for i in range(min(diag + 1, self.H)):
                    j = diag - i
                    if j < self.W:
                        idx = i * self.W + (self.W - j - 1)
                        o4.append(idx)
                        o4_inverse[idx] = len(o4) - 1
                        d4.append(1 if j == diag else 4)
            else:
                for j in range(min(diag + 1, self.W)):
                    i = diag - j
                    if i < self.H:
                        idx = i * self.W + (self.W - j - 1)
                        o4.append(idx)
                        o4_inverse[idx] = len(o4) - 1
                        d4.append(4 if i == diag else 1)
        d4 = [0] + d4[:-1]

        return [o1, o2, o3, o4], [o1_inverse, o2_inverse, o3_inverse, o4_inverse]

    def zigzag(self):
        indexes = np.arange(self.L)
        indexes = indexes.reshape(self.H, self.W)
        o1 = []
        for i in range(2 * self.H - 1): # FIXME: CHECKOUT IT OUT H OR W
            if i % 2 == 0:
                start_col = max(0, i - self.W + 1)
                end_col = min(i, self.W - 1)
                for j in range(start_col, end_col + 1):
                    o1.append(indexes[i - j, j])
            else:
                start_row = max(0, i - self.H + 1)
                end_row = min(i, self.H - 1)
                for j in range(start_row, end_row + 1):
                    o1.append(indexes[j, i - j])
        o1 = np.array(o1)
        o1_inverse = np.argsort(o1)
        return o1.tolist(), o1_inverse.tolist()

    def zorder(self):
        indexes = np.arange(self.L)
        zi = ZOrderIndexer((0, self.H - 1), (0, self.W - 1))
        o1 = []
        for z in indexes:
            r, c = zi.rc(int(z))
            o1.append(c * self.H + r)
        o1 = np.array(o1)
        o1_inverse = np.argsort(o1)
        return o1.tolist(), o1_inverse.tolist()
    
    def forward(self, hw_shape):
        self.H, self.W = hw_shape
        self.L = self.H * self.W
        o, o_inverse = [], []
        for ss in [self.zorder(), self.sass(), self.zigzag()]:
            o_seq, o_inv_seq = ss
            if type(o_seq) is list:
                for oi in o_seq:
                     o.append(oi)
            else:
                o.append(o_seq)
            if type(o_inverse) is list:
                for ioi in o_inv_seq:
                     o_inverse.append(ioi)
            else:
                o_inverse.append(o_inverse)
            o_inverse.append(o_inv_seq)
        return o, o_inverse


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
        # y_scan: a list
        y_scan = selective_scan_fn(
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

        # cause the y_scan's token order is consistent to the original order (position consistent one-by-one), we can directly sum them up
        y = y_scan * self.act(z) # sum 4 sequences([B, L, d_inner]) and * z([B, L, d_inner])
        # y = y_scan * self.act(z) # sum 4 sequences([B, L, d_inner]) and * z([B, L, d_inner])
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


class SwitchMoE_HS(nn.Module):
    """
    A module that implements the Switched Mixture of Experts (MoE) architecture. 

    Args:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float, optional): The capacity factor that controls the capacity of the MoE. Defaults to 1.0.
        mult (int, optional): The multiplier for the hidden dimension of the feedforward network. Defaults to 4.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        dim (int): The input dimension.
        hidden_dim (int): The hidden dimension of the feedforward network.
        output_dim (int): The output dimension.
        num_experts (int): The number of experts in the MoE.
        capacity_factor (float): The capacity factor that controls the capacity of the MoE.
        mult (int): The multiplier for the hidden dimension of the feedforward network.
        experts (nn.ModuleList): The list of feedforward networks representing the experts.
        gate (SwitchGate): The switch gate module.

    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        output_dim: int,
        mamba_cfg,
        # num_experts: int,
        capacity_factor: float = 1.0,
        mult: int = 4,
        use_aux_loss: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # self.num_experts = num_experts # set to the same as the number of serialization strategies
        self.capacity_factor = capacity_factor
        self.mult = mult
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
        self.gate = SwitchGate_Conv(
            self.dim,
            self.num_experts,
            self.capacity_factor,
        )

    def forward(self, x: torch.Tensor, hw_shape):
        """
        Forward pass of the SwitchMoE_HS module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor of the MoE.

        """
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        # (batch_size, seq_len, num_experts)
        gate_scores, loss = self.gate(x, use_aux_loss=self.use_aux_loss)
        
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


class HSMM_layer(nn.Module):
    def __init__(
            self,
            embed_dims,
            use_rms_norm,
            with_dwconv,
            drop_path_rate,
            mamba_cfg,
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

        # version2
        self.HSMM = SwitchMoE_HS(dim=embed_dims, hidden_dim=embed_dims, output_dim=embed_dims, mamba_cfg=mamba_cfg)

        self.drop_path = build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))
        self.linear_256 = nn.Linear(in_features=256, out_features=256, bias=True)
        self.GN_256 = nn.GroupNorm(num_channels=256, num_groups=16)
        self.GBC_C = GBC(embed_dims)
        self.PAF_256 = PAF(embed_dims, embed_dims // 2)

    def forward(self, x, hw_shape):
        # B, L, C = x.shape
        # H = W = int(math.sqrt(L))
        # x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # for i in range(2):
        #     x = self.GBC_C(x)

        # x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # version base: same as SAVSS but more scanning strategies unified
        # orders, inverse_orders = self.serialization_strategies(hw_shape)
        # mixed_x = self.drop_path(self.HSMM(self.norm(x), hw_shape, orders, inverse_orders))

        # version1: get serialization first based on gate alone, return top-K serialization results
        # orders used to get corresponding serialization index
        # inverse_orders used to transform the output sequences back to the original order
        # TODO
        
        # version2: inject SSM into experts
        (mixed_x, _) = self.HSMM(self.norm(x), hw_shape)
        mixed_x = self.drop_path(mixed_x)
        
        # version3: combine serialization strategies with MLP then select top-K serializations
        # TODO

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
        return mixed_x + mixed_x_res
