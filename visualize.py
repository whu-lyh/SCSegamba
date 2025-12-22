import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from pyzorder import ZOrderIndexer

from util.hilbert import decode


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
        locs = decode(torch.tensor(indexes), 2, bit)
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


def draw_path(ax, path, H, W, label, color="black"):
    """
    Draw a single traversal path with arrows on given axes.
    """
    xs = [p % W for p in path]
    ys = [p // W for p in path]

    ax.plot(xs, ys, marker="o", label=label, color=color)
    for k in range(len(xs) - 1):
        ax.annotate("",
                    xy=(xs[k+1], ys[k+1]),
                    xytext=(xs[k], ys[k]),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color=color))

    ax.set_aspect("equal")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.invert_yaxis()
    ax.legend()
    ax.set_title(label)


def plot_serializations(serializations, H, W):
    """
    Plot all serialization paths in a loop, saving each as a PNG.

    Args:
        serializations (list of tuples): [(name, sequence_list)]
        H, W (int): Grid dimensions
    """
    for name, seq in serializations:
        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        # Generate a random color
        color = np.random.rand(3,)
        draw_path(ax, seq, H, W, label=name, color=color)
        filename = f"{name}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved {filename}")




@torch.no_grad()
def indices_to_row_col(indices: torch.Tensor, width: int):
    """
    Convert 1D linear indices to 2D spatial coordinates.

    Args:
        indices (Tensor): 1D tensor of linear indices, shape (L,).
        width (int): Image width.

    Returns:
        row (Tensor): Row indices, shape (L,).
        col (Tensor): Column indices, shape (L,).
    """
    row = indices // width
    col = indices % width
    return row, col


@torch.no_grad()
def compute_jump_statistics(indices: torch.Tensor):
    """
    Compute jump-based statistics along the scan sequence.

    Statistics:
        f1: Mean absolute jump distance
        f2: Jump variance
        f3: Local continuity ratio (fraction of unit jumps)

    Args:
        indices (Tensor): 1D scan order indices, shape (L,).

    Returns:
        Tensor: Shape (3,), [f1, f2, f3]
    """
    d_index = (indices[1:] - indices[:-1]).abs().float()

    mean_jump = d_index.mean()
    jump_variance = d_index.var(unbiased=False)
    continuity_ratio = (d_index == 1).float().mean()

    return torch.stack([mean_jump, jump_variance, continuity_ratio])


@torch.no_grad()
def compute_spatial_step_statistics(indices: torch.Tensor, width: int):
    """
    Compute average Manhattan step length in spatial domain.

    Statistics:
        f4: Mean spatial Manhattan step

    Args:
        indices (Tensor): 1D scan order indices.
        width (int): Image width.

    Returns:
        Tensor: Shape (1,), [f4]
    """
    row, col = indices_to_row_col(indices, width)

    d_row = (row[1:] - row[:-1]).abs()
    d_col = (col[1:] - col[:-1]).abs()

    manhattan_step = (d_row + d_col).float()
    mean_step = manhattan_step.mean()

    return torch.tensor([mean_step], device=indices.device)


@torch.no_grad()
def compute_direction_statistics(indices: torch.Tensor, width: int):
    """
    Compute direction-related scan statistics.

    Statistics:
        f5: Directional consistency
        f8: Order symmetry

    Args:
        indices (Tensor): 1D scan order indices.
        width (int): Image width.

    Returns:
        Tensor: Shape (2,), [f5, f8]
    """
    device = indices.device
    seq_len = indices.numel()

    row, col = indices_to_row_col(indices, width)
    d_row = row[1:] - row[:-1]
    d_col = col[1:] - col[:-1]

    # ---- Directional consistency ----
    if d_row.numel() > 1:
        sign_row = torch.sign(d_row)
        sign_col = torch.sign(d_col)

        same_direction = (
            (sign_row[1:] == sign_row[:-1]) &
            (sign_col[1:] == sign_col[:-1])
        ).float()

        directional_consistency = same_direction.mean()
    else:
        directional_consistency = torch.tensor(0.0, device=device)

    # ---- Order symmetry ----
    order_symmetry = (indices == (seq_len - 1 - indices.flip(0))).float().mean()

    return torch.stack([directional_consistency, order_symmetry])


@torch.no_grad()
def compute_coverage_statistics(indices: torch.Tensor, height: int, width: int):
    """
    Compute spatial coverage and center bias statistics.

    Statistics:
        f6: Prefix coverage rate
        f7: Center bias

    Args:
        indices (Tensor): 1D scan order indices.
        height (int): Image height.
        width (int): Image width.

    Returns:
        Tensor: Shape (2,), [f6, f7]
    """
    row, col = indices_to_row_col(indices, width)

    # ---- Prefix coverage ----
    row_seq = row.unsqueeze(0)
    col_seq = col.unsqueeze(0)

    min_row, _ = torch.cummin(row_seq, dim=1)
    max_row, _ = torch.cummax(row_seq, dim=1)
    min_col, _ = torch.cummin(col_seq, dim=1)
    max_col, _ = torch.cummax(col_seq, dim=1)

    coverage_radius = (max_row - min_row) + (max_col - min_col)
    prefix_coverage = (coverage_radius / (height + width)).mean().squeeze()

    # ---- Center bias ----
    center_row = (height - 1) / 2.0
    center_col = (width - 1) / 2.0

    center_distance = (
        (row.float() - center_row).abs() +
        (col.float() - center_col).abs()
    )

    center_bias = (center_distance / (height + width)).mean()

    return torch.stack([prefix_coverage, center_bias])


@torch.no_grad()
def compute_local_revisit_rate(indices: torch.Tensor, window: int = 8):
    """
    Compute local revisit frequency within a sliding temporal window.

    Statistic:
        f_A: Fraction of positions revisiting recently visited locations

    Args:
        indices (Tensor): 1D scan order indices.
        window (int): Temporal window size.

    Returns:
        Tensor: Scalar revisit rate
    """
    seq_len = indices.numel()
    if seq_len <= window:
        return torch.tensor(0.0, device=indices.device)

    revisit_count = 0
    for t in range(window, seq_len):
        if indices[t] in indices[t - window:t]:
            revisit_count += 1

    return torch.tensor(revisit_count / (seq_len - window), device=indices.device)


@torch.no_grad()
def compute_cross_row_jump_ratio(indices: torch.Tensor, width: int):
    """
    Compute ratio of jumps crossing more than one row.

    Statistic:
        f_B: Cross-row jump ratio

    Args:
        indices (Tensor): 1D scan order indices.
        width (int): Image width.

    Returns:
        Tensor: Scalar ratio
    """
    row = indices // width
    row_jump = (row[1:] - row[:-1]).abs()
    return (row_jump > 1).float().mean()


@torch.no_grad()
def compute_direction_entropy(indices: torch.Tensor, height: int, width: int):
    """
    Compute entropy of spatial movement directions.

    Statistic:
        f_C: Normalized direction entropy (4-connected)

    Args:
        indices (Tensor): 1D scan order indices.
        height (int): Image height (unused, kept for interface consistency).
        width (int): Image width.

    Returns:
        Tensor: Scalar entropy value in [0, 1]
    """
    row = indices // width
    col = indices % width

    d_row = row[1:] - row[:-1]
    d_col = col[1:] - col[:-1]

    directions = []
    for i in range(d_row.numel()):
        if d_row[i] == 0 and d_col[i] == 1:
            directions.append(0)  # right
        elif d_row[i] == 0 and d_col[i] == -1:
            directions.append(1)  # left
        elif d_row[i] == 1 and d_col[i] == 0:
            directions.append(2)  # down
        elif d_row[i] == -1 and d_col[i] == 0:
            directions.append(3)  # up

    if len(directions) == 0:
        return torch.tensor(0.0, device=indices.device)

    dir_tensor = torch.tensor(directions, device=indices.device)
    hist = torch.bincount(dir_tensor, minlength=4).float()
    prob = hist / hist.sum()

    entropy = -(prob * (prob + 1e-8).log()).sum()
    return entropy / math.log(4)


@torch.no_grad()
def compute_temporal_reversal_rate(indices: torch.Tensor):
    """
    Compute frequency of temporal direction reversals.

    Statistic:
        f_D: Temporal reversal rate

    Args:
        indices (Tensor): 1D scan order indices.

    Returns:
        Tensor: Scalar reversal frequency
    """
    if indices.numel() < 3:
        return torch.tensor(0.0, device=indices.device)

    d1 = indices[1:] - indices[:-1]
    reversal = (d1[1:] * d1[:-1] < 0).float()

    return reversal.mean()


@torch.no_grad()
def compute_block_disruption_rate(indices: torch.Tensor, height: int, width: int, block_size: int = 4):
    """
    Compute frequency of crossing spatial block boundaries.

    Statistic:
        f_E: Block disruption rate

    Args:
        indices (Tensor): 1D scan order indices.
        height (int): Image height.
        width (int): Image width.
        block_size (int): Spatial block size.

    Returns:
        Tensor: Scalar disruption rate
    """
    row = indices // width
    col = indices % width

    block_row = row // block_size
    block_col = col // block_size

    block_change = (
        (block_row[1:] != block_row[:-1]) |
        (block_col[1:] != block_col[:-1])
    )

    return block_change.float().mean()


@torch.no_grad()
def compute_scan_statistics(indices: torch.Tensor, height: int, width: int):
    """
    Aggregate all scan-order structural statistics.

    Output order:
        [f1, f2, f3, f4, f5, f6, f7, f8,
         f_A, f_B, f_C, f_D, f_E]

    Args:
        indices (Tensor): 1D scan order indices.
        height (int): Image height.
        width (int): Image width.

    Returns:
        Tensor: 1D statistics vector
    """
    jump_stats = compute_jump_statistics(indices)                  # (3,)
    step_stats = compute_spatial_step_statistics(indices, width)  # (1,)
    dir_stats = compute_direction_statistics(indices, width)      # (2,)
    cov_stats = compute_coverage_statistics(indices, height, width)  # (2,)

    # revisit_stat = compute_local_revisit_rate(indices)
    cross_row_stat = compute_cross_row_jump_ratio(indices, width)
    # entropy_stat = compute_direction_entropy(indices, height, width)
    reversal_stat = compute_temporal_reversal_rate(indices)
    block_stat = compute_block_disruption_rate(indices, height, width)

    return torch.cat([
        jump_stats,
        step_stats,
        dir_stats[:1],
        cov_stats,
        dir_stats[1:],
        # revisit_stat.unsqueeze(0),
        cross_row_stat.unsqueeze(0),
        # entropy_stat.unsqueeze(0),
        reversal_stat.unsqueeze(0),
        block_stat.unsqueeze(0)
    ], dim=0)


def plot_statistics(stats: torch.Tensor):
    """Plot the radar chart for the given different statistics.

    Args:
        stats (torch.Tensor): the serialization index
    """

    labels = [
        "Jump", "J-Var", "Cont", "Step",
        "DirC", "Cover", "Center-bias", "Sym",
        "Revisit", "Cross", "DirEnt", "Reverse", "Block"
    ]

    num_vars = len(labels)

    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1]

    fig = plt.figure()
    ax = plt.axes(polar=True)

    for i in range(stats.shape[0]):
        values = stats[i].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2)
        ax.fill(angles, values, alpha=0.1)

    ax.set_thetagrids([a * 180 / math.pi for a in angles[:-1]], labels)
    ax.set_title("Scan Structural Fingerprint (Radar Plot)")

    filename = f"Scan_Structural_Fingerprint_radar_plot.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved {filename}")

def rank_norm(stats):
    ranks = stats.argsort(dim=0).argsort(dim=0)
    return ranks.float() / (stats.shape[0] - 1)


if __name__ == "__main__":

    H, W = 16, 16

    # Assuming you already have these sequences from your SerializationStrategies
    # Example: o1_parallel_snake_horizontal, o1_parallel_snake_vertical, etc.
    # Here just using placeholders:
    parallel_snake_horizontal, _ = SerializationStrategies.Parallel_snake_horizontal((H, W))
    parallel_snake_horizontal2, _ = SerializationStrategies.Parallel_snake_horizontal2((H, W))
    parallel_snake_vertical, _ = SerializationStrategies.Parallel_snake_vertical((H, W))
    diagonal_snake_right, _ = SerializationStrategies.Diagonal_snake_right((H, W))
    diagonal_snake_left, _ = SerializationStrategies.Diagonal_snake_left((H, W))
    zorder, _ = SerializationStrategies.zorder((H, W))
    zigzag, _ = SerializationStrategies.zigzag((H, W))
    scan, _ = SerializationStrategies.scan((H, W))
    hilbert, _ = SerializationStrategies.hilbert((H, W))

    # List of (name, sequence) pairs
    serializations = [
        ("plot_parallel_snake_horizontal", parallel_snake_horizontal),
        ("plot_parallel_snake_horizontal2", parallel_snake_horizontal2),
        ("plot_parallel_snake_vertical", parallel_snake_vertical),
        ("plot_diagonal_snake_right", diagonal_snake_right),
        ("plot_diagonal_snake_left", diagonal_snake_left),
        ("plot_zorder", zorder),
        ("plot_zigzag", zigzag),
        ("plot_scan", scan),
        ("plot_hilbert", hilbert)
    ]
    # print(parallel_snake_horizontal)
    # plot_serializations(serializations, H, W)

    stats_list = []
    for _, idx in serializations:
        s1 = compute_scan_statistics(torch.tensor(idx), H, W)
        stats_list.append(rank_norm(s1))

    stats_all = torch.stack(stats_list, dim=0)
    plot_statistics(stats_all)