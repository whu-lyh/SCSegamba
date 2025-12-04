import matplotlib.pyplot as plt


def get_o1(hw):
    H, W = hw
    o1 = []
    i, j = 0, 0
    d = 1
    for i in range(H):
        row = list(range(i*W, i*W+W))
        if i % 2 == 1:
            row.reverse()
        o1.extend(row)
    return o1

def get_o2(hw):
    H, W = hw
    o2 = []
    for j in range(W):
        col = [i*W + j for i in range(H)]
        if j % 2 == 1:
            col.reverse()
        o2.extend(col)
    return o2

def get_o3(hw):
    H, W = hw
    o3 = []
    # (0,0)->(1,1)->(2,2) ... anti-diagonals
    for s in range(H + W - 1):
        diag = []
        for i in range(H):
            j = s - i
            if 0 <= j < W:
                diag.append(i * W + j)
        if s % 2 == 1:  # zigzag
            diag.reverse()
        o3.extend(diag)
    return o3

def get_o4(hw):
    H, W = hw
    o4 = []
    visited = set()
    dirs = [(0,1),(1,0),(0,-1),(-1,0)]  # right, down, left, up
    d = 0
    i, j = 0, 0
    for _ in range(H * W):
        idx = i * W + j
        o4.append(idx)
        visited.add((i,j))
        ni, nj = i + dirs[d][0], j + dirs[d][1]
        if not (0 <= ni < H and 0 <= nj < W and (ni,nj) not in visited):
            d = (d + 1) % 4
            ni, nj = i + dirs[d][0], j + dirs[d][1]
        i, j = ni, nj
    return o4


def sass(hw_shape):
        H, W = hw_shape
        L = H * W
        o1, o2, o3, o4 = [], [], [], []
        d1, d2, d3, d4 = [], [], [], []
        o1_inverse = [-1 for _ in range(L)]
        o2_inverse = [-1 for _ in range(L)]
        o3_inverse = [-1 for _ in range(L)]
        o4_inverse = [-1 for _ in range(L)]

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

        i, j = 0, 0
        i_d = "down"
        while j < W:
            assert i_d in ["down", "up"]
            idx = i * W + j
            o2_inverse[idx] = len(o2)
            o2.append(idx)
            if i_d == "down":
                if i < H - 1:
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
        for diag in range(H + W - 1):
            if diag % 2 == 0:
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + j
                        o3.append(idx)
                        o3_inverse[idx] = len(o1) - 1
                        d3.append(1 if j == diag else 4)
            else:
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + j
                        o3.append(idx)
                        o3_inverse[idx] = len(o1) - 1
                        d3.append(4 if i == diag else 1)
        d3 = [0] + d3[:-1]

        for diag in range(H + W - 1):
            if diag % 2 == 0:
                for i in range(min(diag + 1, H)):
                    j = diag - i
                    if j < W:
                        idx = i * W + (W - j - 1)
                        o4.append(idx)
                        o4_inverse[idx] = len(o4) - 1
                        d4.append(1 if j == diag else 4)
            else:
                for j in range(min(diag + 1, W)):
                    i = diag - j
                    if i < H:
                        idx = i * W + (W - j - 1)
                        o4.append(idx)
                        o4_inverse[idx] = len(o4) - 1
                        d4.append(4 if i == diag else 1)
        d4 = [0] + d4[:-1]

        return (tuple(o1), tuple(o2), tuple(o3), tuple(o4)), \
            (tuple(o1_inverse), tuple(o2_inverse), tuple(o3_inverse), tuple(o4_inverse)), \
            (tuple(d1), tuple(d2), tuple(d3), tuple(d4))


def draw_path(ax, path, H, W, label, color):
    xs = [p % W for p in path]
    ys = [p // W for p in path]

    ax.plot(xs, ys, marker="o", label=label, color=color)

    # arrows
    for k in range(len(xs) - 1):
        ax.annotate("",
                    xy=(xs[k+1], ys[k+1]),
                    xytext=(xs[k], ys[k]),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color=color))


H, W = 4, 4
# o1 = get_o1((H,W))
# o2 = get_o2((H,W))
# o3 = get_o3((H,W))
# o4 = get_o4((H,W))

(o1, o2, o3, o4), _, _ = sass((H, W))

plt.figure(figsize=(8, 8))
ax = plt.gca()

# draw_path(ax, o1, H, W, "o1 snake horizontal", "tab:blue")
# draw_path(ax, o2, H, W, "o2 snake vertical", "tab:orange")
# draw_path(ax, o3, H, W, "o3 zigzag diagonal", "tab:green")
draw_path(ax, o4, H, W, "o4 zigzag", "tab:red")

ax.set_title("Four Paths with Arrows (o1/o2/o3/o4)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_aspect("equal")
ax.grid(True)
ax.invert_yaxis()
ax.legend()

plt.savefig("four_paths_with_arrows.png", dpi=300)