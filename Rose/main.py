"""
ASCII Rose  ->  rose.gif  (+ rose.mp4 when ffmpeg is installed)

A parametric 3D rose, z-buffered and lit with key/fill/rim/specular lights,
drawn entirely in ASCII characters with a soft bloom glow, twinkling stars,
drifting sparkles and a hand-written "I love you" that shimmers in.

Setup:   pip install numpy pillow          (ffmpeg on PATH for the MP4)
Run:     python main.py                    full animation
         python main.py --preview 120      one frame -> preview.png
"""
import math
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ============================== settings ==============================
MESSAGE = "I love you"
OUT_GIF, OUT_MP4 = "rose.gif", "rose.mp4"

CW, CH = 8, 15                 # character cell size in pixels
COLS, ROWS = 135, 90           # -> 1080 x 1350 (Instagram 4:5)
W, H = COLS * CW, ROWS * CH
SCENE_ROWS = 70                # rows available to the rose
TEXT_Y0, TEXT_ROWS = 71, 16    # rows for the message

N_FRAMES = 168
FPS = 24
FADE = 10                      # fade in / out frames (makes the loop seamless)
BLOOM_END = 72                 # frame where the rose is fully open
TEXT_START, TEXT_SWEEP = 58, 40

BG_TOP = np.array([12, 6, 22]) / 255
BG_BOTTOM = np.array([3, 1, 6]) / 255
BG_GLOW = np.array([60, 8, 40]) / 255


# ============================== fonts ==============================
def find_font(candidates, fc_pattern):
    for c in candidates:
        try:
            ImageFont.truetype(c, 10)
            return c
        except OSError:
            pass
    if shutil.which("fc-match"):
        p = subprocess.run(["fc-match", "-f", "%{file}", fc_pattern],
                           capture_output=True, text=True).stdout.strip()
        if p:
            return p
    return None


MONO_FONT = find_font([
    "JetBrainsMono-Bold.ttf",
    "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Bold.ttf",
    "DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/liberation/LiberationMono-Bold.ttf",
    "consolab.ttf", "/System/Library/Fonts/Menlo.ttc",
], "monospace:bold")

SCRIPT_FONT = find_font([
    "/usr/share/fonts/noto/NotoSerifDisplay-BoldItalic.ttf",
    "NotoSerifDisplay-BoldItalic.ttf",
    "/usr/share/fonts/liberation/LiberationSerif-BoldItalic.ttf",
    "georgiaz.ttf", "timesbi.ttf",
    "/System/Library/Fonts/Supplemental/Georgia Bold Italic.ttf",
], "serif:bold:italic")


def load(path, size):
    if path:
        return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def fit_mono_font():
    size = 20
    while size > 6:
        f = load(MONO_FONT, size)
        a, d = f.getmetrics()
        if f.getlength("M") <= CW + 0.25 and a + d <= CH + 3:
            return f
        size -= 1
    return load(MONO_FONT, 6)


FONT = fit_mono_font()


def glyph_mask(ch):
    img = Image.new("L", (CW, CH), 0)
    if ch.strip():
        d = ImageDraw.Draw(img)
        x0, y0, x1, y1 = d.textbbox((0, 0), ch, font=FONT)
        d.text(((CW - (x1 - x0)) / 2 - x0, (CH - (y1 - y0)) / 2 - y0),
               ch, font=FONT, fill=255)
    return np.asarray(img, np.float32) / 255


def build_ramp(chars, n):
    """Order chars by real ink coverage and keep n evenly spaced levels."""
    cov = sorted((glyph_mask(c).mean(), c) for c in set(chars))
    cmax = cov[-1][0]
    ramp, used = [], set()
    for target in np.linspace(0, cmax, n):
        best = min((abs(v - target), c) for v, c in cov if c not in used)[1]
        used.add(best)
        ramp.append(best)
    return "".join(sorted(ramp, key=lambda c: glyph_mask(c).mean()))


RAMP = build_ramp(" .,-~:;=!*+%#&$@", 16)
NG = len(RAMP)
GLYPHS = {c: i for i, c in enumerate(RAMP)}
EXTRA = "+*.'"                                   # sparkles / stars
ALL_CHARS = RAMP + "".join(c for c in EXTRA if c not in GLYPHS)
GLYPHS = {c: i for i, c in enumerate(ALL_CHARS)}
MASKS = np.stack([glyph_mask(c) for c in ALL_CHARS])


# ============================== colour ==============================
def hexrgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255


def gradient(stops):
    pos = np.array([p for p, _ in stops])
    cols = np.stack([hexrgb(c) for _, c in stops])

    def f(t):
        t = np.clip(t, 0, 1)
        return np.stack([np.interp(t, pos, cols[:, k]) for k in range(3)], -1)
    return f


ROSE = gradient([(0.00, "#1c0008"), (0.18, "#4f0517"), (0.40, "#9c0f2e"),
                 (0.62, "#e0234a"), (0.80, "#ff5c7c"), (0.92, "#ffa3b5"),
                 (1.00, "#fff1f3")])
LEAF = gradient([(0.00, "#062312"), (0.25, "#11552b"), (0.55, "#2a9a48"),
                 (0.80, "#74db7f"), (1.00, "#e2ffd9")])
STEM = gradient([(0.00, "#082a14"), (0.40, "#1b6d34"), (0.80, "#4fb35c"),
                 (1.00, "#c2f2b4")])
TEXT = gradient([(0.00, "#5a0a2a"), (0.35, "#c2185b"), (0.65, "#ff6f91"),
                 (0.88, "#ffc2d1"), (1.00, "#fff4e6")])
GOLD = hexrgb("#ffe3a3")
MATERIALS = [ROSE, LEAF, STEM]


# ============================== geometry ==============================
def normals(P):
    n = np.cross(np.gradient(P, axis=0), np.gradient(P, axis=1))
    return n / (np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12)


def rose_surface(bloom, nt=3200, nx=130):
    """Parametric rose; `bloom` (0..1) opens the petals."""
    x, t = np.meshgrid(np.linspace(0, 1, nx),
                       np.linspace(-2 * np.pi, 15 * np.pi, nt))
    p = (np.pi / 2) * np.exp(-t / (8 * np.pi)) * bloom
    # petal outline (pointed tips) and the outward curl of each petal's edge
    u = 1 - 0.5 * (1.25 * (1 - np.mod(3.6 * t, 2 * np.pi) / np.pi) ** 2 - 0.25) ** 2
    y = 1.95653 * x ** 2 * (1.27689 * x - 1) ** 2 * np.sin(p)
    r = u * (x * np.sin(p) + y * np.cos(p))
    P = np.stack([r * np.cos(t), r * np.sin(t),
                  u * (x * np.cos(p) - y * np.sin(p))], -1)
    inner = (t + 2 * np.pi) / (17 * np.pi)       # 0 outer petals -> 1 core
    ao = (0.45 + 0.55 * x ** 0.7) * (1 - 0.4 * inner ** 1.5)
    ao = ao + 0.45 * x ** 10                     # bright rolled petal rims
    return P.reshape(-1, 3), normals(P).reshape(-1, 3), ao.reshape(-1)


STEM_LEN = 2.25


def stem_x(s):
    return 0.07 * np.sin(2.4 * s + 0.3) - 0.02


def stem_y(s):
    return 0.04 * np.sin(1.7 * s)


def stem_point(z):
    s = (-0.02 - z) / STEM_LEN
    return stem_x(s), stem_y(s)


def leaf(zb, phi, L, lift=0.55):
    bx, by = stem_point(zb)
    l = np.linspace(0, 1, 170)[:, None]
    w = np.linspace(-1, 1, 44)[None, :]
    serr = 1 + 0.07 * np.abs(np.sin(26 * l)) * np.abs(w) ** 4
    half = 0.36 * L * np.sin(np.pi * l ** 0.85) ** 0.9 * serr
    along = L * l
    side = w * half
    fold = 0.18 * np.abs(side)                  # V-shaped along the midrib
    X = bx + along * np.cos(phi) - side * np.sin(phi)
    Y = by + along * np.sin(phi) + side * np.cos(phi)
    Z = zb + lift * along - 0.9 * along ** 2 / L + fold
    P = np.stack(np.broadcast_arrays(X, Y, Z), -1)
    vein = np.exp(-(w / 0.06) ** 2) * 0.35
    side_veins = 0.12 * (np.sin(18 * l - 7 * np.abs(w)) > 0.85)
    ao = 0.62 + 0.38 * (1 - np.abs(w) ** 2) + vein + side_veins
    ao = np.broadcast_to(ao, P.shape[:2])
    return P.reshape(-1, 3), normals(P).reshape(-1, 3), ao.reshape(-1)


def thorn(zb, phi, size=0.07):
    bx, by = stem_point(zb)
    l = np.linspace(0, 1, 20)[:, None]
    a = np.linspace(0, 2 * np.pi, 16, endpoint=False)[None, :]
    rad = 0.022 * (1 - l)
    d = np.array([np.cos(phi), np.sin(phi), -0.5])
    d /= np.linalg.norm(d)
    e1 = np.array([-np.sin(phi), np.cos(phi), 0.0])
    e2 = np.cross(d, e1)
    P = (np.array([bx, by, zb]) + (size * l)[..., None] * d
         + (rad * np.cos(a))[..., None] * e1 + (rad * np.sin(a))[..., None] * e2)
    return P.reshape(-1, 3), normals(P).reshape(-1, 3), np.full(P.shape[0] * P.shape[1], 0.8)


def stem_and_leaves():
    s = np.linspace(0, 1, 600)[:, None]
    a = np.linspace(0, 2 * np.pi, 28, endpoint=False)[None, :]
    rad = 0.045 + 0.015 * (s < 0.03) * (1 - s / 0.03)   # flare under the bloom
    P = np.stack(np.broadcast_arrays(stem_x(s) + rad * np.cos(a),
                                     stem_y(s) + rad * np.sin(a),
                                     -0.02 - STEM_LEN * s), -1)
    N = np.stack(np.broadcast_arrays(np.cos(a), np.sin(a), 0 * s), -1)
    stem = (P.reshape(-1, 3), N.reshape(-1, 3), np.full(P.shape[0] * P.shape[1], 0.9))

    # sepals: small pointed green leaves cupping the bloom
    sepals = [leaf(-0.03, k * 2 * np.pi / 5 + 0.3, 0.22, lift=0.25) for k in range(5)]
    leaves = [leaf(-0.75, 0.35, 0.85), leaf(-1.2, np.pi + 0.15, 0.78),
              leaf(-1.6, 1.9, 0.6)]
    thorns = [thorn(z, ph) for z, ph in
              [(-0.35, 2.4), (-0.82, -0.6), (-1.15, 1.2), (-1.5, 3.6)]]
    stem_parts = [stem] + thorns
    leaf_parts = sepals + leaves

    def cat(parts):
        return [np.concatenate(k) for k in zip(*parts)]
    return cat(leaf_parts), cat(stem_parts)


(LEAF_P, LEAF_N, LEAF_A), (STEM_P, STEM_N, STEM_A) = stem_and_leaves()


# ============================== lighting & z-buffer ==============================
def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


KEY = unit([-0.55, -0.6, 0.6])      # warm key light, upper left, toward viewer
FILL = unit([0.8, -0.3, -0.1])      # soft fill from the right
VIEW = np.array([0.0, -1.0, 0.0])   # viewer looks along +y
HALF = unit(KEY + VIEW)

SCALE = 250.0                        # pixels per world unit
CX = W / 2
CY = 0.30 * H                        # screen y of the bloom's base


def shade_scene(parts, angle, tilt):
    """Project, light and z-buffer every point. Returns per-cell grids."""
    ca, sa, ct, st = np.cos(angle), np.sin(angle), np.cos(tilt), np.sin(tilt)

    def rot(V):
        x = V[:, 0] * ca - V[:, 1] * sa
        y = V[:, 0] * sa + V[:, 1] * ca
        z = V[:, 2]
        return np.stack([x, y * ct - z * st, y * st + z * ct], -1)

    P = rot(np.concatenate([p[0] for p in parts]))
    N = rot(np.concatenate([p[1] for p in parts]))
    ao = np.concatenate([p[2] for p in parts])
    mat = np.concatenate([np.full(len(p[0]), i) for i, p in enumerate(parts)])

    N = N * np.where((N @ VIEW) < 0, -1.0, 1.0)[:, None]    # two-sided
    diff = np.clip(N @ KEY, 0, 1)
    fill = np.clip(N @ FILL, 0, 1)
    ndv = np.clip(N @ VIEW, 0, 1)
    rim = (1 - ndv) ** 3
    spec = np.clip(N @ HALF, 0, 1) ** 28
    lum = (0.10 + 0.80 * diff + 0.22 * fill) * ao + 0.30 * rim + 0.55 * spec
    lum = 1 - np.exp(-1.6 * lum)                               # soft tone-map
    lum = np.clip((lum - 0.12) / 0.8, 0, 1) ** 1.25            # more contrast

    col = np.floor((CX + P[:, 0] * SCALE) / CW).astype(int)
    row = np.floor((CY - P[:, 2] * SCALE) / CH).astype(int)
    ok = (col >= 0) & (col < COLS) & (row >= 0) & (row < SCENE_ROWS)
    cell = (row * COLS + col)[ok]
    depth, lum, mat, spec = P[ok, 1], lum[ok], mat[ok], spec[ok]
    order = np.argsort(depth, kind="stable")                   # nearest first
    _, first = np.unique(cell[order], return_index=True)
    pick = order[first]

    D = np.full(ROWS * COLS, np.inf)
    L = np.zeros(ROWS * COLS)
    M = np.full(ROWS * COLS, -1)
    S = np.zeros(ROWS * COLS)
    D[cell[pick]], L[cell[pick]] = depth[pick], lum[pick]
    M[cell[pick]], S[cell[pick]] = mat[pick], spec[pick]
    D, L, M, S = (a.reshape(ROWS, COLS) for a in (D, L, M, S))

    # contact shadows: darken cells sitting just behind a nearer neighbour,
    # which separates the overlapping petals into readable layers
    pad = np.pad(D, 1, constant_values=np.inf)
    near = np.minimum.reduce([pad[:-2, 1:-1], pad[2:, 1:-1],
                              pad[1:-1, :-2], pad[1:-1, 2:]])
    behind = (M >= 0) & (near < D - 0.06)
    L = np.where(behind, L * 0.5, L)
    return M, L, S


# ============================== background layers ==============================
yy, xx = np.mgrid[0:H, 0:W]
_v = (yy / H)[..., None]
BACKGROUND = BG_TOP * (1 - _v) + BG_BOTTOM * _v
_r = np.hypot((xx - CX) / (0.55 * W), (yy - CY - 60) / (0.42 * H))
BACKGROUND += BG_GLOW * (np.exp(-_r ** 2 * 2.6) * 0.32)[..., None]
_vig = np.clip(1 - 0.55 * np.hypot((xx - W / 2) / W, (yy - H / 2) / H) ** 2 * 2.2, 0, 1)
BACKGROUND *= _vig[..., None]
del yy, xx, _v, _r, _vig

rng = np.random.default_rng(7)
N_STARS = 260
STAR_R = rng.integers(0, ROWS, N_STARS)
STAR_C = rng.integers(0, COLS, N_STARS)
STAR_PH = rng.uniform(0, 2 * np.pi, N_STARS)
STAR_SPD = rng.uniform(0.6, 1.4, N_STARS)
STAR_TINT = rng.uniform(0, 1, N_STARS)

N_SPARK = 46
SP_C0 = rng.uniform(4, COLS - 4, N_SPARK)
SP_R0 = rng.uniform(0, SCENE_ROWS, N_SPARK)
SP_V = rng.choice([1, 2], N_SPARK) * SCENE_ROWS / N_FRAMES   # loops exactly
SP_PH = rng.uniform(0, 2 * np.pi, N_SPARK)
SP_GOLD = rng.uniform(0, 1, N_SPARK) < 0.45


def draw_stars(G, C, f):
    tw = 0.5 + 0.5 * np.sin(2 * np.pi * f / N_FRAMES * 4 * STAR_SPD + STAR_PH)
    free = G[STAR_R, STAR_C] == 0
    show = free & (tw > 0.25)
    lvl = np.where(tw > 0.85, "*", np.where(tw > 0.6, "+", "."))
    tint = (hexrgb("#b9a6ff") * (1 - STAR_TINT[:, None]) + hexrgb("#ffd6e6") * STAR_TINT[:, None])
    bright = (0.25 + 0.75 * tw)[:, None] * tint
    for i in np.flatnonzero(show):
        G[STAR_R[i], STAR_C[i]] = GLYPHS[lvl[i]]
        C[STAR_R[i], STAR_C[i]] = bright[i]


def draw_sparkles(G, C, f, strength):
    if strength <= 0:
        return
    r = (SP_R0 - SP_V * f) % SCENE_ROWS
    c = SP_C0 + 2.5 * np.sin(2 * np.pi * f / N_FRAMES * 3 + SP_PH)
    tw = 0.5 + 0.5 * np.sin(2 * np.pi * f / N_FRAMES * 9 + SP_PH * 3)
    edge = np.clip(np.minimum(r, SCENE_ROWS - r) / 8, 0, 1)    # fade at ends
    b = tw * edge * strength
    ri, ci = r.astype(int), np.clip(c.astype(int), 0, COLS - 1)
    for i in np.flatnonzero(b > 0.15):
        if G[ri[i], ci[i]] != 0:
            continue
        G[ri[i], ci[i]] = GLYPHS["*" if b[i] > 0.7 else "+" if b[i] > 0.4 else "."]
        base = GOLD if SP_GOLD[i] else hexrgb("#ff8fab")
        C[ri[i], ci[i]] = base * (0.35 + 0.65 * b[i])


# ============================== message ==============================
def text_masks():
    w, h = W, TEXT_ROWS * CH
    probe = load(SCRIPT_FONT, 200)
    x0, y0, x1, y1 = probe.getbbox(MESSAGE)
    size = int(200 * min(0.90 * w / (x1 - x0), 0.80 * h / (y1 - y0)))
    font = load(SCRIPT_FONT, max(size, 8))
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = d.textbbox((0, 0), MESSAGE, font=font)
    d.text(((w - (x1 - x0)) / 2 - x0, (h - (y1 - y0)) / 2 - y0),
           MESSAGE, font=font, fill=255)
    cover = np.asarray(img.resize((COLS, TEXT_ROWS), Image.BOX), float) / 255
    halo = np.asarray(img.resize((COLS, TEXT_ROWS), Image.BOX)
                      .filter(ImageFilter.GaussianBlur(1.6)), float) / 255
    return cover, halo


T_COVER, T_HALO = text_masks()
TR, TC = np.mgrid[0:TEXT_ROWS, 0:COLS]
T_DIAG = TC + (TEXT_ROWS - TR) * 0.45          # slanted, like the italic


def draw_text(G, C, f):
    span = COLS + TEXT_ROWS
    reveal = (f - TEXT_START) / TEXT_SWEEP * (span + 6)
    if reveal <= 0:
        return
    rg, cg = G[TEXT_Y0:TEXT_Y0 + TEXT_ROWS], C[TEXT_Y0:TEXT_Y0 + TEXT_ROWS]
    shown = T_DIAG <= reveal

    # moving sheen across the letters, repeating every ~2.3 s once written
    sweep_t = ((f - TEXT_START - TEXT_SWEEP) % 56) / 56
    sheen = np.exp(-((T_DIAG - sweep_t * (span + 40) + 20) / 7) ** 2)
    if f < TEXT_START + TEXT_SWEEP:
        sheen *= 0

    halo = shown & (T_COVER < 0.12) & (T_HALO > 0.10)
    rg[halo] = GLYPHS["."]
    cg[halo] = TEXT(0.25 + T_HALO[halo]) * 0.55

    ink = shown & (T_COVER >= 0.12)
    lvl = 0.25 + 0.75 * T_COVER
    rg[ink] = np.clip((lvl[ink] * (NG - 1)).round().astype(int), 1, NG - 1)
    col = TEXT(0.30 + 0.62 * T_COVER + 0.10 * np.sin(TC * 0.08 + f * 0.12))
    col = col * (1 - sheen[..., None]) + GOLD * sheen[..., None] * 1.1
    cg[ink] = col[ink]

    # glittering pen-tip at the writing edge
    tip = (np.abs(T_DIAG - reveal) < 1.6) & (T_HALO > 0.05)
    if f < TEXT_START + TEXT_SWEEP + 2:
        rg[tip] = GLYPHS["*"]
        cg[tip] = np.array([1.0, 0.95, 0.85])


# ============================== compositing ==============================
def cells_to_image(G, C):
    m = MASKS[G].transpose(0, 2, 1, 3).reshape(H, W)[..., None]
    pix = np.repeat(np.repeat(C, CH, 0), CW, 1)
    return BACKGROUND * (1 - m) + pix * m


def bloom(img):
    im8 = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    small = im8.resize((W // 4, H // 4), Image.BILINEAR)
    g = (np.asarray(small.filter(ImageFilter.GaussianBlur(2.5)), float) * 0.6
         + np.asarray(small.filter(ImageFilter.GaussianBlur(9)), float) * 0.9) / 255
    g = np.asarray(Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8))
                   .resize((W, H), Image.BILINEAR), float) / 255
    return 1 - (1 - img) * (1 - 0.85 * g)              # screen blend


def ease(v):
    v = np.clip(v, 0, 1)
    return 1 - (1 - v) ** 3


def smooth(v):
    v = np.clip(v, 0, 1)
    return v * v * (3 - 2 * v)


FULL_ROSE = rose_surface(0.9)


def render(f):
    b = ease(f / BLOOM_END)
    rose = FULL_ROSE if b >= 0.999 else rose_surface(0.2 + 0.7 * b)
    angle = 2 * np.pi * f / N_FRAMES + 0.9
    tilt = 0.42 + 0.04 * np.sin(2 * np.pi * f / N_FRAMES)
    M, L, S = shade_scene([rose, (LEAF_P, LEAF_N, LEAF_A),
                           (STEM_P, STEM_N, STEM_A)], angle, tilt)

    G = np.zeros((ROWS, COLS), int)
    C = np.zeros((ROWS, COLS, 3))
    for i, grad in enumerate(MATERIALS):
        sel = M == i
        G[sel] = np.clip((L[sel] ** 0.95 * (NG - 1)).round().astype(int), 1, NG - 1)
        C[sel] = grad(L[sel] * 1.05)
    hot = (M == 0) & (S > 0.35)                             # sparkling dew
    C[hot] = C[hot] * 0.5 + 0.5

    draw_stars(G, C, f)
    draw_sparkles(G, C, f, smooth((f - 20) / 40))
    draw_text(G, C, f)

    img = bloom(cells_to_image(G, C))
    fade = smooth(min(f / FADE, (N_FRAMES - 1 - f) / FADE, 1))
    img = BACKGROUND * (1 - fade) + img * fade
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ============================== output ==============================
def main():
    if "--preview" in sys.argv:
        i = sys.argv.index("--preview")
        f = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 130
        Image.fromarray(render(f)).save("preview.png")
        print("Saved preview.png")
        return

    ff = None
    if shutil.which("ffmpeg"):
        ff = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
             "-c:v", "libx264", "-preset", "slow", "-crf", "16",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", OUT_MP4],
            stdin=subprocess.PIPE)

    frames = []
    for f in range(N_FRAMES):
        a = render(f)
        if ff:
            ff.stdin.write(a.tobytes())
        frames.append(Image.fromarray(a))
        print(f"\rrendering frame {f + 1}/{N_FRAMES}", end="", flush=True)
    print()

    if ff:
        ff.stdin.close()
        ff.wait()
        print(f"Saved {OUT_MP4}")

    # one shared palette built from the busiest frames keeps colours stable
    sample = Image.fromarray(np.concatenate(
        [np.asarray(frames[k]) for k in (BLOOM_END, 110, 140)], 0))
    pal = sample.quantize(256, method=Image.Quantize.MEDIANCUT)
    gif = [fr.quantize(palette=pal, dither=Image.Dither.NONE) for fr in frames]
    gif[0].save(OUT_GIF, save_all=True, append_images=gif[1:],
                duration=round(1000 / FPS), loop=0, optimize=False, disposal=1)
    print(f"Saved {OUT_GIF}")


if __name__ == "__main__":
    main()
