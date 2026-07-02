#!/usr/bin/env python
"""
ANNOTATED DIAGNOSTIC FIGURES for the cut-region depth analysis.

This is a VISUALIZATION-ONLY pass. It uses the same loaders / no-flip convention /
background-only point-to-plane ICP / reference = run_1_start in-hand cloud as analyze.py
(section 4d), and the same explicit NOISE_BOX + cut-ROI config. It does NOT re-derive any
analysis numbers beyond what is needed to draw contours; ROI median dz / cavity area
annotations are read from cutregion_summary.csv (produced by analyze.py).

Sign convention (inherited): +dz = surface RECEDED away from camera = tissue REMOVED = RED;
                             -dz = surface APPROACHED camera = BLUE.
Regions are defined by the SAME explicit XY config constants as analyze.py section 4d
(NOISE_BOX_MM, CUT_ROI_CENTER_MM, CUT_ROI_HALF_MM, see below). The significance threshold is
derived here from the NOISE_BOX (robust MAD sigma of reference-relative dz inside it; 2*sigma),
NOT hardcoded -- so it stays in sync with analyze.py. Contours are drawn at dz > +2sigma (red
cavity) and dz < -2sigma (blue approached lobe).

Produces (into analysis_outputs/figures/):
  annotated_regions_overview.png
  annotated_regions_progression.png
  annotated_roi_zoom.png

EXPLICIT REGION CONFIG (mm, in-hand optical frame; keep in sync with analyze.py SECTION 0):
  NOISE_BOX_MM      = (x_min, x_max, y_min, y_max) static background box for the noise floor
  CUT_ROI_CENTER_MM = (cx, cy) cut-region box center (future: planned waypoint XY)
  CUT_ROI_HALF_MM   = half-width of the square cut box

Run:  conda run -n analysis python annotate_regions.py
Reads ONLY the parent experiment folder + cutregion_summary.csv; writes ONLY the 3 PNGs.
"""
import os, glob, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import open3d as o3d
from scipy import ndimage

# >>> SET THIS to the dataset folder you want to analyze (same as analyze.py) <<<
ROOT = "/media/btllab/B2EEF271EEF22CEB/Ubuntu/franka3-sonopet-ros2/data_collection/experiments/20260605T172526_chicken_5_90_50_15_1.5"
OUT  = os.path.join(ROOT, "analysis_outputs")
FIG  = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 220, "font.size": 9})

TRIALS = [("run_1", "_0", 0),
          ("run_2", "_1", 1),
          ("run_3", "_2", 2)]

def manifest_file(run_index):
    """Use indexed manifests while still reading the older first-run filename if present."""
    indexed = os.path.join(ROOT, f"manifest_{run_index}.json")
    legacy = os.path.join(ROOT, "manifest.json")
    if os.path.exists(indexed):
        return indexed
    if run_index == 0 and os.path.exists(legacy):
        return legacy
    raise FileNotFoundError(indexed)

MAN = {r: json.load(open(manifest_file(i))) for r, _, i in TRIALS}

# ---- loaders (verbatim from explore_cutregion.py) --------------------------
def load_raw(path):
    pc = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pc.points)
    valid = np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 1e-9)
    return xyz[valid]

def load_clean(path, z_gate=(0.08, 0.22)):
    """Keep the near meat surface while rejecting the far board/petri background."""
    pc = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pc.points)
    valid = np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 1e-9)
    pc = pc.select_by_index(np.where(valid)[0])
    xyz = np.asarray(pc.points)
    zm = (xyz[:, 2] > z_gate[0]) & (xyz[:, 2] < z_gate[1])
    pc = pc.select_by_index(np.where(zm)[0])
    if len(pc.points) > 200:
        pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return np.asarray(pc.points)

def to_pcd(pts):
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(pts)
    return pc

def depth_grid(xyz, extent, cell=0.0015):
    x0, x1, y0, y1 = extent
    nx = max(20, min(int((x1 - x0) / cell), 300))
    ny = max(20, min(int((y1 - y0) / cell), 300))
    ix = np.clip(((xyz[:, 0] - x0) / (x1 - x0) * (nx - 1)), 0, nx - 1).astype(int)
    iy = np.clip(((xyz[:, 1] - y0) / (y1 - y0) * (ny - 1)), 0, ny - 1).astype(int)
    g = np.full((ny, nx), np.nan)
    order = np.argsort(-xyz[:, 2])
    g[iy[order], ix[order]] = xyz[order, 2]
    return g

def in_hand_cloud_record(run, label):
    """Fetch the in-hand point-cloud record from the manifest instead of filename convention."""
    target = f"pointcloud_{label}"
    return [pc for pc in MAN[run]["pointcloud_scans"]
            if pc["camera"] == "in_hand" and pc["label"] == target][0]

def cloud_inventory():
    clouds = []
    for run, suf, _ in TRIALS:
        for label in ["start", "stop"]:
            pc = in_hand_cloud_record(run, label)
            p = os.path.join(ROOT, "camera_in_hand", f"pointcloud_{label}{suf}.pcd")
            ts = pc["timestamp"]
            clouds.append((f"{run}_{label}", p, ts))
    clouds.sort(key=lambda c: c[2])
    return clouds

CLOUDS = cloud_inventory()
REF_NAME, REF_PATH, REF_TS = [c for c in CLOUDS if c[0] == "run_1_start"][0]
REF_ZSIGN = np.sign(np.median(load_raw(REF_PATH)[:, 2]))

def corrected_clean(path):
    pts = load_clean(path)
    if np.sign(np.median(pts[:, 2])) != REF_ZSIGN:
        pts = pts.copy(); pts[:, 2] *= -1
    return pts

def load_raster_cut_roi(root):
    """Use the manually recorded raster patch as the plotted cut ROI."""
    with open(os.path.join(root, "raster_patch.json"), encoding="utf-8") as fh:
        patch = json.load(fh)
    center_x, center_y, _ = patch["center_m"]
    xy_min, xy_max = patch["xy_min_m"], patch["xy_max_m"]
    half_x = 0.5 * (xy_max[0] - xy_min[0])
    half_y = 0.5 * (xy_max[1] - xy_min[1])
    return (center_x * 1000.0, center_y * 1000.0), max(half_x, half_y) * 1000.0

# ---- ROI / background-mask setup -------------------------------------------
CUT_ROI_CENTER_MM, CUT_ROI_HALF_MM = load_raster_cut_roi(ROOT)
ref_pts = corrected_clean(REF_PATH)

EXT = (np.percentile(ref_pts[:, 0], 1), np.percentile(ref_pts[:, 0], 99),
       np.percentile(ref_pts[:, 1], 1), np.percentile(ref_pts[:, 1], 99))

ref_pc = to_pcd(ref_pts).voxel_down_sample(0.001)
ref_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
ref_grid = depth_grid(np.asarray(ref_pc.points), EXT)
NY, NX = ref_grid.shape

xs = np.linspace(EXT[0], EXT[1], NX); ys = np.linspace(EXT[2], EXT[3], NY)
GX, GY = np.meshgrid(xs, ys)

LL_CX, LL_CY = CUT_ROI_CENTER_MM
LL_HW_X = max(18.0, CUT_ROI_HALF_MM + 8.0)
LL_HW_Y = max(22.0, CUT_ROI_HALF_MM + 12.0)

def in_tissue_zone(pts):
    return (np.abs(pts[:, 0]*1000 - LL_CX) < LL_HW_X) & (np.abs(pts[:, 1]*1000 - LL_CY) < LL_HW_Y)

ref_bg_pts = ref_pts[~in_tissue_zone(ref_pts)]
ref_bg_pc = to_pcd(ref_bg_pts).voxel_down_sample(0.001)
ref_bg_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))

# ---- bg-only ICP + diff map (verbatim) -------------------------------------
def icp_bg_only(path):
    pts = corrected_clean(path)
    src_bg = to_pcd(pts[~in_tissue_zone(pts)]).voxel_down_sample(0.001)
    src_bg.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    T = np.eye(4); last = None
    for rad, it in [(0.008, 60), (0.003, 60)]:
        last = o3d.pipelines.registration.registration_icp(
            src_bg, ref_bg_pc, rad, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it))
        T = last.transformation
    out = to_pcd(pts); out.transform(T)
    return last, np.asarray(out.points)

def diff_map(regpts):
    g = depth_grid(regpts, EXT)
    return (g - ref_grid) * 1000.0   # mm, +ve = receded (removed)

PANELS = ["run_1_start", "run_1_stop", "run_2_stop", "run_3_stop"]
PATHS = {c[0]: c[1] for c in CLOUDS}

diffs = {}
for name in PANELS:
    if name == REF_NAME:
        diffs[name] = np.zeros_like(ref_grid)   # reference vs itself = 0
        continue
    _, rp = icp_bg_only(PATHS[name])
    diffs[name] = diff_map(rp)

# ---- EXPLICIT REGION CONFIG (keep in sync with analyze_raster.py) -----------
NOISE_BOX_MM      = (580.0, 610.0, 145.0, 175.0)   # static background box in fr3_link0 mm
CUT_CX, CUT_CY = CUT_ROI_CENTER_MM
CUT_HALF_MM = CUT_ROI_HALF_MM

# noise floor derived HERE from the NOISE_BOX (robust MAD sigma of reference-relative dz inside
# it); 2*sigma = significance threshold. Pooled over the SAME set of well-registered captures
# analyze.py uses (the 5 non-reference scans incl. the *_start ones), so SIG_THRESH matches
# analyze.py's canonical value (2-sigma ~= 1.0 mm) rather than only the displayed stop panels.
NBx0, NBx1, NBy0, NBy1 = NOISE_BOX_MM
_nbmask = ((GX*1000 >= NBx0) & (GX*1000 <= NBx1) &
           (GY*1000 >= NBy0) & (GY*1000 <= NBy1) & np.isfinite(ref_grid))
NOISE_POOL = ["run_1_stop", "run_2_start", "run_2_stop", "run_3_start", "run_3_stop"]
_nbdiffs = {}
for n in NOISE_POOL:
    _nbdiffs[n] = diffs[n] if n in diffs else diff_map(icp_bg_only(PATHS[n])[1])
_nb = np.concatenate([_nbdiffs[n][_nbmask & np.isfinite(_nbdiffs[n])] for n in NOISE_POOL])
NOISE_MED  = float(np.median(_nb))
SIG_THRESH = float(2.0*1.4826*np.median(np.abs(_nb-NOISE_MED)))   # mm, 2-sigma

extmm = [e*1000 for e in EXT]

# shared color scale across all four panels (98th pct of |dz| over the well panels)
allv = np.concatenate([diffs[n][np.isfinite(diffs[n])] for n in PANELS if n != REF_NAME])
VLIM = max(float(np.nanpercentile(np.abs(allv), 98)), 2.0)

# summary table for per-panel annotation
SUM = pd.read_csv(os.path.join(OUT, "cutregion_summary.csv"))
VOL = pd.read_csv(os.path.join(OUT, "cutregion_volume_proxy.csv"))
def sum_row(name):
    key = "run_1_start(REF)" if name == "run_1_start" else name
    r = SUM[SUM.cloud == key]
    return r.iloc[0] if len(r) else None

def volume_row(name):
    r = VOL[VOL.cloud == name]
    return r.iloc[0] if len(r) else None

# ---- contour helper: dominant receded / approached blob masks ---------------
cut_mask = (np.abs(GX*1000 - CUT_CX) < CUT_HALF_MM) & (np.abs(GY*1000 - CUT_CY) < CUT_HALF_MM)

def smooth_mask(m):
    """binary-close + fill to get clean contourable blobs"""
    m = ndimage.binary_closing(m, iterations=1)
    m = ndimage.binary_fill_holes(m)
    return m

def dominant_blob(diff, sign, win=None):
    """Largest contiguous significant blob. sign=+1 receded(red), -1 approached(blue).
    Optionally restricted to window `win`. Returns float mask (1.0 inside) or None."""
    if sign > 0:
        sig = (diff > SIG_THRESH)
    else:
        sig = (diff < -SIG_THRESH)
    sig = sig & np.isfinite(diff)
    if win is not None:
        sig = sig & win
    if sig.sum() == 0:
        return None
    lab, n = ndimage.label(sig)
    if n == 0:
        return None
    # rank by integrated |dz|
    score = ndimage.sum(np.abs(np.nan_to_num(diff)), lab, index=np.arange(1, n+1))
    m = lab == (int(np.argmax(score)) + 1)
    return smooth_mask(m)

# red cavity: dominant receded blob within the lower-left search window
# blue lobe: dominant approached blob within the search window, ADJACENT to the cut box
SEARCH_WIN = (np.abs(GX*1000 - LL_CX) < LL_HW_X) & (np.abs(GY*1000 - LL_CY) < LL_HW_Y)

def red_mask(name):  return dominant_blob(diffs[name], +1, SEARCH_WIN)
def blue_mask(name): return dominant_blob(diffs[name], -1, SEARCH_WIN)

def draw_box(ax, cx, cy, half, **kw):
    ax.add_patch(Rectangle((cx-half, cy-half), 2*half, 2*half, fill=False, **kw))

def draw_contour(ax, mask, color, lw=2.0, ls="-"):
    if mask is None or mask.sum() == 0:
        return False
    ax.contour(GX*1000, GY*1000, mask.astype(float), levels=[0.5],
               colors=[color], linewidths=lw, linestyles=ls)
    return True

# fraction of a box that is red(receded) / blue(approached)
def box_fractions(diff, cx, cy, half):
    m = (np.abs(GX*1000 - cx) < half) & (np.abs(GY*1000 - cy) < half) & np.isfinite(diff)
    d = diff[m]
    if len(d) == 0:
        return 0.0, 0.0
    return float(np.mean(d > SIG_THRESH)), float(np.mean(d < -SIG_THRESH))

# ============================================================
# FIGURE 1: OVERVIEW (run_3_stop, full FOV, fully annotated)
# ============================================================
def fig_overview():
    name = "run_3_stop"
    diff = diffs[name]
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    im = ax.imshow(diff, origin="lower", extent=extmm, cmap=plt.cm.RdBu_r,
                   vmin=-VLIM, vmax=VLIM, aspect="equal")

    # ROI boxes: explicit raster cut ROI (lime) + static NOISE_BOX (magenta).
    draw_box(ax, CUT_CX, CUT_CY, CUT_HALF_MM, edgecolor="lime", lw=2.4)
    ax.add_patch(Rectangle((NBx0, NBy0), NBx1-NBx0, NBy1-NBy0, fill=False,
                           edgecolor="magenta", lw=2.0))

    # cavity + lobe contours
    rm, bm = red_mask(name), blue_mask(name)
    draw_contour(ax, rm, "#7f0000", lw=2.4)
    draw_contour(ax, bm, "#08306b", lw=2.4)

    ax.plot(CUT_CX, CUT_CY, "k+", ms=12, mew=2)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(f"Annotated cut-region overview  ({name}, full field of view)\n"
                 f"Reference-relative depth vs {REF_NAME}, bg-only point-to-plane ICP. "
                 f"+dz=receded (removed, red).")
    cb = plt.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("dz vs reference (mm)\n+ = receded (tissue removed)")

    legend = [
        Line2D([0],[0], color="lime", lw=2.4, label=f"cut ROI: {2*CUT_HALF_MM:.0f} mm box @ ({CUT_CX:.0f},{CUT_CY:.0f})"),
        Line2D([0],[0], color="magenta", lw=2.0, label=f"NOISE_BOX x[{NBx0:.0f},{NBx1:.0f}] y[{NBy0:.0f},{NBy1:.0f}] (noise floor)"),
        Line2D([0],[0], color="#7f0000", lw=2.4, label=f"receded cavity (removed), dz>+{SIG_THRESH:.2f} mm (2$\\sigma$)"),
        Line2D([0],[0], color="#08306b", lw=2.4, label=f"approached lobe (standoff/dipole), dz<-{SIG_THRESH:.2f} mm"),
    ]
    ax.legend(handles=legend, fontsize=7.5, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(FIG, "annotated_regions_overview.png")
    fig.savefig(out); plt.close(fig); return out

# ============================================================
# FIGURE 2: PROGRESSION (1x4, shared scale)
# ============================================================
def fig_progression():
    fig, axs = plt.subplots(1, 4, figsize=(19, 5.2), sharey=True)
    im = None
    for ax, name in zip(axs, PANELS):
        diff = diffs[name]
        im = ax.imshow(diff, origin="lower", extent=extmm, cmap=plt.cm.RdBu_r,
                       vmin=-VLIM, vmax=VLIM, aspect="equal")
        draw_box(ax, CUT_CX, CUT_CY, CUT_HALF_MM, edgecolor="lime", lw=2.0)
        ax.add_patch(Rectangle((NBx0, NBy0), NBx1-NBx0, NBy1-NBy0, fill=False,
                               edgecolor="magenta", lw=1.4))
        draw_contour(ax, red_mask(name), "#7f0000", lw=2.0)
        draw_contour(ax, blue_mask(name), "#08306b", lw=2.0)
        r = sum_row(name)
        ann = ""
        if r is not None:
            ann = (f"ROI med dz = {r.roi_med_dz:+.2f} mm\n"
                   f"cavity area = {r.blob_area_mm2:.0f} mm$^2$")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("x (mm)")
        if ann:
            ax.text(0.02, 0.02, ann, transform=ax.transAxes, fontsize=8,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.85))
    axs[0].set_ylabel("y (mm)")
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01)
    cb.set_label("dz vs reference (mm)   + = receded (removed, red)")
    fig.suptitle(f"Cavity progression across runs  (shared scale +/-{VLIM:.1f} mm, bg-only ICP vs {REF_NAME}).  "
                 f"Lime = {2*CUT_HALF_MM:.0f} mm cut ROI @ ({CUT_CX:.0f},{CUT_CY:.0f}).  Magenta = NOISE_BOX (noise floor).  "
                 f"Dark-red contour = receded cavity (dz>+{SIG_THRESH:.2f} mm, 2$\\sigma$).  "
                 f"Dark-blue contour = approached lobe (dz<-{SIG_THRESH:.2f} mm).",
                 fontsize=10)
    out = os.path.join(FIG, "annotated_regions_progression.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig); return out

# ============================================================
# FIGURE 3: ROI ZOOM (run_2_stop + run_3_stop, ~5x5 cm window)
# The cut ROI is the manually recorded raster patch from raster_patch.json.
def fig_zoom():
    zoom_names = ["run_2_stop", "run_3_stop"]
    win = 25.0  # half-window mm -> 5x5 cm
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 6.6), sharey=True)
    fracs = {}
    im = None
    for ax, name in zip(axs, zoom_names):
        diff = diffs[name]
        im = ax.imshow(diff, origin="lower", extent=extmm, cmap=plt.cm.RdBu_r,
                       vmin=-VLIM, vmax=VLIM, aspect="equal")
        draw_box(ax, CUT_CX, CUT_CY, CUT_HALF_MM, edgecolor="lime", lw=2.4)
        draw_contour(ax, red_mask(name), "#7f0000", lw=2.2)
        draw_contour(ax, blue_mask(name), "#08306b", lw=2.2)

        rc, bc = box_fractions(diff, CUT_CX, CUT_CY, CUT_HALF_MM)
        fracs[name] = dict(adopt_red=rc, adopt_blue=bc)

        ax.set_xlim(CUT_CX-win, CUT_CX+win)
        ax.set_ylim(CUT_CY-win, CUT_CY+win)
        ax.set_xlabel("x (mm)"); ax.set_title(name, fontsize=11)
        sr = sum_row(name)
        vr = volume_row(name)
        txt = f"raster ROI ({CUT_CX:.0f},{CUT_CY:.0f}): red {rc*100:.0f}% / blue {bc*100:.0f}%"
        if sr is not None and vr is not None:
            txt += (f"\nmedian dz {sr.roi_med_dz:+.2f} mm, mean dz {vr.roi_mean_dz_mm:+.2f} mm"
                    f"\nnet volume {vr.vol_net_mm3:.0f} mm3 ({vr.vol_net_mm3/1000:.2f} cm3)"
                    f"\npos-only volume {vr.vol_pos_only_mm3:.0f} mm3 ({vr.vol_pos_only_mm3/1000:.2f} cm3)")
        ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8.5,
                va="bottom", ha="left",
                bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.9))
    axs[0].set_ylabel("y (mm)")
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01)
    cb.set_label("dz vs reference (mm)   + = receded (removed, red)")
    legend = [
        Line2D([0],[0], color="lime", lw=2.4, label=f"raster cut ROI ({2*CUT_HALF_MM:.0f} mm @ {CUT_CX:.0f},{CUT_CY:.0f})"),
        Line2D([0],[0], color="#7f0000", lw=2.2, label=f"receded cavity (dz>+{SIG_THRESH:.2f} mm)"),
        Line2D([0],[0], color="#08306b", lw=2.2, label=f"approached lobe (dz<-{SIG_THRESH:.2f} mm)"),
    ]
    fig.legend(handles=legend, fontsize=8.5, loc="upper center", ncol=3, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Cut-region zoom (~5x5 cm): raster_patch.json fixes the manual cut ROI.",
                 fontsize=11, y=1.06)
    out = os.path.join(FIG, "annotated_roi_zoom.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out, fracs

if __name__ == "__main__":
    p1 = fig_overview()
    p2 = fig_progression()
    p3, fr = fig_zoom()
    print("FIGS:", p1, p2, p3)
    print(f"NOISE_BOX {NOISE_BOX_MM} -> 2sigma = {SIG_THRESH:.3f} mm; "
          f"CUT_ROI center {CUT_ROI_CENTER_MM} half {CUT_ROI_HALF_MM} mm")
    for nm, f in fr.items():
        print(f"{nm}: raster box red={f['adopt_red']*100:.1f}% blue={f['adopt_blue']*100:.1f}%")
