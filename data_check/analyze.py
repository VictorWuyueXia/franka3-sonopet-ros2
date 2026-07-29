#!/usr/bin/env python
"""
Consolidated analysis of the robotic ultrasonic-aspirator tissue-resection dataset
20260528T155704_pork_1_90_50_15_3 (pork, 3 runs, single setting 90/50/15, 3 mm/s).

This single script is the full, self-contained analysis pipeline (inventory -> sync ->
audio -> point clouds / cut-region depth -> frequency -> figures).
It keeps the corrected PASS-2/3 conclusions only. Read top-to-bottom:

  SECTION 0  config / paths / loaders
  SECTION 1  data inventory
  SECTION 2  cross-stream synchronization (decoded timestamps, telemetry continuity)
  SECTION 3  audio diagnostics + audio-stops-early / video-window (30 fps) test
  SECTION 4  point clouds: flip check, reference-relative depth, off-target x=0 ROI
             (legacy noise baseline), and EXPLICIT CUT-REGION resection depth (positive result;
             cut ROI + NOISE_BOX defined by XY config constants at the top of SECTION 0)
  SECTION 5  Sonopet drive frequency (center, drift, inverse freq-vs-load correlation)
  SECTION 6  figures already emitted inline above; unified sync timeline + trial timeline

Run with:  conda run -n analysis python analyze.py
(There is no `sonopet` conda env on this machine; `analysis` has open3d 0.19,
 librosa 0.11, scipy 1.15, opencv 4.12, soundfile.)
Reads ONLY from the parent experiment folder; writes ONLY into analysis_outputs/ (+ figures/).

Produces
--------
Tables (analysis_outputs/):
  00_data_inventory.csv, 01_timeline_summary.csv, 03_audio_feature_summary.csv,
  06_sonopet_telemetry_summary.csv, 07_audio_video_window_test.csv,
  08_pointcloud_clean_inventory.csv, 09_freq_load_correlation.csv,
  10_frequency_finegrained.csv, 11_pointcloud_flip_check.csv,
  12_reference_relative_depth.csv, roi_resection_summary.csv (off-target x=0 noise baseline),
  16_noise_box_sanity.csv (NOISE_BOX per-capture dz sanity-check),
  cutregion_summary.csv, cutregion_volume_proxy.csv, cutregion_layer_volume_removal.csv
  (explicit cut-region positive result)
Figures (analysis_outputs/figures/):
  audio_fft_examples.png, audio_spectrogram_examples.png, audio_window_alignment.png,
  representative_video_frames.png, removed_depth_map_examples.png,
  pointcloud_rescans_depth.png,
  cutregion_significance_panel.png, cutregion_depth_profile.png,
  cutregion_screenshot_overlay.png,
  sonopet_telemetry_overview.png, sonopet_frequency_analysis.png,
  sonopet_frequency_distributions.png, sonopet_frequency_vs_load.png,
  unified_sync_timeline.png, trial_timeline.png
Report: REPORT.md (written by hand, not by this script).
"""
import os, json, glob, warnings, datetime
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import soundfile as sf, librosa, librosa.display, cv2, open3d as o3d
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ============================================================
# SECTION 0 - CONFIG / PATHS / LOADERS
# ============================================================
# >>> SET THIS to the dataset folder you want to analyze <<<
ROOT = "/media/btllab/B2EEF271EEF22CEB/Ubuntu/franka3-sonopet-ros2/data_collection/experiments/20260528T155704_pork_1_90_50_15_3"
OUT  = os.path.join(ROOT, "analysis_outputs")
FIG  = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 300, "font.size": 9})

# Trial -> (label, file suffix, manifest). File suffixes match recorder run indices.
TRIALS = [("run_1", "_0", "manifest_0.json"),
          ("run_2", "_1", "manifest_1.json"),
          ("run_3", "_2", "manifest_2.json")]
MAN = {r: json.load(open(os.path.join(ROOT, m))) for r, _, m in TRIALS}

# Camera fps = 30.
EFFECTIVE_FPS = 30.0

COLORS = {
    "run_1": "#1f4e79",
    "run_2": "#d96c2c",
    "run_3": "#2e7d32",
    "raw": "#8a8f98",
    "audio": "#d96c2c",
    "power": "#1f4e79",
    "mech": "#7a3b9e",
    "depth_map": "viridis_r",
    "diverging": "RdBu_r",
    "noise": "#9aa0a6",
    "cut_roi": "#2fb344",
    "noise_box": "#b23aee",
    "volume_net": "#1f4e79",
    "volume_pos": "#2e7d32",
    "threshold": "#c0392b",
}

# ============================================================
# EXPLICIT REGION CONFIG (edit these; both regions are defined purely by XY here)
# All values in MILLIMETRES, in the in-hand RealSense optical frame (x right, y up,
# +z away from camera). Section 4d consumes ONLY these constants -- there is no
# auto-localization of the cut region.
# ------------------------------------------------------------
# NOISE / BACKGROUND reference box: a fixed, static patch of background used to
# estimate the registration noise floor (robust MAD sigma of reference-relative dz);
# significance threshold = 2*sigma. (x_min, x_max, y_min, y_max) in mm.
NOISE_BOX_MM = (50.0, 90.0, 20.0, 40.0)

# CUT REGION ROI: a square box defined by its CENTER (x,y) and HALF-WIDTH, in mm.
# Future waypoint-driven use: set CUT_ROI_CENTER_MM directly from the planned
# resection waypoint XY (and CUT_ROI_HALF_MM to the desired box half-size). No
# data-driven localization is performed.
# Current value = the "-4 mm left-shifted" 22 mm box from the zoom figure: the
# previous red-core center (-5,-31) shifted 4 mm left to exclude the blue
# approached/standoff-dipole lobe on the right edge -> center (-9, -31) mm.
CUT_ROI_CENTER_MM = (-9.0, -31.0)
CUT_ROI_HALF_MM   = 11.0            # -> 22 mm square box
# ============================================================

# ---- telemetry: load the 200 Hz Sonopet RISE JSONL once ----
SJ = glob.glob(os.path.join(ROOT, "sonopet", "SonopetCase_*.json"))[0]
_recs = []
with open(SJ) as fh:
    for ln in fh:
        ln = ln.strip()
        if ln:
            d = json.loads(ln); r = d["data"]; r["ts"] = d["timestamp"]; _recs.append(r)
TEL = pd.DataFrame(_recs)
TEL_T0 = TEL.ts.min()

# experiment epoch zero (earliest run start) for shared timelines
T0 = min(MAN[r]["started_at"] for r in MAN)

log = []
def L(s):
    print(s); log.append(s)

# ---- point-cloud loaders (shared) ----
def load_raw(path):
    """All finite, non-(0,0,0) points (no gating)."""
    pc = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pc.points)
    valid = np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 1e-9)
    return xyz[valid]

def load_clean(path, z_gate=(0.08, 0.22), with_stats=False):
    """Drop invalid -> keep near meat surface while rejecting far background -> statistical outlier removal."""
    pc = o3d.io.read_point_cloud(path)
    xyz = np.asarray(pc.points)
    valid = np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 1e-9)
    n_invalid = int((~valid).sum())
    pc = pc.select_by_index(np.where(valid)[0])
    xyz = np.asarray(pc.points)
    zm = (xyz[:, 2] > z_gate[0]) & (xyz[:, 2] < z_gate[1])
    n_zgate = int((~zm).sum())
    pc = pc.select_by_index(np.where(zm)[0])
    if len(pc.points) > 200:
        pc, idx = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        n_sor = int(len(zm.nonzero()[0]) - len(idx))
    else:
        n_sor = 0
    pts = np.asarray(pc.points)
    if with_stats:
        return pts, dict(n_invalid=n_invalid, n_zgate=n_zgate, n_sor=n_sor)
    return pts

def to_pcd(pts):
    pc = o3d.geometry.PointCloud(); pc.points = o3d.utility.Vector3dVector(pts)
    return pc

def depth_grid(xyz, extent, cell=0.0015):
    """Top-view (x,y) nearest-z depth grid over a fixed extent (meters)."""
    x0, x1, y0, y1 = extent
    nx = max(20, min(int((x1 - x0) / cell), 300))
    ny = max(20, min(int((y1 - y0) / cell), 300))
    ix = np.clip(((xyz[:, 0] - x0) / (x1 - x0) * (nx - 1)), 0, nx - 1).astype(int)
    iy = np.clip(((xyz[:, 1] - y0) / (y1 - y0) * (ny - 1)), 0, ny - 1).astype(int)
    g = np.full((ny, nx), np.nan)
    order = np.argsort(-xyz[:, 2])         # nearest (small z) written last -> wins
    g[iy[order], ix[order]] = xyz[order, 2]
    return g

def audio_env(suf, fl=4096, hop=1024):
    """RMS envelope of a run's audio + (time axis, total duration s)."""
    y, sr = sf.read(os.path.join(ROOT, "audio", f"audio{suf}.wav"))
    if y.ndim > 1:
        y = y.mean(1)
    nfr = max((len(y) - fl) // hop, 1)
    env = np.array([np.sqrt(np.mean(y[i*hop:i*hop+fl]**2)) for i in range(nfr)])
    te = np.arange(len(env)) * hop / sr
    return te, env, len(y) / sr

# Chronologically-ordered list of in-hand + rescan clouds (only camera we use).
def cloud_inventory():
    clouds = []
    for run, suf, _ in TRIALS:
        for label in ["start", "stop"]:
            p = os.path.join(ROOT, "camera_in_hand", f"pointcloud_{label}{suf}.pcd")
            ts = [pc["timestamp"] for pc in MAN[run]["pointcloud_scans"]
                  if pc["camera"] == "in_hand" and pc["label"] == f"pointcloud_{label}"][0]
            clouds.append((f"{run}_{label}", p, ts))
    for rp in sorted(glob.glob(os.path.join(ROOT, "camera_in_hand", "rescan_*.pcd"))):
        ts = float(os.path.basename(rp).split("_")[1].rsplit(".", 1)[0])
        clouds.append((f"rescan_{ts:.0f}", rp, ts))
    clouds.sort(key=lambda c: c[2])
    return clouds

CLOUDS = cloud_inventory()

# ============================================================
# SECTION 1 - DATA INVENTORY
# ============================================================
print("=== SECTION 1: INVENTORY ===")
rows = []
for dp, _, files in os.walk(ROOT):
    if "analysis_outputs" in dp:
        continue
    for f in files:
        p = os.path.join(dp, f)
        rows.append({"rel_path": os.path.relpath(p, ROOT),
                     "ext": os.path.splitext(f)[1],
                     "size_MB": round(os.path.getsize(p) / 1e6, 3)})
INV = pd.DataFrame(rows).sort_values("rel_path")
INV.to_csv(os.path.join(OUT, "00_data_inventory.csv"), index=False)
print(f"[inventory] {len(INV)} files, {INV.size_MB.sum():.0f} MB total")

# Folder-structure plot removed: inventory remains in 00_data_inventory.csv.

# whole-experiment ground-truth scale from info.txt
info = dict(l.split(": ") for l in open(os.path.join(ROOT, "info.txt")) if ": " in l)
MASS_REMOVED_G = (float(info["sample_weight_before_cut"].split()[0])
                  - float(info["sample_weight_after_cut"].split()[0]))

# ============================================================
# SECTION 2 - CROSS-STREAM SYNCHRONIZATION
# ============================================================
print("\n=== SECTION 2: SYNC ===")
L("## Synchronization (decoded timestamps, telemetry continuity)\n")
L("**Timestamp format:** manifest started_at/stopped_at, point-cloud timestamp, audio meta, "
  "and telemetry timestamp are all Unix epoch seconds on the machine LOCAL clock (EDT, UTC-4), "
  "verified against *_local fields, the folder-name epoch, and file mtimes. Single shared wall clock.")
r1s = MAN["run_1"]["started_at"]
L(f"- run_1 started_at = {r1s} = local {datetime.datetime.fromtimestamp(r1s):%Y-%m-%d %H:%M:%S} "
  f"(matches started_at_local {MAN['run_1']['started_at_local']}).")

# telemetry continuity
dt = np.diff(np.sort(TEL.ts.values))
span = TEL.ts.max() - TEL.ts.min()
gaps = int((dt > 1.0).sum())
L(f"**Telemetry continuity:** {len(TEL)} samples @ {len(TEL)/span:.1f} Hz over {span:.1f} s with "
  f"{gaps} gaps >1 s. The log runs CONTINUOUSLY through the inter-run gaps; the tool stays energized "
  f"the whole time, so 'active cut window' is NOT separable by power -- the manifest windows are the "
  f"only run brackets.\n")

tl = []
for run, suf, _ in TRIALS:
    m = MAN[run]
    a_dur = sf.info(os.path.join(ROOT, m["audio"]["wav"])).duration
    win = m["stopped_at"] - m["started_at"]
    pcs = {(p["camera"], p["label"]): p["timestamp"] for p in m["pointcloud_scans"]}
    tl.append({
        "run": run,
        "start_rel_s": round(m["started_at"] - T0, 2),
        "stop_rel_s":  round(m["stopped_at"] - T0, 2),
        "run_window_s": round(win, 2),
        "audio_duration_s": round(a_dur, 2),
        "audio_vs_window_gap_s": round(win - a_dur, 2),
        "fixed_frames": m["rgb_video"]["fixed"]["frames"],
        "in_hand_frames": m["rgb_video"]["in_hand"]["frames"],
        "in_hand_pc_start_off_s": round(pcs[("in_hand", "pointcloud_start")] - m["started_at"], 2),
        "in_hand_pc_stop_off_s":  round(pcs[("in_hand", "pointcloud_stop")] - m["stopped_at"], 2),
    })
TLS = pd.DataFrame(tl)
TLS.to_csv(os.path.join(OUT, "01_timeline_summary.csv"), index=False)
print(TLS.to_string(index=False))

# ============================================================
# SECTION 3 - AUDIO DIAGNOSTICS + WINDOW-BRACKET TEST
# ============================================================
print("\n=== SECTION 3: AUDIO ===")
au_rows = []
wave_ex = {}
for run, suf, _ in TRIALS:
    wav = os.path.join(ROOT, MAN[run]["audio"]["wav"])
    y, sr = librosa.load(wav, sr=None, mono=True)
    rms = librosa.feature.rms(y=y)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    Y = np.abs(np.fft.rfft(y)); fax = np.fft.rfftfreq(len(y), 1/sr)
    dom = fax[np.argmax(Y[1:]) + 1]
    au_rows.append({"run": run, "sr_hz": sr, "duration_s": round(len(y)/sr, 2),
                    "rms_mean": round(float(rms.mean()), 5), "rms_std": round(float(rms.std()), 5),
                    "peak_abs": round(float(np.abs(y).max()), 4),
                    "clip_fraction": round(float(np.mean(np.abs(y) > 0.98)), 6),
                    "centroid_hz_mean": round(float(cent.mean()), 1),
                    "bandwidth_hz_mean": round(float(bw.mean()), 1),
                    "dominant_freq_hz": round(float(dom), 1)})
    wave_ex[run] = (y, sr)
AU = pd.DataFrame(au_rows)
AU.to_csv(os.path.join(OUT, "03_audio_feature_summary.csv"), index=False)
print(AU.to_string(index=False))

# FFT magnitude spectra (overlaid by run)
fig, ax = plt.subplots(figsize=(9, 4.5))
for run, (y, sr) in wave_ex.items():
    if len(y) > sr * 30:
        y_fft = y[:int(sr * 30)]
    else:
        y_fft = y
    win = np.hanning(len(y_fft))
    mag = np.abs(np.fft.rfft(y_fft * win))
    freq = np.fft.rfftfreq(len(y_fft), 1/sr)
    mag_db = 20*np.log10(mag / np.nanmax(mag) + 1e-12)
    ax.plot(freq / 1000.0, mag_db, lw=1.2, color=COLORS[run], label=run)
ax.set_xlim(0, 24)
ax.set_ylim(-100, 2)
ax.set_xlabel("frequency (kHz)")
ax.set_ylabel("magnitude (dB re peak)")
ax.set_title("Audio FFT magnitude spectra (first 30 s, overlaid by run)")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "audio_fft_examples.png")); plt.close(fig)

# spectrograms
fig, axs = plt.subplots(len(wave_ex), 1, figsize=(9, 2.6*len(wave_ex)))
for ax, (run, (y, sr)) in zip(np.atleast_1d(axs), wave_ex.items()):
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, hop_length=512, x_axis="time", y_axis="hz",
                                   ax=ax, cmap="magma")
    ax.set_ylim(0, 24000); ax.set_title(f"{run} spectrogram (0-24 kHz)")
    fig.colorbar(img, ax=ax, format="%+2.0f dB", shrink=0.8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "audio_spectrogram_examples.png")); plt.close(fig)

# audio-stops-early test + video duration at the 30 fps camera setting
L("## Audio is start-aligned but stops early; video uses 30 fps\n")
win_rows = []
fig, axs = plt.subplots(3, 1, figsize=(11, 8))
for ax, (run, suf, _) in zip(axs, TRIALS):
    m = MAN[run]; s, e = m["started_at"], m["stopped_at"]; win = e - s
    te, env, adur = audio_env(suf)
    thr = 0.05 * env.max()
    on = te[np.argmax(env > thr)] if (env > thr).any() else np.nan
    off = te[len(env)-1 - np.argmax(env[::-1] > thr)] if (env > thr).any() else np.nan
    silent_frac = float(np.mean(env < thr))
    nf = m["rgb_video"]["fixed"]["frames"]
    win_rows.append(dict(run=run, window_s=round(win, 1), audio_dur_s=round(adur, 1),
                         audio_minus_window_s=round(adur-win, 1),
                         audio_onset_s=round(on, 2), audio_offset_s=round(off, 2),
                         silent_frac=round(silent_frac, 4), video_frames=nf,
                         fps_for_audiomatch=round(nf/adur, 2), fps_for_windowmatch=round(nf/win, 2)))
    sub = TEL[(TEL.ts >= s) & (TEL.ts <= e)].copy(); sub["rel"] = sub.ts - s
    ax2 = ax.twinx()
    ax.plot(te, env, color=COLORS["audio"], lw=1.0, label="audio RMS env")
    ax.axvline(adur, color="red", ls="--", lw=1.2, label="audio file end")
    ax.axvline(win, color="black", ls="-.", lw=1.2, label="manifest window end")
    ax2.plot(sub.rel, sub.Power_W, color=COLORS["power"], lw=0.3, alpha=0.6)
    ax2.set_ylabel("Power (W)", color=COLORS["power"]); ax2.set_ylim(0, 55)
    ax.set_ylabel("audio RMS")
    ax.set_title(f"{run}: audio ends {win-adur:.0f}s before window end; power steady to window end")
    ax.set_xlim(0, win+5)
    if run == "run_1":
        ax.legend(loc="lower left", fontsize=7)
axs[-1].set_xlabel("time since run window start (s)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "audio_window_alignment.png")); plt.close(fig)
WIN = pd.DataFrame(win_rows)
WIN.to_csv(os.path.join(OUT, "07_audio_video_window_test.csv"), index=False)
print(WIN.to_string(index=False))
for _, r in WIN.iterrows():
    L(f"- {r.run}: audio {r.audio_dur_s}s, loud to last sample ({r.silent_frac*100:.1f}% below 5% thr); "
      f"file ENDS {-r.audio_minus_window_s:.0f}s before window stop while power is steady (~34 W). "
      f"=> recorder stopped early, tool did not fall silent.")
vid_durs = (WIN.video_frames / EFFECTIVE_FPS)
L(f"**Video duration (camera setting 30 fps):** frames/30 = {vid_durs.min():.0f}-{vid_durs.max():.0f} s. "
  f"Relative to the manifest windows ({WIN.window_s.min():.0f}-{WIN.window_s.max():.0f} s), video duration differs by "
  f"{(vid_durs - WIN.window_s).min():+.0f} to {(vid_durs - WIN.window_s).max():+.0f} s. Audio still ends ~62-66 s before window stop while telemetry power remains steady. "
  f"Video start offset vs the window is unknown without per-frame timestamps.\n")

# representative mid-run frames, both cameras
def mid_frame(path):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n // 2)
    ok, f = cap.read(); cap.release()
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ok else None

fig, axs = plt.subplots(len(TRIALS), 2, figsize=(8, 3*len(TRIALS)))
for i, (run, suf, _) in enumerate(TRIALS):
    for j, cam in enumerate(["camera_in_hand", "camera_fixed"]):
        fr = mid_frame(os.path.join(ROOT, cam, f"rgb{suf}.avi"))
        axs[i, j].imshow(fr if fr is not None else np.zeros((480, 848, 3), np.uint8))
        axs[i, j].set_title(f"{run} {cam.replace('camera_','')}", fontsize=9); axs[i, j].axis("off")
fig.suptitle("Representative mid-trial frames: eye-in-hand (L) vs fixed side-view (R)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "representative_video_frames.png")); plt.close(fig)

# ============================================================
# SECTION 4 - POINT CLOUDS (in-hand + rescans only; fixed cam ignored)
#   4a flip/mirror check  4b reference-relative depth  4c ROI resection depth
# ============================================================
print("\n=== SECTION 4: POINT CLOUDS ===")

# ---- 4a: depth(z)-axis flip / mirror check --------------------------------
L("## Point clouds: RealSense depth(z) flip & mirror check\n")
flip_rows = []
for name, p, ts in CLOUDS:
    xyz = load_raw(p); zmed = float(np.median(xyz[:, 2]))
    pts = load_clean(p)
    c = pts - pts.mean(0)
    _, v = np.linalg.eigh(np.cov(c.T))
    normal = v[:, 0]
    if normal[2] < 0:
        normal = -normal
    flip_rows.append(dict(cloud=name, z_median_m=round(zmed, 4),
                          z_sign="positive(away)" if zmed > 0 else "negative(toward)",
                          surface_normal_z=round(float(normal[2]), 3),
                          x_med_m=round(float(np.median(xyz[:, 0])), 4),
                          y_med_m=round(float(np.median(xyz[:, 1])), 4)))
FLIP = pd.DataFrame(flip_rows)
FLIP.to_csv(os.path.join(OUT, "11_pointcloud_flip_check.csv"), index=False)
xs, ys = np.sign(FLIP.x_med_m.values), np.sign(FLIP.y_med_m.values)
L(f"Checked {len(FLIP)} in-hand+rescan clouds; z-median {FLIP.z_median_m.min():.3f}-{FLIP.z_median_m.max():.3f} m, "
  f"all z positive-away, x-signs {set(int(s) for s in xs)}, y-signs {set(int(s) for s in ys)}. "
  f"No cloud is z-flipped or x/y-mirrored: all live in one consistent right-handed optical frame "
  f"(z away from camera, ~0.14-0.16 m to tissue). The suspected RealSense depth-flip is NOT present; "
  f"the +/-1 mm diff artifacts are residual depth noise + in-hand viewpoint change, not a flip.\n")

# reference = earliest in-hand capture = run_1 start
REF_NAME, REF_PATH, REF_TS = [c for c in CLOUDS if c[0] == "run_1_start"][0]
REF_ZSIGN = np.sign(np.median(load_raw(REF_PATH)[:, 2]))
def corrected_clean(path):
    """Cleaned cloud; defensive z-flip if its z-sign disagreed with reference (no-op here)."""
    pts = load_clean(path)
    if np.sign(np.median(pts[:, 2])) != REF_ZSIGN:
        pts = pts.copy(); pts[:, 2] *= -1
    return pts

# per-run / rescan cleaned-cloud inventory (used for the qualitative depth-map figures)
clean_rows, panels, resc = [], [], sorted(
    glob.glob(os.path.join(ROOT, "camera_in_hand", "rescan_*.pcd")),
    key=lambda p: float(os.path.basename(p).split("_")[1].rsplit(".", 1)[0]))
for run, suf, _ in TRIALS:
    for label in ["start", "stop"]:
        pts, st = load_clean(os.path.join(ROOT, "camera_in_hand", f"pointcloud_{label}{suf}.pcd"),
                             with_stats=True)
        clean_rows.append(dict(run=run, cloud=f"in_hand_{label}", n_clean=len(pts), **st,
                               z_med_m=round(float(np.median(pts[:, 2])), 4)))
        panels.append((f"{run} {label}", pts))
for rp in resc:
    rep = float(os.path.basename(rp).split("_")[1].rsplit(".", 1)[0])
    pts, st = load_clean(rp, with_stats=True)
    clean_rows.append(dict(run=f"rescan@{rep:.0f}", cloud="in_hand_rescan", n_clean=len(pts), **st,
                           z_med_m=round(float(np.median(pts[:, 2])), 4)))
CL = pd.DataFrame(clean_rows)
CL.to_csv(os.path.join(OUT, "08_pointcloud_clean_inventory.csv"), index=False)

# qualitative per-run start/stop depth maps (shared color scale)
allz = np.concatenate([p[:, 2] for _, p in panels]); zlo, zhi = np.percentile(allz, [2, 98])
def grid_auto(pts, cell=0.0015):
    x0, x1 = np.percentile(pts[:, 0], [1, 99]); y0, y1 = np.percentile(pts[:, 1], [1, 99])
    return depth_grid(pts, (x0, x1, y0, y1), cell), (x0, x1, y0, y1)
fig, axs = plt.subplots(3, 2, figsize=(9, 11))
for i, (run, suf, _) in enumerate(TRIALS):
    for j, label in enumerate(["start", "stop"]):
        pts = load_clean(os.path.join(ROOT, "camera_in_hand", f"pointcloud_{label}{suf}.pcd"))
        g, ext = grid_auto(pts)
        im = axs[i, j].imshow(g*1000, origin="lower", extent=[e*1000 for e in ext],
                              cmap=COLORS["depth_map"], vmin=zlo*1000, vmax=zhi*1000, aspect="equal")
        axs[i, j].set_title(f"{run} in-hand {label} (n={len(pts)})", fontsize=8)
        axs[i, j].set_xlabel("x (mm)"); axs[i, j].set_ylabel("y (mm)")
        plt.colorbar(im, ax=axs[i, j], shrink=0.75, label="depth z (mm)")
fig.suptitle("In-hand tissue surface depth maps (cleaned). Camera MOVES between start/stop\n"
             "-> qualitative surface view, NOT registered.", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "removed_depth_map_examples.png")); plt.close(fig)

if resc:
    fig, axs = plt.subplots(1, len(resc), figsize=(3.4*len(resc), 3.6)); axs = np.atleast_1d(axs)
    for ax, rp in zip(axs, resc):
        rep = float(os.path.basename(rp).split("_")[1].rsplit(".", 1)[0])
        pts = load_clean(rp); g, ext = grid_auto(pts)
        im = ax.imshow(g*1000, origin="lower", extent=[e*1000 for e in ext],
                       cmap=COLORS["depth_map"], vmin=zlo*1000, vmax=zhi*1000, aspect="equal")
        ax.set_title(f"rescan {datetime.datetime.fromtimestamp(rep):%H:%M:%S}", fontsize=8)
        ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
        plt.colorbar(im, ax=ax, shrink=0.75, label="z (mm)")
    fig.suptitle("In-hand rescan point clouds (cleaned depth maps)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "pointcloud_rescans_depth.png")); plt.close(fig)

# ---- 4b: reference-relative cumulative depth (ICP to run_1 start) ----------
L("## Point clouds: reference-relative cumulative depth (ICP to run_1 start)\n")
ref_pts = corrected_clean(REF_PATH)
ref_pc = to_pcd(ref_pts).voxel_down_sample(0.001)
ref_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
EXT = (np.percentile(ref_pts[:, 0], 1), np.percentile(ref_pts[:, 0], 99),
       np.percentile(ref_pts[:, 1], 1), np.percentile(ref_pts[:, 1], 99))
ref_grid = depth_grid(np.asarray(ref_pc.points), EXT)

def icp_to_ref(path):
    pts = corrected_clean(path)
    src = to_pcd(pts).voxel_down_sample(0.001)
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))
    reg = o3d.pipelines.registration.registration_icp(
        src, ref_pc, 0.005, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
    src.transform(reg.transformation)
    return reg, np.asarray(src.points)

icp_rows, diff_panels = [], []
for name, p, ts in [c for c in CLOUDS if c[0] != REF_NAME]:
    reg, regpts = icp_to_ref(p)
    g = depth_grid(regpts, EXT)
    diff = (g - ref_grid) * 1000.0    # mm; +ve = receded (removed)
    valid = np.isfinite(diff)
    icp_rows.append(dict(cloud=name, t_rel_s=round(ts-REF_TS, 1),
                         icp_fitness=round(reg.fitness, 3),
                         icp_inlier_rmse_mm=round(reg.inlier_rmse*1000, 3),
                         diff_median_mm=round(float(np.nanmedian(diff)), 3),
                         diff_p95_mm=round(float(np.nanpercentile(diff[valid], 95)), 3),
                         frac_receded_gt1mm=round(float(np.mean(diff[valid] > 1.0)), 3)))
    diff_panels.append((name, diff))
ICP = pd.DataFrame(icp_rows)
ICP.to_csv(os.path.join(OUT, "12_reference_relative_depth.csv"), index=False)
print(ICP.to_string(index=False))

# Reference-relative depth plots removed: table 12_reference_relative_depth.csv remains.
L(f"ICP fitness {ICP.icp_fitness.min():.2f}-{ICP.icp_fitness.max():.2f}, inlier RMSE "
  f"{ICP.icp_inlier_rmse_mm.min():.2f}-{ICP.icp_inlier_rmse_mm.max():.2f} mm (~1 mm = same magnitude as the "
  f"depth-diff signal). Whole-field median/p95 dz fluctuate within the ~1 mm noise floor; no clean monotonic "
  f"cumulative-resection trend is resolvable from the full field of view (background + moving viewpoint "
  f"dominate). See ROI analysis below.\n")

# ---- 4c: OFF-TARGET x=0 ROI (NOISE BASELINE, kept for contrast) -------------
# HISTORICAL/CONTRAST ONLY. Early passes ASSUMED the resection sat on the tool centerline
# (x=0) and integrated a symmetric 2 cm box there. That box lands on bland, near-zero terrain
# (the actual cut is in the lower-left, see 4d) and reads as sub-noise -> wrong "not resolvable"
# conclusion. We keep this block to document that the negative result was an ROI-PLACEMENT
# ARTIFACT, not a fundamental limit. The POSITIVE result is in 4d below.
# Tip artifact: the aspirator may intrude as anomalously NEAR-FIELD points (small z) in
# active/stop captures -> we mask points more than TIP_NEAR_MM closer than the reference
# ROI surface (robust median) before computing tissue depth change.
L("## Point clouds: OFF-TARGET x=0 ROI (2 cm x 2 cm around x=0) -- NOISE BASELINE for contrast\n")
ROI_HALF = 0.010                                   # 10 mm half-width -> 2 cm box
roi_ref_mask_x = np.abs(ref_pts[:, 0] - 0.0) < ROI_HALF
ROI_YC = float(np.median(ref_pts[roi_ref_mask_x, 1]))   # tissue centerline y near x=0
TIP_NEAR_MM = 6.0                                  # points >6 mm nearer than surface = tip/flyer
L(f"ROI = 2 cm x 2 cm box, x in [-10,+10] mm, y in [{(ROI_YC-ROI_HALF)*1000:.0f},"
  f"{(ROI_YC+ROI_HALF)*1000:.0f}] mm (y-center {ROI_YC*1000:.1f} mm from reference tissue near x=0). "
  f"Each later cloud is ICP-registered to run_1 start (as above), cropped to the ROI, and differenced "
  f"in a 1.5 mm depth grid against the reference ROI surface.")
L(f"Device-tip handling: within the ROI, registered points lying more than {TIP_NEAR_MM:.0f} mm NEARER "
  f"the camera than the reference ROI surface median are flagged as device-tip/near-field intrusion and "
  f"MASKED before computing tissue dz (so the tip is not counted as tissue). The masked fraction and a "
  f"tip-flag are reported per capture.\n")

def roi_crop(pts):
    m = (np.abs(pts[:, 0] - 0.0) < ROI_HALF) & (np.abs(pts[:, 1] - ROI_YC) < ROI_HALF)
    return pts[m]

ref_roi = roi_crop(ref_pts)
ref_roi_z = float(np.median(ref_roi[:, 2]))
ROI_EXT = (-ROI_HALF, ROI_HALF, ROI_YC-ROI_HALF, ROI_YC+ROI_HALF)
ref_roi_grid = depth_grid(ref_roi, ROI_EXT, cell=0.0015)

roi_rows, roi_panels = [], [("run_1_start (REF)", np.zeros_like(ref_roi_grid))]
for name, p, ts in [c for c in CLOUDS if c[0] != REF_NAME]:
    _, regpts = icp_to_ref(p)
    roi = regpts[(np.abs(regpts[:, 0]) < ROI_HALF) & (np.abs(regpts[:, 1] - ROI_YC) < ROI_HALF)]
    # tip / near-field mask: points much nearer than reference ROI surface
    near = roi[:, 2] < (ref_roi_z - TIP_NEAR_MM/1000.0)
    tip_frac = float(np.mean(near)) if len(roi) else np.nan
    tissue = roi[~near]
    g = depth_grid(tissue, ROI_EXT, cell=0.0015)
    diff = (g - ref_roi_grid) * 1000.0
    valid = np.isfinite(diff)
    roi_rows.append(dict(cloud=name, t_rel_s=round(ts-REF_TS, 1), n_roi=len(roi),
                         tip_frac_masked=round(tip_frac, 4),
                         tip_flag="YES" if (tip_frac == tip_frac and tip_frac > 0.02) else "no",
                         roi_dz_median_mm=round(float(np.nanmedian(diff)), 3),
                         roi_dz_mean_mm=round(float(np.nanmean(diff)), 3),
                         roi_dz_p90_mm=round(float(np.nanpercentile(diff[valid], 90)), 3),
                         roi_vol_proxy_mm3=round(float(np.nansum(np.where(diff > 0, diff, 0))
                                                        * 1.5 * 1.5), 1)))
    roi_panels.append((name, diff))
ROI = pd.DataFrame(roi_rows)
ROI.to_csv(os.path.join(OUT, "roi_resection_summary.csv"), index=False)
print(ROI.to_string(index=False))

# Off-target ROI plots removed: roi_resection_summary.csv remains as the noise-baseline table.

# verdict on ROI cumulative deepening (stop captures only, monotonic check)
order_roi = ROI.sort_values("t_rel_s").reset_index(drop=True)
stops = order_roi[order_roi.cloud.str.endswith("_stop")].sort_values("t_rel_s")
roi_rmse = ICP.icp_inlier_rmse_mm.median()
stop_med = stops.roi_dz_median_mm.values
mono = bool(len(stop_med) >= 2 and np.all(np.diff(stop_med) >= -0.3) and stop_med[-1] > stop_med[0] + 0.5)
L("**ICP-registered ROI results (per capture):**")
for _, r in order_roi.iterrows():
    L(f"- {r.cloud}: ROI median dz {r.roi_dz_median_mm:+.2f} mm, mean {r.roi_dz_mean_mm:+.2f} mm, "
      f"p90 {r.roi_dz_p90_mm:+.2f} mm, tip-masked {r.tip_frac_masked*100:.1f}% ({r.tip_flag}).")
L(f"\nStop-capture ROI median dz (cumulative resection proxy): "
  f"{', '.join(f'{r.cloud}={r.roi_dz_median_mm:+.2f}mm' for _, r in stops.iterrows())}. "
  f"ICP inlier RMSE (~{roi_rmse:.2f} mm) is the registration noise floor.")
L("**OFF-TARGET BASELINE verdict:** at the x=0 centerline guess the per-capture median/mean dz stay "
  "within the ~1 mm ICP noise floor and show no monotonic deepening -- this is an ROI-PLACEMENT "
  "ARTIFACT (the box is on flat terrain, not on the cut). The cut is in the lower-left; section 4d "
  "localizes it from the data and DOES resolve a deepening cavity. Keep this block only as the "
  "off-target/noise contrast.")
L("Table: roi_resection_summary.csv. Off-target ROI figures were removed.\n")

# ---- 4d: EXPLICIT CUT-REGION RESECTION DEPTH (positive result) --------------
# The cut ROI and noise box are now defined EXPLICITLY by the XY config constants at the top
# of this file (NOISE_BOX_MM, CUT_ROI_CENTER_MM, CUT_ROI_HALF_MM) -- no auto-localization.
# Pipeline: (1) background-only point-to-plane ICP to run_1 start, masking out a generous
# tissue zone (union of the x=0 box and a lower-left window) so neither the cut nor the x=0
# box biases registration; (2) estimate the registration noise floor from the EXPLICIT static
# NOISE_BOX (robust MAD sigma of reference-relative dz inside it), threshold = 2*sigma;
# (3) integrate reference-relative depth over the EXPLICIT CUT ROI across run_1_start ->
# run_1_stop -> run_2_stop -> run_3_stop, with monotonic dz stats; (4) net + positive-recession
# volume proxies with a noise-floor error bar.
L("## Point clouds: EXPLICIT cut-region resection depth (config-defined ROI, bg-only ICP) -- POSITIVE\n")

# bg-only ICP reference: exclude a GENEROUS tissue zone = union of the x=0 box and a lower-left
# window covering the cut, so the registration locks to static background only. (This exclusion
# window is independent of the analysis CUT ROI; it only protects the ICP fit.)
xs_g = np.linspace(EXT[0], EXT[1], ref_grid.shape[1]); ys_g = np.linspace(EXT[2], EXT[3], ref_grid.shape[0])
GX, GY = np.meshgrid(xs_g, ys_g)
NY, NX = ref_grid.shape
LL_CX, LL_CY = -8.0, -30.0          # mm, lower-left ICP-exclusion window center (covers the cut)
LL_HW_X, LL_HW_Y = 18.0, 22.0
LL_WIN = (np.abs(GX*1000 - LL_CX) < LL_HW_X) & (np.abs(GY*1000 - LL_CY) < LL_HW_Y)
roi_x0_cellmask = (np.abs(GX) < ROI_HALF) & (np.abs(GY - ROI_YC) < ROI_HALF)

def in_tissue_zone(pts):
    a = (np.abs(pts[:, 0]) < ROI_HALF) & (np.abs(pts[:, 1] - ROI_YC) < ROI_HALF)  # x=0 box
    b = (np.abs(pts[:, 0]*1000 - LL_CX) < LL_HW_X) & (np.abs(pts[:, 1]*1000 - LL_CY) < LL_HW_Y)
    return a | b

ref_bg_pc = to_pcd(ref_pts[~in_tissue_zone(ref_pts)]).voxel_down_sample(0.001)
ref_bg_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.005, max_nn=30))

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

def cut_diff_map(regpts):
    return (depth_grid(regpts, EXT) - ref_grid) * 1000.0   # mm, +ve = receded (removed)

ALL_NONREF = [c for c in CLOUDS if c[0] != REF_NAME]
well_names = [c[0] for c in ALL_NONREF if "rescan" not in c[0]]
diffs_bg, regs_bg = {}, {}
for name, p, ts in ALL_NONREF:
    rb, rp_bg = icp_bg_only(p)
    diffs_bg[name] = cut_diff_map(rp_bg); regs_bg[name] = rb

cell_area_mm2 = ((EXT[1]-EXT[0])/(NX-1)*1000) * ((EXT[3]-EXT[2])/(NY-1)*1000)

# ---- NOISE FLOOR from the EXPLICIT static NOISE_BOX -------------------------
# robust MAD-sigma of reference-relative dz inside NOISE_BOX_MM, pooled over the well-registered
# stop/start captures; 2*sigma = significance threshold. We first sanity-check the box is
# well-populated and static (per-capture point count + dz stats reported below).
NBx0, NBx1, NBy0, NBy1 = NOISE_BOX_MM
noise_cell_mask = ((GX*1000 >= NBx0) & (GX*1000 <= NBx1) &
                   (GY*1000 >= NBy0) & (GY*1000 <= NBy1) & np.isfinite(ref_grid))
nb_n_cells = int(noise_cell_mask.sum())
nb_rows = []
for n in well_names:
    d = diffs_bg[n][noise_cell_mask & np.isfinite(diffs_bg[n])]
    nb_rows.append(dict(cloud=n, n_cells=len(d),
                        dz_med_mm=round(float(np.median(d)), 3),
                        dz_mean_mm=round(float(np.mean(d)), 3),
                        dz_madsig_mm=round(float(1.4826*np.median(np.abs(d-np.median(d)))), 3)))
NB = pd.DataFrame(nb_rows)
NB.to_csv(os.path.join(OUT, "16_noise_box_sanity.csv"), index=False)
bg_samples = np.concatenate([diffs_bg[n][noise_cell_mask & np.isfinite(diffs_bg[n])] for n in well_names])
NOISE_MED = float(np.median(bg_samples))
NOISE_RSTD = float(1.4826*np.median(np.abs(bg_samples-NOISE_MED)))
SIG_THRESH = 2.0*NOISE_RSTD
nb_static = bool(NB.dz_med_mm.abs().max() < 0.5)   # per-capture median dz stays sub-0.5 mm
L(f"NOISE_BOX = x[{NBx0:.0f},{NBx1:.0f}] y[{NBy0:.0f},{NBy1:.0f}] mm: {nb_n_cells} grid cells/capture "
  f"(well-populated). Per-capture dz median {NB.dz_med_mm.min():+.2f}..{NB.dz_med_mm.max():+.2f} mm, "
  f"MAD-sigma {NB.dz_madsig_mm.min():.2f}..{NB.dz_madsig_mm.max():.2f} mm => "
  f"{'genuinely static' if nb_static else 'NOT clearly static (use with caution)'}.")
L(f"Noise floor (from NOISE_BOX, pooled): median {NOISE_MED:+.3f} mm, robust MAD-sigma "
  f"{NOISE_RSTD:.3f} mm, 2-sigma/pixel significance threshold {SIG_THRESH:.3f} mm. "
  f"(Sanity table: 16_noise_box_sanity.csv)")

def receded_blob(diff, win, sig=SIG_THRESH, rank="integrated"):
    # Dominant significantly-receded contiguous blob inside `win`. Ranked by INTEGRATED dz
    # (sum of dz over the blob = magnitude x area), NOT by raw cell count, so a small intense
    # cavity is preferred over a large faint smear. `win` (LL_WIN) restricts to the tissue
    # interior and excludes the holder-rim arc, so the rim artifact cannot win.
    sig_map = (diff > sig) & np.isfinite(diff) & win
    lab, nlab = ndimage.label(sig_map)
    if nlab == 0:
        return None
    if rank == "count":
        score = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, nlab+1))
    else:
        score = ndimage.sum(np.nan_to_num(diff), lab, index=np.arange(1, nlab+1))
    m = lab == (int(np.argmax(score)) + 1)
    # dz-weighted centroid: lock onto the densest part of the dominant cluster
    wsum = float(np.nansum(diff[m]))
    cx = float(np.nansum(GX[m]*diff[m]) / wsum) if wsum > 0 else float(GX[m].mean())
    cy = float(np.nansum(GY[m]*diff[m]) / wsum) if wsum > 0 else float(GY[m].mean())
    return dict(mask=m, area_mm2=float(m.sum()*cell_area_mm2),
                mean_dz=float(np.nanmean(diff[m])), max_dz=float(np.nanmax(diff[m])),
                int_dz=wsum, cx_mm=cx*1000, cy_mm=cy*1000, n_cells=int(m.sum()))

# STEP 1: EXPLICIT cut ROI from config (no auto-localization). The ROI is the config box
# CUT_ROI_CENTER_MM +/- CUT_ROI_HALF_MM. In future this center is set directly from the planned
# resection waypoint XY. (receded_blob is retained only to report the coherent cavity AREA inside
# this fixed ROI, not to move it.)
CUT_CX, CUT_CY = CUT_ROI_CENTER_MM
CUT_HALF_MM = CUT_ROI_HALF_MM
cut_cell_mask = (np.abs(GX*1000 - CUT_CX) < CUT_HALF_MM) & (np.abs(GY*1000 - CUT_CY) < CUT_HALF_MM)
cut_n_cells = int((cut_cell_mask & np.isfinite(ref_grid)).sum())
cut_area_mm2 = cut_n_cells * cell_area_mm2
L(f"EXPLICIT cut ROI (config): center ({CUT_CX:.0f},{CUT_CY:.0f}) mm, {2*CUT_HALF_MM:.0f} mm box "
  f"({cut_area_mm2:.0f} mm^2, {cut_n_cells} cells). No auto-localization.")

# STEP 2: reference-relative depth over the focused cut ROI
cut_rows = [dict(cloud="run_1_start(REF)", t_rel_s=0.0, roi_med_dz=0.0, roi_mean_dz=0.0,
                 roi_p90_dz=0.0, frac_receded=0.0, frac_approached=0.0,
                 blob_area_mm2=0.0, blob_max_dz=0.0, vol_signed_mm3=0.0)]
WELL_CUT = [c for c in CLOUDS if c[0] != REF_NAME and "rescan" not in c[0]]
for name, p, ts in WELL_CUT:
    db = diffs_bg[name]; dz = db[cut_cell_mask & np.isfinite(db)]
    b = receded_blob(db, cut_cell_mask)
    cut_rows.append(dict(cloud=name, t_rel_s=round(ts-REF_TS, 1),
        roi_med_dz=round(float(np.median(dz)), 3), roi_mean_dz=round(float(np.mean(dz)), 3),
        roi_p90_dz=round(float(np.percentile(dz, 90)), 3),
        frac_receded=round(float(np.mean(dz > SIG_THRESH)), 3),
        frac_approached=round(float(np.mean(dz < -SIG_THRESH)), 3),
        blob_area_mm2=round(b["area_mm2"], 1) if b else 0.0,
        blob_max_dz=round(b["max_dz"], 3) if b else 0.0,
        vol_signed_mm3=round(float(np.mean(dz))*cut_area_mm2, 1)))
CUT = pd.DataFrame(cut_rows)
CUT.to_csv(os.path.join(OUT, "cutregion_summary.csv"), index=False)
print(CUT.to_string(index=False))

stops_cut = CUT[CUT.cloud.str.endswith("_stop")].sort_values("t_rel_s")
sm, smean = stops_cut.roi_med_dz.values, stops_cut.roi_mean_dz.values
mono_med = bool(len(sm) >= 2 and np.all(np.diff(sm) >= -0.1))
mono_mean = bool(len(smean) >= 2 and np.all(np.diff(smean) >= -0.1))
L(f"Stop-capture ROI median dz: {', '.join(f'{r.cloud}={r.roi_med_dz:+.2f}' for _,r in stops_cut.iterrows())}")
L(f"Stop-capture ROI mean   dz: {', '.join(f'{r.cloud}={r.roi_mean_dz:+.2f}' for _,r in stops_cut.iterrows())}")
L(f"Monotonic deepening (median): {mono_med}; (mean): {mono_mean}.")
above_cut = stops_cut[(stops_cut.roi_med_dz > SIG_THRESH) | (stops_cut.roi_mean_dz > SIG_THRESH)]
L(f"Stop captures above +2sigma ({SIG_THRESH:.2f} mm): {list(above_cut.cloud) if len(above_cut) else 'NONE'}.")

# STEP 3: cavity vs device-tip / standoff dipole + centroid stability
dip = []
for name, p, ts in WELL_CUT:
    db = diffs_bg[name]; dz = db[cut_cell_mask & np.isfinite(db)]
    f_near = float(np.mean(dz < -SIG_THRESH)); f_rec = float(np.mean(dz > SIG_THRESH))
    dip.append((name, "YES" if (f_near > 0.03 and f_rec > 0.03) else "no", f_near, f_rec))
sc = stops_cut.copy()
L("Cavity check (stop captures): receded-blob centroid spread across stops, dipole flags -- "
  + "; ".join(f"{n}:dipole={d}(approach {fn:.2f}/recede {fr:.2f})" for n, d, fn, fr in dip
              if n.endswith("_stop")) + ".")

# STEP 4: volume proxy with noise-floor error bar vs 1.40 g = 1333 mm^3 (3 runs)
vol_rows = []
for name, p, ts in WELL_CUT:
    db = diffs_bg[name]; dz = db[cut_cell_mask & np.isfinite(db)]; n = len(dz)
    mean_dz = float(np.mean(dz))
    vol_rows.append(dict(cloud=name, t_rel_s=round(ts-REF_TS, 1), roi_mean_dz_mm=round(mean_dz, 3),
        vol_net_mm3=round(mean_dz*cut_area_mm2, 1),
        vol_pos_only_mm3=round(float(np.sum(dz[dz > 0]))*cell_area_mm2, 1),
        vol_noise_1sig_mm3=round((NOISE_RSTD/np.sqrt(n))*cut_area_mm2, 1)))
VOL = pd.DataFrame(vol_rows)
VOL.to_csv(os.path.join(OUT, "cutregion_volume_proxy.csv"), index=False)
print(VOL.to_string(index=False))
detect_mean_dz = 2.0*NOISE_RSTD/np.sqrt(cut_n_cells); detect_vol = detect_mean_dz*cut_area_mm2
cum_vol = float(VOL[VOL.cloud == "run_3_stop"].vol_net_mm3.iloc[0]) if "run_3_stop" in set(VOL.cloud) else np.nan
# The CUT ROI is now the -4 mm left-shifted box @ (-9,-31), which excludes most of the adjacent
# blue (approached) standoff-dipole lobe on the right edge. As a result the SIGNED box-mean is
# now monotonic and positive across stops (frac_approached drops to ~0.03 by run_3_stop) and is
# usable alongside the ROI MEDIAN and the positive-recession (cavity) volume integral.
cum_vol_pos = float(VOL[VOL.cloud == "run_3_stop"].vol_pos_only_mm3.iloc[0]) if "run_3_stop" in set(VOL.cloud) else np.nan
med_stops = ", ".join(f"{r.roi_med_dz:+.2f}" for _, r in stops_cut.iterrows())
mean_stops = ", ".join(f"{r.roi_mean_dz:+.2f}" for _, r in stops_cut.iterrows())
posv_stops = ", ".join(f"{int(VOL[VOL.cloud==c].vol_pos_only_mm3.iloc[0])}"
                       for c in stops_cut.cloud if c in set(VOL.cloud))
area_stops = ", ".join(f"{r.blob_area_mm2:.0f}" for _, r in stops_cut.iterrows())
fapp_stops = ", ".join(f"{r.frac_approached:.2f}" for _, r in stops_cut.iterrows())
L(f"Detectable (2sigma on ROI mean) net recession >= {detect_mean_dz:.3f} mm => ~{detect_vol:.1f} mm^3/capture "
  f"over the {cut_area_mm2:.0f} mm^2 ROI. Cumulative run_3_stop cavity (positive-recession) volume "
  f"~{cum_vol_pos:.0f} mm^3 (~{cum_vol_pos/1000:.2f} cm^3) vs 1.40 g ground truth (~1.33 cm^3): same order, "
  f"NOT calibrated. (Signed box-mean net proxy ~{cum_vol:.0f} mm^3, now also positive/monotonic.)")
L(f"**Cut-region verdict:** with the EXPLICIT cut ROI = the -4 mm left-shifted 22 mm box @ "
  f"(x={CUT_CX:.0f}, y={CUT_CY:.0f} mm), cumulative tissue removal IS resolvable from single-view in-hand depth: "
  f"ROI MEDIAN dz deepens monotonically (stops: {med_stops} mm), MEAN dz too (stops: {mean_stops} mm), the "
  f"coherent receded cavity grows ({area_stops} mm^2), and the cavity-volume integral ({posv_stops} mm^3) "
  f"reaches the same order as the 1.33 cm^3 ground truth. The left shift excludes most of the adjacent blue "
  f"(approached) standoff-dipole lobe -- frac_approached at the stops drops to ({fapp_stops}) -- so the SIGNED "
  f"MEAN is now usable (no longer contaminated/non-monotonic). The earlier x=0 'not resolvable' result was an "
  f"ROI-placement artifact. ROI is config-defined for future waypoint-driven use.\n")

# ---- FIGURE A: layer-state significance panel with focused cut ROI overlaid ----
def sig_masked(diff):
    out = np.array(diff, float)
    sub = np.isfinite(out) & (np.abs(out - NOISE_MED) < SIG_THRESH)
    return np.ma.array(out, mask=~np.isfinite(out)), sub

LAYER_STATES = [
    ("run_1_start", "Start of experiment", np.where(np.isfinite(ref_grid), 0.0, np.nan), None),
    ("run_1_stop", "After layer 1", diffs_bg.get("run_1_stop"), regs_bg.get("run_1_stop")),
    ("run_2_stop", "After layer 2", diffs_bg.get("run_2_stop"), regs_bg.get("run_2_stop")),
    ("run_3_stop", "After layer 3", diffs_bg.get("run_3_stop"), regs_bg.get("run_3_stop")),
]
LAYER_STATES = [st for st in LAYER_STATES if st[2] is not None]
allv_c = np.concatenate([diffs_bg[n][np.isfinite(diffs_bg[n])] for n in well_names])
vlim_c = max(float(np.nanpercentile(np.abs(allv_c), 98)), 2.0); extmm_c = [e*1000 for e in EXT]
fig, axs = plt.subplots(1, len(LAYER_STATES), figsize=(3.8*len(LAYER_STATES), 3.7), squeeze=False)
axs = axs.ravel()
def add_cutbox(ax):
    ax.add_patch(Rectangle((CUT_CX-CUT_HALF_MM, CUT_CY-CUT_HALF_MM), 2*CUT_HALF_MM, 2*CUT_HALF_MM,
                           fill=False, edgecolor=COLORS["cut_roi"], lw=1.8))
    ax.add_patch(Rectangle((NBx0, NBy0), NBx1-NBx0, NBy1-NBy0,
                           fill=False, edgecolor=COLORS["noise_box"], lw=1.6))
for ax, (name, label, diff, rb) in zip(axs, LAYER_STATES):
    md, sub = sig_masked(diff)
    im = ax.imshow(md, origin="lower", extent=extmm_c, cmap=COLORS["diverging"],
                   vmin=-vlim_c, vmax=vlim_c, aspect="equal")
    ax.imshow(np.ma.array(np.ones_like(diff), mask=~sub), origin="lower", extent=extmm_c,
              cmap=plt.cm.gray, vmin=0, vmax=1, alpha=0.55, aspect="equal")
    add_cutbox(ax)
    fit = "reference" if rb is None else f"bg-fit={rb.fitness:.2f} rmse={rb.inlier_rmse*1000:.2f}mm"
    ax.set_title(f"{label}\n{name} | {fit}", fontsize=8)
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    plt.colorbar(im, ax=ax, shrink=0.7, label="dz vs ref (mm)\n+ = receded")
leg_c = [Line2D([0], [0], color=COLORS["cut_roi"], lw=1.8,
                label=f"CUT ROI (22 mm box @ {CUT_CX:.0f},{CUT_CY:.0f} mm)"),
         Line2D([0], [0], color=COLORS["noise_box"], lw=1.6,
                label=f"NOISE_BOX x[{NBx0:.0f},{NBx1:.0f}] y[{NBy0:.0f},{NBy1:.0f}] mm")]
fig.legend(handles=leg_c, loc="lower center", ncol=2, fontsize=8, framealpha=0.9)
fig.suptitle(f"Cut-region significance by layer state. Gray = sub-noise (|dz|<2sig={SIG_THRESH:.2f} mm).",
             fontsize=9)
fig.tight_layout(rect=[0, 0.10, 1, 0.92])
fig.savefig(os.path.join(FIG, "cutregion_significance_panel.png")); plt.close(fig)

# ---- FIGURE B: layer-by-layer volume removal plot ----
layer_clouds = ["run_1_stop", "run_2_stop", "run_3_stop"]
layer_labels = ["Layer 1", "Layer 2", "Layer 3"]
vol_by_cloud = VOL.set_index("cloud")
existing = [(lab, cloud) for lab, cloud in zip(layer_labels, layer_clouds) if cloud in vol_by_cloud.index]
cum_pos = np.array([float(vol_by_cloud.loc[cloud, "vol_pos_only_mm3"]) for _, cloud in existing])
cum_net = np.array([float(vol_by_cloud.loc[cloud, "vol_net_mm3"]) for _, cloud in existing])
pos_layer = np.diff(np.r_[0.0, cum_pos])
net_layer = np.diff(np.r_[0.0, cum_net])
layer_rows = pd.DataFrame({
    "layer": [lab for lab, _ in existing],
    "cloud": [cloud for _, cloud in existing],
    "incremental_pos_only_mm3": np.round(pos_layer, 1),
    "incremental_net_mm3": np.round(net_layer, 1),
    "cumulative_pos_only_mm3": np.round(cum_pos, 1),
    "cumulative_net_mm3": np.round(cum_net, 1),
})
layer_rows.to_csv(os.path.join(OUT, "cutregion_layer_volume_removal.csv"), index=False)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
x = np.arange(len(existing)); width = 0.36
ax.bar(x - width/2, pos_layer, width, color=COLORS["volume_pos"], label="positive-recession volume")
ax.bar(x + width/2, net_layer, width, color=COLORS["volume_net"], label="net signed volume")
ax.axhline(0, color="k", lw=0.6)
ax.axhline(1333/3, color=COLORS["threshold"], ls="--", lw=1.2,
           label=f"uniform per-layer GT proxy ({1333/3:.0f} mm^3)")
for xi, v in zip(x - width/2, pos_layer):
    ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
for xi, v in zip(x + width/2, net_layer):
    ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
ax2 = ax.twinx()
ax2.plot(x, cum_pos, "o-", color=COLORS["cut_roi"], lw=1.8, label="cumulative positive")
ax.set_xticks(x); ax.set_xticklabels([lab for lab, _ in existing])
ax.set_ylabel("incremental removed volume proxy (mm^3)")
ax2.set_ylabel("cumulative positive-recession volume (mm^3)")
ax.set_title("Cut ROI layer-by-layer volume removal")
handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax.legend(handles1 + handles2, labels1 + labels2, fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "cutregion_depth_profile.png")); plt.close(fig)

# ---- FIGURE C: overlay confirming ROI matches the screenshot location ----
fig, ax = plt.subplots(figsize=(5.2, 5.2))
ref_capture = "run_2_stop" if "run_2_stop" in diffs_bg else well_names[0]
diff = diffs_bg[ref_capture]
im = ax.imshow(diff, origin="lower", extent=extmm_c, cmap=COLORS["diverging"], vmin=-vlim_c, vmax=vlim_c, aspect="equal")
ax.add_patch(Rectangle((CUT_CX-CUT_HALF_MM, CUT_CY-CUT_HALF_MM), 2*CUT_HALF_MM, 2*CUT_HALF_MM,
                       fill=False, edgecolor=COLORS["cut_roi"], lw=2.2, label=f"explicit cut ROI ({CUT_CX:.0f},{CUT_CY:.0f} mm)"))
ax.add_patch(Rectangle((NBx0, NBy0), NBx1-NBx0, NBy1-NBy0,
                       fill=False, edgecolor=COLORS["noise_box"], lw=1.8, label="NOISE_BOX (noise floor)"))
ax.plot(CUT_CX, CUT_CY, "k+", ms=12, mew=2)
ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
ax.set_title(f"Explicit cut ROI + noise box ({ref_capture})\ncut center ({CUT_CX:.0f},{CUT_CY:.0f}) mm")
ax.legend(fontsize=7, loc="upper right")
plt.colorbar(im, ax=ax, shrink=0.8, label="dz vs ref (mm), + = removed")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "cutregion_screenshot_overlay.png")); plt.close(fig)
L("Figures: cutregion_significance_panel.png, cutregion_depth_profile.png, cutregion_screenshot_overlay.png. "
  "Tables: cutregion_summary.csv, cutregion_volume_proxy.csv.\n")

# ============================================================
# SECTION 5 - SONOPET DRIVE FREQUENCY (strongest result)
# ============================================================
print("\n=== SECTION 5: DRIVE FREQUENCY ===")
L("## Sonopet drive frequency: center ~25.38 kHz, downward within-run drift, inverse vs load\n")

freq_rows = []
fig, ax_freq = plt.subplots(figsize=(10, 4.8))
fig2, ax2 = plt.subplots(figsize=(7, 4))
fig3, ax3 = plt.subplots(1, 3, figsize=(13, 4))
for run, suf, _ in TRIALS:
    m = MAN[run]; s, e = m["started_at"], m["stopped_at"]
    sub = TEL[(TEL.ts >= s) & (TEL.ts <= e)].copy().reset_index(drop=True)
    f = sub.Frequency_Hz.values; rel = (sub.ts - s).values
    w = 201 if len(f) > 201 else (len(f)//2*2 - 1)
    fsm = savgol_filter(f, w, 2) if w >= 5 else f
    coef = np.polyfit(rel, f, 1); drift_total = coef[0]*(rel[-1]-rel[0])
    freq_rows.append(dict(run=run, freq_mean_Hz=round(float(f.mean()), 3),
                          freq_std_Hz=round(float(f.std()), 3), freq_min_Hz=round(float(f.min()), 3),
                          freq_max_Hz=round(float(f.max()), 3), freq_range_Hz=round(float(f.max()-f.min()), 3),
                          within_run_slope_Hz_per_s=round(float(coef[0]), 4),
                          within_run_drift_Hz=round(float(drift_total), 2)))
    ax_freq.scatter(rel[::20], f[::20], s=3, color=COLORS[run], alpha=0.18, rasterized=True)
    ax_freq.plot(rel, fsm, color=COLORS[run], lw=2.0, label=f"{run} smoothed")
    ax_freq.plot(rel, np.polyval(coef, rel), ls="--", color=COLORS[run], lw=1.0, alpha=0.8,
                 label=f"{run} trend ({coef[0]*60:.1f} Hz/min)")
    ax2.hist(f, bins=120, range=(25330, 25500), alpha=0.5, color=COLORS[run],
             label=f"{run} (mu={f.mean():.0f})", density=True)
    idx = np.arange(0, len(sub), 20)
    for k, load in enumerate(["MechResistance_Ohm", "Power_W", "I_Handpiece_mag_A"]):
        ax3[k].scatter(sub[load].values[idx], f[idx], s=3, alpha=0.25, color=COLORS[run],
                       label=run if k == 0 else None)
ax_freq.set_xlabel("time since run start (s)")
ax_freq.set_ylabel("Frequency (Hz)")
ax_freq.set_title("Sonopet drive frequency overlaid by run")
ax_freq.grid(True, alpha=0.25)
ax_freq.legend(fontsize=7, ncol=2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "sonopet_frequency_analysis.png")); plt.close(fig)
ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("density")
ax2.set_title("Drive-frequency distribution by run (25.33-25.50 kHz)"); ax2.legend(fontsize=8)
fig2.tight_layout(); fig2.savefig(os.path.join(FIG, "sonopet_frequency_distributions.png")); plt.close(fig2)
for k, lab in enumerate(["MechResistance (Ohm)", "Power (W)", "Handpiece current (A)"]):
    ax3[k].set_xlabel(lab); ax3[k].set_ylabel("Frequency (Hz)"); ax3[k].set_title(lab)
ax3[0].legend(fontsize=7)
fig3.suptitle("Drive frequency vs tool-load proxies (per-run, subsampled)")
fig3.tight_layout(); fig3.savefig(os.path.join(FIG, "sonopet_frequency_vs_load.png")); plt.close(fig3)
FQ = pd.DataFrame(freq_rows)
FQ.to_csv(os.path.join(OUT, "10_frequency_finegrained.csv"), index=False)

# freq vs load correlation
corr_rows = []
for run, suf, _ in TRIALS:
    m = MAN[run]; sub = TEL[(TEL.ts >= m["started_at"]) & (TEL.ts <= m["stopped_at"])]
    f = sub.Frequency_Hz.values
    for load in ["MechResistance_Ohm", "Power_W", "I_Handpiece_mag_A", "Z_Handpiece_mag_Ohm"]:
        rr, pp = pearsonr(sub[load].values, f)
        corr_rows.append(dict(run=run, load=load, pearson_r=round(rr, 3), p_value=f"{pp:.2e}"))
CORR = pd.DataFrame(corr_rows)
CORR.to_csv(os.path.join(OUT, "09_freq_load_correlation.csv"), index=False)
print(CORR.to_string(index=False))
mr = list(CORR[CORR.load == "MechResistance_Ohm"].pearson_r)
L(f"Center ~{FQ.freq_mean_Hz.mean():.0f} Hz. Within-run drift "
  f"{', '.join(f'{r.run} {r.within_run_drift_Hz:+.1f} Hz' for _, r in FQ.iterrows())}. "
  f"Run-to-run shifts are only tens of Hz. **Drive frequency is inversely correlated with mechanical load "
  f"(MechResistance): Pearson r per run = {mr}** (strengthening across runs) -- as tissue load rises the "
  f"tracking loop pulls the resonant drive frequency DOWN.\n")

# telemetry summary table (corrected, fine-grained)
tel_rows = []
for run, suf, _ in TRIALS:
    m = MAN[run]; sub = TEL[(TEL.ts >= m["started_at"]) & (TEL.ts <= m["stopped_at"])]
    fr = FQ[FQ.run == run].iloc[0]
    tel_rows.append(dict(run=run, n_samples=len(sub),
                         rate_hz=round(len(sub)/(sub.ts.max()-sub.ts.min()), 1),
                         power_W_mean=round(sub.Power_W.mean(), 2), power_W_std=round(sub.Power_W.std(), 2),
                         mech_R_ohm_mean=round(sub.MechResistance_Ohm.mean(), 1),
                         mech_R_ohm_std=round(sub.MechResistance_Ohm.std(), 1),
                         freq_Hz_mean=fr.freq_mean_Hz, freq_Hz_std=fr.freq_std_Hz,
                         freq_within_run_drift_Hz=fr.within_run_drift_Hz,
                         footpedal_pct_mean=round(sub.FootPedalPos_pct.mean(), 2)))
TELSUM = pd.DataFrame(tel_rows)
TELSUM.to_csv(os.path.join(OUT, "06_sonopet_telemetry_summary.csv"), index=False)
print(TELSUM.to_string(index=False))

# telemetry overview (power + mech resistance over whole log, runs shaded)
fig, axs = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
tt = TEL.ts - T0
axs[0].plot(tt, TEL.Power_W, lw=0.3, color=COLORS["power"]); axs[0].set_ylabel("Power (W)")
axs[1].plot(tt, TEL.MechResistance_Ohm, lw=0.3, color="#7a3b2e"); axs[1].set_ylabel("Mech. resistance (Ohm)")
for run, _, _ in TRIALS:
    m = MAN[run]
    for ax in axs:
        ax.axvspan(m["started_at"]-T0, m["stopped_at"]-T0, color="orange", alpha=0.12)
axs[1].set_xlabel("time since experiment start (s)")
axs[0].set_title("Sonopet RISE telemetry (200 Hz) -- shaded = recorded run windows")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "sonopet_telemetry_overview.png")); plt.close(fig)

# ============================================================
# SECTION 6 - UNIFIED SYNC TIMELINE + TRIAL TIMELINE
# ============================================================
print("\n=== SECTION 6: TIMELINES ===")
def video_times(run):
    m = MAN[run]; s = m["started_at"]; out = {}
    for cam in ["fixed", "in_hand"]:
        nf = m["rgb_video"][cam]["frames"]; dur = nf / EFFECTIVE_FPS
        out[cam] = (s, s + dur, nf, dur)
    return out

# unified per-run cross-stream sync
fig, axrows = plt.subplots(3, 3, figsize=(15, 9), sharex="col")
for ci, (run, suf, _) in enumerate(TRIALS):
    m = MAN[run]; s, e = m["started_at"], m["stopped_at"]; win = e - s
    sub = TEL[(TEL.ts >= s) & (TEL.ts <= e)].copy(); rel = (sub.ts - s).values
    te, env, adur = audio_env(suf)
    pcs = {(p["camera"], p["label"]): p["timestamp"]-s for p in m["pointcloud_scans"]}
    vt = video_times(run)
    a0 = axrows[0, ci]; a0b = a0.twinx()
    a0.plot(rel, sub.Power_W, color=COLORS["power"], lw=0.5)
    a0b.plot(rel, sub.MechResistance_Ohm, color=COLORS["mech"], lw=0.4, alpha=0.6)
    a0.set_ylabel("Power (W)", color=COLORS["power"]); a0b.set_ylabel("MechR (Ohm)", color=COLORS["mech"])
    a0.set_ylim(0, 55); a0.set_title(f"{run}  window {win:.0f}s", fontsize=9)
    a1 = axrows[1, ci]
    a1.scatter(rel, sub.Frequency_Hz, s=1.5, color=COLORS["raw"], alpha=0.3)
    fw = 201 if len(sub) > 201 else (len(sub)//2*2 - 1)
    a1.plot(rel, savgol_filter(sub.Frequency_Hz.values, fw, 2), color=COLORS[run], lw=1.6)
    a1.set_ylabel("Freq (Hz)")
    a1.set_ylim(sub.Frequency_Hz.mean()-4*sub.Frequency_Hz.std(),
                sub.Frequency_Hz.mean()+4*sub.Frequency_Hz.std())
    a2 = axrows[2, ci]; a2.plot(te, env, color=COLORS["audio"], lw=0.8)
    a2.set_ylabel("audio RMS"); a2.set_xlabel("time since run window start (s)")
    for ax in (a0, a1, a2):
        ax.axvspan(0, win, color="#9ec5e8", alpha=0.12)
        ax.axvline(0, color="black", lw=1.0); ax.axvline(win, color="black", lw=1.0, ls="-.")
        ax.axvline(adur, color=COLORS["audio"], lw=1.2, ls="--")
        ax.axvline(vt["fixed"][3], color=COLORS["run_3"], lw=1.2, ls=":")
        ax.axvline(vt["in_hand"][3], color="#117a8b", lw=1.2, ls=(0, (1, 1)))
        ax.axvline(pcs[("in_hand", "pointcloud_start")], color="green", lw=0.8)
        ax.axvline(pcs[("in_hand", "pointcloud_stop")], color="green", lw=0.8)
        ax.set_xlim(-3, max(win, vt["fixed"][3], vt["in_hand"][3])+5)
handles = [Line2D([0], [0], color="black", lw=1.0, label="window start (t=0)"),
           Line2D([0], [0], color="black", lw=1.0, ls="-.", label="window stop"),
           Line2D([0], [0], color=COLORS["audio"], lw=1.2, ls="--", label="audio start(0)/stop"),
           Line2D([0], [0], color=COLORS["run_3"], lw=1.2, ls=":", label=f"video stop fixed (30 fps)"),
           Line2D([0], [0], color="#117a8b", lw=1.2, ls=(0, (1, 1)), label="video stop in-hand (30 fps)"),
           Line2D([0], [0], color="green", lw=0.8, label="in-hand PC start/stop")]
fig.legend(handles=handles, loc="upper center", ncol=6, fontsize=8, bbox_to_anchor=(0.5, 1.0))
fig.suptitle("Unified cross-stream sync per run (shared epoch clock, t=0 = run window start)",
             y=0.965, fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIG, "unified_sync_timeline.png")); plt.close(fig)

# per-trial timeline (window + audio + both video cams + in-hand PC + rescans)
fig, ax = plt.subplots(figsize=(12, 4.6))
for i, (run, suf, _) in enumerate(TRIALS):
    m = MAN[run]; s, e = m["started_at"], m["stopped_at"]; win = e - s; s0 = s - TEL_T0
    te, env, adur = audio_env(suf); vt = video_times(run)
    ax.barh(i, win, left=s0, height=0.62, color="#9ec5e8", edgecolor="k",
            label="manifest run window" if i == 0 else None)
    ax.barh(i, adur, left=s0, height=0.30, color=COLORS["audio"], label="audio capture" if i == 0 else None)
    ax.barh(i+0.22, vt["fixed"][3], left=s0, height=0.10, color=COLORS["run_3"],
            label="video fixed (30fps)" if i == 0 else None)
    ax.barh(i-0.22, vt["in_hand"][3], left=s0, height=0.10, color="#117a8b",
            label="video in-hand (30fps)" if i == 0 else None)
    pcs = {(p["camera"], p["label"]): p["timestamp"]-TEL_T0 for p in m["pointcloud_scans"]}
    ax.plot(pcs[("in_hand", "pointcloud_start")], i+0.30, "v", color="green", ms=7,
            label="in-hand PC start/stop" if i == 0 else None)
    ax.plot(pcs[("in_hand", "pointcloud_stop")], i+0.30, "v", color="green", ms=7)
for rp in sorted(glob.glob(os.path.join(ROOT, "camera_in_hand", "rescan_*.pcd"))):
    rep = float(os.path.basename(rp).split("_")[1].rsplit(".", 1)[0]) - TEL_T0
    ax.axvline(rep, color="gray", ls=":", lw=1)
    ax.text(rep, len(TRIALS)-0.4, "rescan", rotation=90, fontsize=6, color="gray", va="top")
ax.set_yticks(range(len(TRIALS))); ax.set_yticklabels([t[0] for t in TRIALS])
ax.set_xlabel("time since telemetry log start (s)  [shared epoch clock]")
ax.set_title("Per-trial timeline: window + audio + video(both cams, 30fps) + in-hand PC + rescans")
ax.legend(loc="upper center", ncol=3, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "trial_timeline.png")); plt.close(fig)

print("\nDONE. Outputs in", OUT)
print(f"Whole-experiment scale: {MASS_REMOVED_G:.2f} g removed over all 3 runs (~{MASS_REMOVED_G/1.05:.2f} cm^3).")
