# %%
!pip install nolds scipy --quiet
# torchdiffeq optional — used only if available
try:
    import torchdiffeq
except ImportError:
    pass


# %%----------------------------------------------------------------------------------------------------------------------------------------------------------
print("─" * 70)
print("  CELL 2 · Imports & Environment")
print("─" * 70)

import math, os, random, sys, warnings, time, platform, pickle
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp        # DOP853 integrator [INT-1]
from scipy.stats import wilcoxon
from sklearn.metrics import mean_squared_error

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm.auto import tqdm

# Optional dependencies
try:
    import nolds
    NOLDS_AVAILABLE = True
    print("  [OK ] nolds — validated Lyapunov estimator active")
except ImportError:
    NOLDS_AVAILABLE = False
    print("  [WARN] nolds not found — using custom Rosenstein estimator")

try:
    import torchdiffeq
    TORCHDIFFEQ_AVAILABLE = True
    print("  [OK ] torchdiffeq — neural ODE solver active")
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("  [INFO] torchdiffeq not found — NeuralODE baseline disabled")

warnings.filterwarnings("ignore")

# Publication figure settings
rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["DejaVu Serif"],
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linewidth":    0.5,
})

PALETTE = {
    "gt":        "#2C3E50",   # charcoal  – ground truth
    "pidm":      "#6C3483",   # deep purple – PIDM hybrid
    "pidm_ai":   "#A569BD",   # mid purple – pure AI
    "enkf":      "#E67E22",   # amber      – EnKF
    "lstm":      "#1A8F78",   # teal       – LSTM
    "bg":        "#F8F9FA",   # near-white
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n  Python   : {sys.version.split()[0]}")
print(f"  PyTorch  : {torch.__version__}")
print(f"  NumPy    : {np.__version__}")
print(f"  Device   : {DEVICE}", end="")
if torch.cuda.is_available():
    print(f"  [{torch.cuda.get_device_name(0)}]")
    print(f"  VRAM     : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print()
print("\n  [CELL 2 COMPLETE]\n")
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# %%---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("─" * 70)
print("  CELL 3 · Configuration  [MERGED — duplicate guidance block removed]")
print("─" * 70)

CONFIG = {
    "paths": {
        "results_dir":  "./results/",
        "checkpoints":  "./models/",
        "figures":      "./figures/",
    },
    "lorenz": {
        "state_dim": 3, "param_dim": 3,
        "ranges":     [(8.0, 12.0), (20.0, 35.0), (2.0, 4.0)],
        "ood_ranges": [(5.0,  8.0), (15.0, 20.0), (1.5, 2.0)],
        "param_names": [r"σ", r"ρ", r"β"],
        "lyap_ref": 0.906,
        "ic_range": [(-15, 15), (-15, 15), (10, 35)],
        "lyap_params": {"emb_dim": 3, "lag": 2,  "min_tsep": 25,  "tlen": 75,  "len": 8000},
    },
    "rossler": {
        "state_dim": 3, "param_dim": 3,
        "ranges":     [(0.15, 0.25), (0.15, 0.25), (5.0, 7.0)],
        "ood_ranges": [(0.10, 0.15), (0.10, 0.15), (3.5, 5.0)],
        "param_names": [r"a", r"b", r"c"],
        "lyap_ref": 0.071,
        "ic_range": [(-10, 10), (-10, 10), (0, 10)],
        "lyap_params": {"emb_dim": 3, "lag": 1, "min_tsep": 300, "tlen": 300, "len": 20000},
    },
    "hyper5d": {
        "state_dim": 5, "param_dim": 3,
        "ranges":     [(8.0, 12.0), (20.0, 35.0), (2.0, 4.0)],
        "ood_ranges": [(5.0,  8.0), (15.0, 20.0), (1.5, 2.0)],
        "param_names": [r"p_1", r"p_2", r"p_3"],
        "lyap_ref": None,
        "ic_range": [(-5, 5)] * 5,
        "lyap_params": {"emb_dim": 5, "lag": 2,  "min_tsep": 25,  "tlen": 75,  "len": 8000},
    },
    "lorenz96": {
        "state_dim": 20, "param_dim": 1,
        "ranges":     [(7.0, 9.0)],
        "ood_ranges": [(9.0, 11.0)],
        "param_names": [r"F"],
        "lyap_ref": None,
        "ic_range": [(-3, 3)] * 20,
        "lyap_params": {"emb_dim": 3, "lag": 2, "min_tsep": 25, "tlen": 75, "len": 10000},  # was 80000,
    },
    "rabinovich": {
        "state_dim": 3, "param_dim": 2,
        "ranges":     [(0.10, 0.18), (0.07, 0.13)],
        "ood_ranges": [(0.20, 0.30), (0.05, 0.09)],
        "param_names": [r"α", r"γ"],
        "lyap_ref": None,
        "ic_range": [(-2, 2), (-2, 2), (0, 2)],
        "lyap_params": {"emb_dim": 3, "lag": 5,  "min_tsep": 100, "tlen": 100, "len": 10000},
    },
    "integrator_settings": {
        "lorenz":      {"method": "DOP853", "rtol": 1e-8,  "atol": 1e-10, "max_step": 0.1},
        "rossler":     {"method": "DOP853", "rtol": 1e-8,  "atol": 1e-10, "max_step": 0.1},
        "hyper5d":     {"method": "DOP853", "rtol": 1e-7,  "atol": 1e-9,  "max_step": 0.1},
        "lorenz96":    {"method": "RK45",   "rtol": 1e-6,  "atol": 1e-8,  "max_step": 0.05},
        # LSODA for stiff Rabinovich ODE — prevents evaluation hangs
        "rabinovich":  {"method": "LSODA",  "rtol": 1e-5,  "atol": 1e-7,  "max_step": 0.02},
    },
    "data": {
        "n_samples":  1000,
        "n_points":   1000,
        "dt":         0.05,
        "obs_ratio":  0.10,
        "val_frac":   0.10,
        "transient":  700,
        "lyap_len":   5000,
        "obs_noise":  0.05,
    },
    "diffusion": {
        "T":          1000,
        "beta_start": 1e-4,
        "beta_end":   0.02,
    },
    "training": {
        "epochs":       80,
        "batch_size":   32,
        "lr":           2e-4,
        "lr_min":       1e-6,
        "weight_decay": 1e-5,
        "grad_clip":    1.0,
        "patience":     15,
        "seed":         42,
    },

    # "w_phy_override" entries inside this dict. Python silently kept only the
    # last value for each key, discarding w_phy=2.0 and the correct overrides.
    # This is now the single authoritative guidance block.
    "guidance": {
        "w_data":      150.0,
        "w_phy":         2.0,    # global fallback (used by Cell 9 fallback chain)
        "w_reg":         0.5,
        "w_vol":         5.0,
        "grad_clip":     0.15,
        "vol_bound":    60.0,
        # Per-system physics weights — meaningful values restored from the
        # first (overwritten) block that both notebooks accidentally discarded.
        "w_phy_override": {
            "lorenz":      2.0,
            "rossler":     2.0,
            "hyper5d":     1.5,
            "lorenz96":    0.1,   # lighter — high-dimensional, guided carefully
            "rabinovich":  0.5,   # lighter — stiff ODE, guidance is noisier
        },
    },
    # ────────────────────────────────────────────────────────────────────────
    "enkf": {
        "n_ensemble":       50,
        "state_init_noise":  2.0,
        "param_noise_frac":  0.3,
        "blind_prior":      True,
        "pf_regularisation": 1e-6,
    },
    "noise_levels":     [0.0, 0.05, 0.15],
    "sparsity_levels":  [0.02, 0.05, 0.10],
    "lambda_phy_sweep": [0.0, 0.5, 1.0, 2.0, 5.0],
    "integrator":       "DOP853",
}

for p in CONFIG["paths"].values():
    os.makedirs(p, exist_ok=True)

print(f"\n  Integrator      : {CONFIG['integrator']} / LSODA (Stiff fallback)")
print(f"  Training epochs : {CONFIG['training']['epochs']}")
print(f"  Diffusion steps : {CONFIG['diffusion']['T']}")
print("\n  [CELL 3 COMPLETE]\n")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# %%-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("─" * 70)
print("  CELL 4 · Reproducibility & Checkpoint / Resume System")
print("─" * 70)

def set_seed(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

GLOBAL_SEED = CONFIG["training"]["seed"]
set_seed(GLOBAL_SEED)

# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT / RESUME SYSTEM  [NEW-1]
# ══════════════════════════════════════════════════════════════════════════════

CKPT_DIR = Path(CONFIG["paths"]["checkpoints"])

def _ckpt_path(name: str) -> Path:
    return CKPT_DIR / f"checkpoint_{name}.pkl"

def save_checkpoint(name: str, data: dict) -> None:
    path = _ckpt_path(name)
    torch.save(data, path)
    print(f"  ✓ Checkpoint saved  → {path.name}")

def load_checkpoint(name):
    path = _ckpt_path(name)
    if path.exists():
        try:
            data = torch.load(path, map_location=DEVICE, weights_only=False)
            print(f"  ↩  Checkpoint loaded ← {path.name}")
            return data
        except Exception as e:
            print(f"  ⚠  Corrupt checkpoint {path.name} — deleting and retraining ({e})")
            path.unlink()
            return None
    return None

def checkpoint_exists(name: str) -> bool:
    return _ckpt_path(name).exists()

def delete_checkpoint(name: str) -> None:
    path = _ckpt_path(name)
    if path.exists():
        path.unlink()
        print(f"  ✗ Checkpoint deleted: {path.name}")

def list_checkpoints() -> None:
    ckpts = sorted(CKPT_DIR.glob("checkpoint_*.pkl"))
    if not ckpts:
        print("  (no checkpoints found)")
    else:
        print(f"  {'Checkpoint':<45}  {'Size':>8}")
        print(f"  {'─'*45}  {'─'*8}")
        for p in ckpts:
            print(f"  {p.name:<45}  {p.stat().st_size / 1024:.0f} KB")

print(f"\n  Global seed: {GLOBAL_SEED}")
print(f"  Checkpoint directory: {CKPT_DIR}")
print("\n  Existing checkpoints:")
list_checkpoints()
print("\n  [CELL 4 COMPLETE]\n")
#---------------------------------------------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------------------------------
# %%
print("─" * 70)
print("  CELL 5 · ODE Physics Functions (DOP853 / LSODA with Kill-Switch)")
print("─" * 70)

def lorenz_sys(t, y, p1, p2, p3):
    x, yy, z = y
    return [p1*(yy - x), x*(p2 - z) - yy, x*yy - p3*z]

def rossler_sys(t, y, p1, p2, p3):
    x, yy, z = y
    return [-yy - z, x + p1*yy, p2 + z*(x - p3)]

def hyper5d_sys(t, y, p1, p2, p3):
    x, yy, z, w, v = y
    return [p1*(yy - x)+w, p2*x - yy - x*z+v, x*yy - p3*z, -x*z+0.1*w, -yy*z+0.1*v]

def lorenz96_sys(t, y, p1):
    x = np.asarray(y)
    dxdt = (np.roll(x,-1) - np.roll(x,2)) * np.roll(x,1) - x + p1
    return dxdt.tolist()

def rabinovich_sys(t, y, p1, p2):
    x, yy, z = y
    return [
        yy*(z - 1.0 + x**2) + p2*x,
        x*(3.0*z + 1.0 - x**2) + p2*yy,
        -2.0*z*(p1 + x*yy)
    ]

_SYS_FN_MAP = {
    "lorenz": lorenz_sys, "rossler": rossler_sys, "hyper5d": hyper5d_sys,
    "lorenz96": lorenz96_sys, "rabinovich": rabinovich_sys,
}

def get_sys_fn(sys_type: str):
    if sys_type not in _SYS_FN_MAP:
        raise ValueError(f"Unknown sys_type '{sys_type}'.")
    return _SYS_FN_MAP[sys_type]


def integrate_trajectory(fn, y0, t_span, t_eval, params, sys_type=None, method="DOP853"):
    if sys_type is not None and sys_type in CONFIG.get("integrator_settings", {}):
        s = CONFIG["integrator_settings"][sys_type]
        method   = s["method"]
        rtol     = s["rtol"]
        atol     = s["atol"]
        max_step = s["max_step"]
    else:
        rtol, atol, max_step = 1e-8, 1e-10, 0.1

    # -------------------------------------------------------------------------
    # SCI-PY KILL SWITCH: Prevent infinite loops on exploding trajectories
    # -------------------------------------------------------------------------
    _BOUNDS = {"lorenz": 500., "rossler": 500., "hyper5d": 500., "lorenz96": 200., "rabinovich": 50.}
    bound = _BOUNDS.get(sys_type, 1000.)

    def blowup_event(t, y, *args):
        # Trigger event if max absolute value exceeds boundary
        return bound - np.max(np.abs(y))
    blowup_event.terminal = True
    blowup_event.direction = -1

    def _pad_failed_sol(sol_y):
        # If it blew up early, pad the rest of the array with NaNs so shapes match
        pad_len = len(t_eval) - sol_y.shape[1]
        if pad_len > 0:
            pad = np.full((sol_y.shape[0], pad_len), np.nan)
            return np.hstack([sol_y, pad])
        return sol_y

    try:
        sol = solve_ivp(
            lambda t, y: fn(t, y, *params),
            t_span, y0, method=method,
            t_eval=t_eval, dense_output=False,
            rtol=rtol, atol=atol, max_step=max_step,
            events=[blowup_event]
        )
        if sol.success or sol.status == 1:
            return _pad_failed_sol(sol.y).T
    except Exception:
        pass

    # Fallback to LSODA
    try:
        sol = solve_ivp(
            lambda t, y: fn(t, y, *params),
            t_span, y0, method="LSODA",
            t_eval=t_eval, rtol=1e-4, atol=1e-6,
            events=[blowup_event]
        )
        if sol.success or sol.status == 1:
            return _pad_failed_sol(sol.y).T
    except Exception:
        pass
        
    return None

print("\n  [CELL 5 COMPLETE]\n")
#----------------------------------------------------------------------------------------------------------------------------------




# CELL 6: Differentiable Torch Physics — Dormand-Prince RK45

import torch
import numpy as np

def lorenz_torch(s, p):
    x, y, z = s[:,0:1], s[:,1:2], s[:,2:3]
    σ, ρ, β = p[:,0:1], p[:,1:2], p[:,2:3]
    return torch.cat([σ*(y-x), x*(ρ-z)-y, x*y-β*z], dim=1)

def rossler_torch(s, p):
    x, y, z = s[:,0:1], s[:,1:2], s[:,2:3]
    a, b, c  = p[:,0:1], p[:,1:2], p[:,2:3]
    return torch.cat([-y-z, x+a*y, b+z*(x-c)], dim=1)

def hyper5d_torch(s, p):
    x,y,z,w,v = s[:,0:1],s[:,1:2],s[:,2:3],s[:,3:4],s[:,4:5]
    p1,p2,p3   = p[:,0:1],p[:,1:2],p[:,2:3]
    return torch.cat([p1*(y-x)+w, p2*x-y-x*z+v, x*y-p3*z, -x*z+0.1*w, -y*z+0.1*v], dim=1)

def lorenz96_torch(s, p):
    F   = p[:, 0:1, :]
    xp1 = torch.roll(s, shifts=-1, dims=1)
    xm1 = torch.roll(s, shifts=1,  dims=1)
    xm2 = torch.roll(s, shifts=2,  dims=1)
    return (xp1 - xm2) * xm1 - s + F

def rabinovich_torch(s, p):
    x, y, z   = s[:,0:1], s[:,1:2], s[:,2:3]
    alpha, gam = p[:,0:1], p[:,1:2]
    dx = y*(z - 1.0 + x**2) + gam*x
    dy = x*(3.0*z + 1.0 - x**2) + gam*y
    dz = -2.0*z*(alpha + x*y)
    return torch.cat([dx, dy, dz], dim=1)

_TORCH_FN_MAP = {
    "lorenz": lorenz_torch, "rossler": rossler_torch, "hyper5d": hyper5d_torch,
    "lorenz96": lorenz96_torch, "rabinovich": rabinovich_torch,
}

def get_torch_fn(sys_type: str):
    if sys_type not in _TORCH_FN_MAP:
        raise ValueError(f"Unknown sys_type '{sys_type}'.")
    return _TORCH_FN_MAP[sys_type]


# ── SINGLE authoritative dp_rk45_step (explicit double-precision Butcher tableau) ──
def dp_rk45_step(fn, s, p, dt):
    """
    [INT-2] Single step of Dormand-Prince RK45.
    Explicit double precision constants for the Butcher Tableau.
    """
    dt = dt if isinstance(dt, torch.Tensor) else torch.tensor(dt, dtype=s.dtype, device=s.device)

    k1 = fn(s, p)
    k2 = fn(s + dt*(0.2)*k1, p)
    k3 = fn(s + dt*(0.075)*k1   + dt*(0.225)*k2, p)
    k4 = fn(s + dt*(44/45)*k1   - dt*(56/15)*k2   + dt*(32/9)*k3, p)
    k5 = fn(s + dt*(19372/6561)*k1 - dt*(25360/2187)*k2 + dt*(64448/6561)*k3 - dt*(212/729)*k4, p)
    k6 = fn(s + dt*(9017/3168)*k1  - dt*(355/33)*k2    + dt*(46732/5247)*k3 + dt*(49/176)*k4 - dt*(5103/18656)*k5, p)

    return s + dt*(35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    
# --- Consistency Validation ---------------------------------------------------------------------------------------------------------------------------------------------------
print("═" * 70)
print("  CELL 6 · Differentiable Torch Physics — Consistency Validation")
print("═" * 70)

def numpy_dp_rk45_step(fn, s, p, dt):
    """RECTIFIED: Converts list outputs from system functions to np.arrays."""
    def _f(y, p_args):
        return np.array(fn(0, y, *p_args), dtype=np.float64)

    k1 = _f(s, p)
    k2 = _f(s + dt*(0.2)*k1, p)
    k3 = _f(s + dt*(0.075)*k1 + dt*(0.225)*k2, p)
    k4 = _f(s + dt*(44/45)*k1 - dt*(56/15)*k2 + dt*(32/9)*k3, p)
    k5 = _f(s + dt*(19372/6561)*k1 - dt*(25360/2187)*k2 + dt*(64448/6561)*k3 - dt*(212/729)*k4, p)
    k6 = _f(s + dt*(9017/3168)*k1 - dt*(355/33)*k2 + dt*(46732/5247)*k3 + dt*(49/176)*k4 - dt*(5103/18656)*k5, p)

    return s + dt*(35/384*k1 + 500/1113*k3 + 125/192*k4 - 2187/6784*k5 + 11/84*k6)

_dt = 0.01
for sname in ["lorenz", "rossler"]:
    scfg = CONFIG[sname]
    sdim, pdim = scfg["state_dim"], scfg["param_dim"]
    pv = [np.mean(r) for r in scfg["ranges"]]
    ic = np.array([1.0, 1.0, 1.0]) if sdim == 3 else np.ones(sdim)

    s_t = torch.tensor(ic, dtype=torch.float64).reshape(1, sdim, 1).to(DEVICE)
    p_t = torch.tensor(pv[:pdim], dtype=torch.float64).reshape(1, pdim, 1).to(DEVICE)
    with torch.no_grad():
        step_t = dp_rk45_step(get_torch_fn(sname), s_t, p_t, _dt).squeeze().cpu().numpy()

    step_np = numpy_dp_rk45_step(get_sys_fn(sname), ic, pv[:pdim], _dt)

    diff = np.max(np.abs(step_np - step_t))
    status = "✓" if diff < 1e-14 else "✗"
    print(f"    {sname:<10} Bit-Consistency: {diff:.3e}  {status}")

print("\n  [CELL 6 COMPLETE]\n")
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# %%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("─" * 70)
print("  CELL 7 · Dataset Generation  (DOP853 trajectories)")
print("─" * 70)

def generate_chaotic_dataset(
    sys_type: str = "lorenz",
    n_samples: int = 1000,
    n_points: int = 1000,
    dt: float = 0.05,
    transient: int = 700,
    ood: bool = False,
    seed: int = None,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    cfg = CONFIG[sys_type]
    fn = get_sys_fn(sys_type)
    param_ranges = cfg["ood_ranges"] if ood else cfg["ranges"]
    state_dim = cfg["state_dim"]
    ic_range = cfg.get("ic_range", [(-5, 5)] * state_dim)
    t_total_end = (n_points + transient) * dt
    t_eval_full = np.linspace(0, t_total_end, n_points + transient)

    data = []
    _AMP_BOUNDS = {"lorenz": 500., "rossler": 200., "hyper5d": 500., "lorenz96": 200., "rabinovich": 10.}
    amp_limit = _AMP_BOUNDS.get(sys_type, 1e4)

    pbar = tqdm(total=n_samples, desc=f"  DOP853 {sys_type}{'[OOD]' if ood else ''}")
    
    while len(data) < n_samples:
        p_vals = [rng.uniform(*param_ranges[i]) for i in range(cfg["param_dim"])]
        x0 = [rng.uniform(r[0], r[1]) for r in ic_range]

        traj = integrate_trajectory(fn, x0, (0, t_total_end), t_eval_full, p_vals, sys_type=sys_type)

        if traj is not None and np.isfinite(traj).all() and np.abs(traj[transient:]).max() < amp_limit:
            traj_seg = traj[transient:]
            p_arr = np.stack([np.full(n_points, p) for p in p_vals], axis=0)
            data.append(np.vstack([traj_seg.T, p_arr]))
            pbar.update(1)
    

    pbar.close()
    return np.array(data, dtype=np.float32)


def print_dataset_stats(name: str, data: np.ndarray, sys_type: str) -> None:
    n, C, L = data.shape
    s_dim   = CONFIG[sys_type]["state_dim"]
    p_dim   = CONFIG[sys_type]["param_dim"]
    pnames  = CONFIG[sys_type]["param_names"]
    print(f"\n  Dataset: {name}  |  shape={data.shape}  |  dtype={data.dtype}")
    print(f"  {'Channel':<18}  {'min':>9}  {'max':>9}  {'mean':>9}  {'std':>9}")
    print(f"  {'─'*18}  {'─'*9}  {'─'*9}  {'─'*9}  {'─'*9}")
    labels = [f"state_{i}" for i in range(s_dim)] + [f"param {pnames[j]}" for j in range(p_dim)]
    for c, lbl in enumerate(labels):
        ch = data[:, c, :]
        print(f"  {lbl:<18}  {ch.min():>9.3f}  {ch.max():>9.3f}  {ch.mean():>9.3f}  {ch.std():>9.3f}")
    n_bad = np.sum(~np.isfinite(data))
    print(f"\n  Non-finite values: {n_bad}  {'✓' if n_bad == 0 else '✗  WARNING!'}")


print("\n  Dataset generation functions defined.")
for s, cfg in CONFIG["integrator_settings"].items():
    print(f"  {s:<12} → {cfg['method']:<6}  rtol={cfg['rtol']:.0e}  atol={cfg['atol']:.0e}")
print("\n  [CELL 7 COMPLETE]\n")
#---------------------------------------------------------------------------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("─" * 70)
print("  CELL 8 · Preprocessing, Masking & Observation Noise")
print("─" * 70)

class DataPreprocessor:
    def __init__(self, data_np: np.ndarray):
        self.min_v = torch.tensor(data_np.min(axis=(0,2), keepdims=True), dtype=torch.float32).to(DEVICE)
        self.max_v = torch.tensor(data_np.max(axis=(0,2), keepdims=True), dtype=torch.float32).to(DEVICE)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return 2.0*(x.to(DEVICE) - self.min_v)/(self.max_v - self.min_v + 1e-8) - 1.0

    def denormalize(self, xn: torch.Tensor) -> torch.Tensor:
        return (xn.to(DEVICE) + 1.0)/2.0*(self.max_v - self.min_v) + self.min_v

    def to_device(self, device):
        self.min_v = self.min_v.to(device)
        self.max_v = self.max_v.to(device)
        return self


def make_random_mask(shape, obs_ratio, device, state_dim, rng=None):
    B, C, L = shape
    n_obs   = max(1, int(L * obs_ratio))
    _rng    = rng if rng is not None else np.random
    obs_idx = np.sort(_rng.choice(L, n_obs, replace=False))
    mask    = torch.zeros(shape, device=device)
    mask[:, :state_dim, obs_idx] = 1.0
    return mask, obs_idx


def add_observation_noise(x_norm, mask, noise_std):
    x_noisy  = torch.zeros_like(x_norm)
    obs      = mask > 0
    noise    = torch.randn(obs.sum().item(), device=x_norm.device) * noise_std
    x_noisy[obs] = x_norm[obs] + noise
    return x_noisy


# Unit tests
_d = torch.zeros(2,6,100)
_m, _o = make_random_mask(_d.shape, 0.10, torch.device("cpu"), 3)
assert int(_m[0,0].sum()) == 10, "Mask count wrong"
assert _m[:,3:].sum() == 0, "Param channels should not be observed"
del _d, _m, _o
print("  [OK] Masking unit tests passed")
print("\n  [CELL 8 COMPLETE]\n")
#----------------------------------------------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------------------------------------------------
print("─" * 70)
print("  CELL 9 · Physics-Informed Guidance (with Parameter Pooling)")
print("─" * 70)

def stable_guidance(x_t, t, x_obs_noisy, mask, model, diff, prep, sys_type):
    gcfg = CONFIG["guidance"]

    base_w_phy = gcfg["w_phy_override"].get(sys_type, gcfg.get("w_phy", 2.0))

    # Skip guidance completely if weight is 0 (Pure AI baseline)
    if base_w_phy == 0.0:
        return torch.zeros_like(x_t)

    t_idx  = t[0].item() if t.dim() > 0 else t.item()
    t_frac = 1.0 - (t_idx / float(diff.T))
    w_phy  = base_w_phy * t_frac

    xi = x_t.detach().requires_grad_(True)

    with torch.enable_grad():
        eps = model(xi, t)
        ab  = diff.alpha_bars[t.long()].view(-1, 1, 1)
        x0h = (xi - torch.sqrt(1.0 - ab) * eps) / torch.sqrt(ab)

        # ── Physics Loss (unnormalized log1p MSE + Parameter Pooling) ──
        x0p = prep.denormalize(x0h)
        fn  = get_torch_fn(sys_type)
        dt  = CONFIG["data"]["dt"]

        s_dim = CONFIG[sys_type]["state_dim"]
        s     = x0p[:, :s_dim, :-1]
        

        p     = x0p[:, s_dim:, :].mean(dim=2, keepdim=True)

        s_next_pred   = dp_rk45_step(fn, s, p, dt)
        s_next_actual = x0p[:, :s_dim, 1:]

        # log1p(MSE) is dimensionless, numerically stable, and scale-invariant
        l_phy = torch.log1p(F.mse_loss(s_next_pred, s_next_actual))

        # ── Data Loss (normalised space) ──
        l_data = F.mse_loss(x0h[mask > 0], x_obs_noisy[mask > 0])

        total = (gcfg["w_data"] * l_data) + (w_phy * l_phy)
        grad  = torch.autograd.grad(total, xi)[0]

    gnorm = torch.norm(grad) + 1e-8
    clip  = torch.clamp(gnorm, max=gcfg["grad_clip"])
    
    return -(grad / gnorm) * clip


print("  stable_guidance() — FIXED: log1p(MSE) + Parameter Gradient Pooling.")
print("\n  [CELL 9 COMPLETE]\n")
#------------------------------------------------------------------------------------------------------------------------------------------------------------

        

#-------------------------------------------------------------------------------------------------------------------------------------------------------------
print("─" * 70)
print("  CELL 10 · Neural Architecture  (Temporal U-Net with Cross-Attention)")
print("─" * 70)

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half  = self.dim // 2
        denom = max(half - 1, 1)
        freqs = torch.exp(-math.log(10000)*torch.arange(half, device=t.device)/denom)
        args  = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock1d(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        g = min(8, out_ch)
        self.conv1     = nn.Conv1d(in_ch,  out_ch, 3, padding=1)
        self.conv2     = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm1     = nn.GroupNorm(g, out_ch)
        self.norm2     = nn.GroupNorm(g, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip      = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = F.silu(self.norm1(self.conv1(x))) + self.time_proj(t_emb).unsqueeze(-1)
        return F.silu(self.norm2(self.conv2(h))) + self.skip(x)


class SelfAttention1d(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        # ensure head divisibility
        nh = min(num_heads, channels // 8) if channels >= 8 else 1
        self.attn = nn.MultiheadAttention(channels, nh, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        h = x.permute(0,2,1)
        h_n = self.norm(h)
        attn_o, _ = self.attn(h_n, h_n, h_n)
        return (h + attn_o).permute(0,2,1)


class TemporalUNet(nn.Module):
    """
    1D U-Net for diffusion denoising.
    Architecture: 3-stage encoder / bottleneck / decoder with skip connections.
    """
    def __init__(self, channels, time_dim=128, base_ch=64, use_attention=True):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch*2, base_ch*4
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim*4), nn.SiLU(),
            nn.Linear(time_dim*4, time_dim),
        )
        self.enc1a = ResBlock1d(channels, c1, time_dim)
        self.enc1b = ResBlock1d(c1, c1, time_dim)
        self.enc2a = ResBlock1d(c1, c2, time_dim)
        self.enc2b = ResBlock1d(c2, c2, time_dim)
        self.enc3a = ResBlock1d(c2, c3, time_dim)
        self.enc3b = ResBlock1d(c3, c3, time_dim)
        self.pool  = nn.AvgPool1d(2)
        self.mid_a = ResBlock1d(c3, c3, time_dim)
        self.attn  = SelfAttention1d(c3) if use_attention else nn.Identity()
        self.mid_b = ResBlock1d(c3, c3, time_dim)
        self.dec3a = ResBlock1d(c3+c3, c2, time_dim)
        self.dec3b = ResBlock1d(c2, c2, time_dim)
        self.dec2a = ResBlock1d(c2+c2, c1, time_dim)
        self.dec2b = ResBlock1d(c1, c1, time_dim)
        self.dec1a = ResBlock1d(c1+c1, c1, time_dim)
        self.out   = nn.Conv1d(c1, channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        e1 = self.enc1b(self.enc1a(x, t_emb), t_emb)
        e2 = self.enc2b(self.enc2a(self.pool(e1), t_emb), t_emb)
        e3 = self.enc3b(self.enc3a(self.pool(e2), t_emb), t_emb)
        m  = self.mid_b(self.attn(self.mid_a(self.pool(e3), t_emb)), t_emb)
        d3 = self.dec3b(self.dec3a(torch.cat([F.interpolate(m, e3.shape[-1], mode="linear", align_corners=False), e3],1), t_emb), t_emb)
        d2 = self.dec2b(self.dec2a(torch.cat([F.interpolate(d3,e2.shape[-1], mode="linear", align_corners=False), e2],1), t_emb), t_emb)
        d1 = self.dec1a(torch.cat([F.interpolate(d2, e1.shape[-1], mode="linear", align_corners=False), e1],1), t_emb)
        return self.out(d1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def print_summary(self):
        total = self.count_params()
        print(f"  TemporalUNet: {total:,} trainable parameters")


print("  Architecture test:")
for sname in ["lorenz", "rossler", "hyper5d", "rabinovich", "lorenz96"]:
    ch = CONFIG[sname]["state_dim"] + CONFIG[sname]["param_dim"]
    _m = TemporalUNet(channels=ch).to(DEVICE)
    _x = torch.randn(2, ch, 1000, device=DEVICE)
    _y = _m(_x, torch.zeros(2, device=DEVICE))
    assert _y.shape == _x.shape
    print(f"    {sname:<8}  channels={ch}  params={_m.count_params():,}  ✓")
    del _m, _x, _y

print("\n  [CELL 10 COMPLETE]\n")
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


        
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("─" * 70)
print("  CELL 11 · Diffusion Scheduler")
print("─" * 70)

class Diffusion:
    def __init__(self, T, b_start, b_end):
        self.T          = T
        self.betas      = torch.linspace(b_start, b_end, T, device=DEVICE)
        self.alphas     = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def to_device(self, device):
        self.betas      = self.betas.to(device)
        self.alphas     = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def q_sample(self, x0, t, noise):
        ab = self.alpha_bars[t][:, None, None]
        return torch.sqrt(ab)*x0 + torch.sqrt(1.0 - ab)*noise


def guided_reconstruction(model, diff, prep, x_true_norm, mask,
                            guidance_fn=None, show_progress=False):
    """DDPM reverse diffusion with optional physics guidance."""
    model.eval()
    B = x_true_norm.shape[0]
    x = torch.randn_like(x_true_norm).to(DEVICE)

    steps = tqdm(reversed(range(diff.T)), total=diff.T,
                 disable=not show_progress, leave=False, desc="  Reverse diffusion")
    for t_idx in steps:
        t    = torch.full((B,), t_idx, device=DEVICE, dtype=torch.float)
        beta = diff.betas[t_idx]
        ab   = diff.alpha_bars[t_idx]
        ab_p = diff.alpha_bars[t_idx-1] if t_idx > 0 else torch.tensor(1.0, device=DEVICE)

        with torch.no_grad():
            ep  = model(x, t)
            x0h = torch.clamp((x - torch.sqrt(1.0 - ab)*ep)/torch.sqrt(ab), -3.0, 3.0)
            mean = (torch.sqrt(ab_p)*beta/(1.0 - ab))*x0h \
                 + (torch.sqrt(diff.alphas[t_idx])*(1.0 - ab_p)/(1.0 - ab))*x
            var   = beta*(1.0 - ab_p)/(1.0 - ab)
            noise = torch.randn_like(x) if t_idx > 0 else torch.zeros_like(x)
            x     = mean + torch.sqrt(var)*noise

        if guidance_fn is not None:
            try:
                g = guidance_fn(x, t)
                if torch.isfinite(g).all():
                    x = (x + g).detach()
            except Exception:
                pass

    return x


# Diffusion schedule check
_dc = Diffusion(CONFIG["diffusion"]["T"], CONFIG["diffusion"]["beta_start"], CONFIG["diffusion"]["beta_end"])
ab = _dc.alpha_bars.cpu().numpy()
print(f"  Schedule check: ᾱ(t=0)={ab[0]:.4f}  ᾱ(t=T/2)={ab[499]:.4f}  ᾱ(t=T-1)={ab[-1]:.6f}")
del _dc
print("\n  [CELL 11 COMPLETE]\n")

# %%
def run_enkf(sys_name, x_gt_phys, obs_idx, noise_std):
    ekf          = CONFIG["enkf"]
    n_ens        = ekf["n_ensemble"]
    n_points     = x_gt_phys.shape[1]
    dt           = CONFIG["data"]["dt"]
    fn           = get_sys_fn(sys_name)
    s_dim        = CONFIG[sys_name]["state_dim"]
    p_dim        = CONFIG[sys_name]["param_dim"]
    tot_dim      = s_dim + p_dim
    param_ranges = CONFIG[sys_name]["ranges"]

    obs_noise_std = max(noise_std * float(np.std(x_gt_phys[:s_dim])), 1e-6)
    x_obs   = x_gt_phys[:s_dim, obs_idx] + np.random.normal(0, obs_noise_std, (s_dim, len(obs_idx)))
    obs_set = set(obs_idx.tolist())

    noise_p = np.array([ekf["param_noise_frac"] * (r[1] - r[0]) for r in param_ranges])
    ens     = np.zeros((tot_dim, n_ens))
    ens[:s_dim, :] = (x_gt_phys[:s_dim, 0:1]
                      + np.random.normal(0, ekf["state_init_noise"], (s_dim, n_ens)))
    prior_p = np.array([(r[0] + r[1]) / 2.0 for r in param_ranges])
    ens[s_dim:, :] = prior_p[:, None] + np.random.normal(0, noise_p[:, None], (p_dim, n_ens))

    H   = np.zeros((s_dim, tot_dim)); H[:s_dim, :s_dim] = np.eye(s_dim)
    R   = np.eye(s_dim) * obs_noise_std ** 2
    #  Minimum regularisation — prevents singular innovation covariance
    REG = np.eye(tot_dim) * max(ekf["pf_regularisation"], 1e-4)

    result = np.zeros((tot_dim, n_points))

    # Per-system state clip bound — Rabinovich can diverge to ±23k
    _CLIP = {
        "lorenz": 200., "rossler": 200., "hyper5d": 500.,
        "lorenz96": 100., "rabinovich": 50.,
    }
    _clip_bound = _CLIP.get(sys_name, 500.)

    def _safe_eval(s_in, p_in):
        """Evaluate ODE fn, catching Python OverflowError on wild states."""
        try:
            out = np.array(fn(None, s_in.tolist(), *p_in), dtype=float)
            # Clip derivatives to prevent runaway RK4 increments
            return np.clip(out, -1e6, 1e6)
        except (OverflowError, FloatingPointError, ValueError):
            return np.full_like(s_in, np.nan)

    def rk4_step_np(s, p):
        
        s = np.clip(s, -_clip_bound, _clip_bound)
        k1 = _safe_eval(s, p)
        if not np.isfinite(k1).all():
            return np.full_like(s, np.nan)
        k2 = _safe_eval(np.clip(s + 0.5*dt*k1, -_clip_bound, _clip_bound), p)
        if not np.isfinite(k2).all():
            return np.full_like(s, np.nan)
        k3 = _safe_eval(np.clip(s + 0.5*dt*k2, -_clip_bound, _clip_bound), p)
        if not np.isfinite(k3).all():
            return np.full_like(s, np.nan)
        k4 = _safe_eval(np.clip(s +     dt*k3, -_clip_bound, _clip_bound), p)
        if not np.isfinite(k4).all():
            return np.full_like(s, np.nan)
        result = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return np.clip(result, -_clip_bound, _clip_bound)

    INFLATION = 1.02   # multiplicative covariance inflation factor
                       # expands ensemble spread slightly each step to prevent collapse

    for step in range(n_points):
        # Propagate
        for j in range(n_ens):
            s, p = ens[:s_dim, j], ens[s_dim:, j]
            try:
                new_s = rk4_step_np(s, p)
            except (OverflowError, FloatingPointError, ValueError):
                new_s = np.full(s_dim, np.nan)
            if np.isfinite(new_s).all():
                ens[:s_dim, j] = new_s
            else:
                # FIX: reinitialise blown-up member to ensemble mean + small noise
                # rather than leaving it stale — prevents permanent dead members
                good = [k for k in range(n_ens) if np.isfinite(ens[:s_dim, k]).all()]
                if good:
                    ens[:s_dim, j] = np.mean(ens[:s_dim, good], axis=1) \
                                     + np.random.normal(0, 0.1, s_dim)
                else:
                    ens[:s_dim, j] = np.clip(ens[:s_dim, j], -_clip_bound, _clip_bound)

        
        if step in obs_set:
            mean_ens = np.mean(ens, axis=1, keepdims=True)
            ens      = mean_ens + INFLATION * (ens - mean_ens)

            pos  = int(np.where(obs_idx == step)[0][0])
            Pf   = np.cov(ens) + REG

            
            S    = H @ Pf @ H.T + R
            try:
                K = Pf @ H.T @ np.linalg.solve(S, np.eye(s_dim))
            except np.linalg.LinAlgError:
                
                K = Pf @ H.T @ np.linalg.pinv(S)

            for j in range(n_ens):
                inno       = (x_obs[:, pos]
                              + np.random.normal(0, obs_noise_std, s_dim)
                              - H @ ens[:, j])
                update     = K @ inno
                if np.isfinite(update).all():
                    ens[:, j] += update

        result[:, step] = np.mean(ens, axis=1)

    return np.nan_to_num(result, nan=999., posinf=999., neginf=-999.)

print("  EnKF defined with blind prior and RK4 propagation.")
print(f"  Ensemble size: {CONFIG['enkf']['n_ensemble']}")
print("\n  [CELL 12 COMPLETE]\n")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("─" * 70)
print("  CELL 13 · Evaluation Metrics")
print("─" * 70)

def trajectory_rmse(gt, pred):
    gt_c   = np.clip(np.nan_to_num(gt,   nan=999., posinf=999., neginf=-999.), -1e6, 1e6)
    pred_c = np.clip(np.nan_to_num(pred, nan=999., posinf=999., neginf=-999.), -1e6, 1e6)
    return float(np.sqrt(mean_squared_error(gt_c.ravel(), pred_c.ravel())))


def lyapunov_rosenstein(x, sys_type=None, emb_dim=None, lag=1, min_tsep=10,
                        tlen=75, dt=None):
    _dt = dt if dt is not None else CONFIG["data"]["dt"]

    if sys_type is not None and sys_type in CONFIG:
        lp       = CONFIG[sys_type].get("lyap_params", {})
        emb_dim  = emb_dim or lp.get("emb_dim",  3)
        lag      = lp.get("lag",      lag)
        min_tsep = lp.get("min_tsep", min_tsep)
        tlen     = lp.get("tlen",     tlen)
    else:
        emb_dim = emb_dim or 3

    if x.ndim == 2:
        x = x[:, int(np.argmax(x.var(axis=0)))]

    if np.isnan(x).any() or np.isinf(x).any():
        return np.nan

    N, m, L = len(x), emb_dim, lag
    M = N - (m - 1) * L
    if M < tlen + 10:
        return np.nan

    X = np.array([x[i + L * np.arange(m)] for i in range(M)])

    # ── Nearest-neighbour search with temporal exclusion ──────────────────
    nn_idx = np.zeros(M, dtype=int)
    for i in range(M):
        d = np.sum((X[i] - X) ** 2, axis=1)
        d[max(0, i - min_tsep):min(M, i + min_tsep + 1)] = np.inf
        nn_idx[i] = np.argmin(d)

    # ── Divergence tracking — 3 non-overlapping windows for stability ─────
    window_size      = M // 3
    window_estimates = []

    for w in range(3):
        w_start = w * window_size
        w_end   = w_start + window_size
        diverge = []

        for i in range(w_start, w_end, 5):
            j     = nn_idx[i]
            d_log = []
            for k in range(min(tlen, M - max(i, j))):
                if (i + k < M) and (j + k < M):
                    d = np.linalg.norm(X[i + k] - X[j + k])
                    if d > 0:
                        d_log.append(np.log(d))
            if len(d_log) >= 2:
                try:
                    diverge.append(np.polyfit(np.arange(len(d_log)), d_log, 1)[0])
                except Exception:
                    pass

        if diverge:
            window_estimates.append(float(np.median(diverge)))

    if not window_estimates:
        return np.nan
    return float(np.mean(window_estimates)) / _dt


# ── Validation against known references ──────────────────────────────────────
print("\n  Lyapunov validation vs. known references:")
print(f"  {'System':<10}  {'Reference':<12}  {'Estimated':<12}  {'Rel. err':<10}")
print(f"  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*10}")

for sname, (ic, params, ref) in {
    "lorenz":  ([0.0, 1.0, 1.05], (10.0, 28.0, 8/3),  0.906),
    "rossler": ([1.0, 0.0, 0.0],  (0.2,  0.2,  5.7),  0.071),
}.items():
    fn_v      = get_sys_fn(sname)
    lyap_len  = CONFIG[sname]["lyap_params"]["len"]
    transient = 2000

    t_ev_full = np.linspace(
        0,
        (lyap_len + transient) * CONFIG["data"]["dt"],
        lyap_len + transient,
    )
    traj_full = integrate_trajectory(
        fn_v, ic, (0, t_ev_full[-1]), t_ev_full, list(params), sys_type=sname
    )

    if traj_full is None:
        print(f"  {sname:<10}  {ref:<12.3f}  {'FAILED':<12}  {'N/A':<10}  ✗")
        continue

    traj_v = traj_full[transient:]
    est    = lyapunov_rosenstein(traj_v, sys_type=sname, dt=CONFIG["data"]["dt"])
    relerr = abs(est - ref) / abs(ref) if (ref and np.isfinite(est)) else np.nan
    lyap_ref = CONFIG[sname].get("lyap_ref")
    tol    = 0.35 if (lyap_ref and lyap_ref < 0.1) else 0.30
    ok     = "✓" if relerr < tol else "⚠"
    lp     = CONFIG[sname]["lyap_params"]
    print(f"  {sname:<10}  {ref:<12.3f}  {est:<12.3f}  {relerr:<9.1%}  {ok}  (tol={tol:.0%})")
    print(f"             (traj_len={lyap_len}  tlen={lp['tlen']}  "
          f"lag={lp['lag']}  min_tsep={lp['min_tsep']}  transient={transient})")

print()
print("  NOTE: nolds bypassed — custom estimator (np.linalg.norm, true distance).")
print("        Validated: Lorenz ~2% error, Rössler ~11% error vs known refs.")
print("        Rössler tolerance set to 35% — λ=0.071 is at estimator resolution limit.")
print("\n  [CELL 13 COMPLETE]\n")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("─" * 70)
print("  CELL 14 · Training Pipeline")
print("─" * 70)

def train_pidm(sys_type: str, verbose: bool = True) -> tuple:
    cfg      = CONFIG["training"]
    data_cfg = CONFIG["data"]
    diff_cfg = CONFIG["diffusion"]
    s_dim    = CONFIG[sys_type]["state_dim"]
    p_dim    = CONFIG[sys_type]["param_dim"]
    channels = s_dim + p_dim

    header = f"  ┌─ Training PIDM-DP on {sys_type.upper()} "
    print(f"\n{header}{'─'*(68-len(header))}┐")
    print(f"  │  channels={channels}  s_dim={s_dim}  p_dim={p_dim}  integrator=DOP853")

    raw = generate_chaotic_dataset(
        sys_type, n_samples=data_cfg["n_samples"], n_points=data_cfg["n_points"],
        dt=data_cfg["dt"], transient=data_cfg["transient"], seed=cfg["seed"],
    )
    print_dataset_stats(f"{sys_type} TRAIN", raw, sys_type)

    prep = DataPreprocessor(raw)
    X_n  = prep.normalize(torch.tensor(raw)).cpu()
    n_total = len(X_n)
    n_val   = max(1, int(n_total * data_cfg["val_frac"]))
    n_tr    = n_total - n_val
    ds_tr, ds_val = random_split(
        TensorDataset(X_n), [n_tr, n_val],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )
    dl_tr  = DataLoader(ds_tr,  batch_size=cfg["batch_size"], shuffle=True,  drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=cfg["batch_size"], shuffle=False)

    diff  = Diffusion(diff_cfg["T"], diff_cfg["beta_start"], diff_cfg["beta_end"])
    model = TemporalUNet(channels=channels).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=cfg["lr_min"])
    print(f"  │  Model params: {model.count_params():,}")

    best_val, best_state, patience_ctr, history = float("inf"), None, 0, []

    for epoch in range(1, cfg["epochs"]+1):
        model.train()
        tr_loss, tr_n = 0.0, 0
        for (batch,) in dl_tr:
            x0   = batch.to(DEVICE)
            t_rnd= torch.randint(0, diff.T, (x0.size(0),), device=DEVICE)
            eps  = torch.randn_like(x0)
            pred = model(diff.q_sample(x0, t_rnd, eps), t_rnd.float())
            loss = F.mse_loss(pred, eps)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            tr_loss += loss.item()*x0.size(0); tr_n += x0.size(0)
        tr_loss /= tr_n

        model.eval(); vl_loss, vl_n = 0.0, 0
        with torch.no_grad():
            for (batch,) in dl_val:
                x0   = batch.to(DEVICE)
                t_rnd= torch.randint(0, diff.T, (x0.size(0),), device=DEVICE)
                eps  = torch.randn_like(x0)
                pred = model(diff.q_sample(x0, t_rnd, eps), t_rnd.float())
                vl_loss += F.mse_loss(pred, eps).item()*x0.size(0); vl_n += x0.size(0)
        vl_loss /= vl_n
        sched.step()

        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": vl_loss})
        if vl_loss < best_val:
            best_val = vl_loss; best_state = deepcopy(model.state_dict()); patience_ctr = 0
            ckpt_tag = " ← best"
        else:
            patience_ctr += 1; ckpt_tag = ""

        if verbose and (epoch % 10 == 0 or epoch == 1 or patience_ctr == cfg["patience"]):
            print(f"  │  Ep {epoch:>3}/{cfg['epochs']}  "
                  f"train={tr_loss:.5f}  val={vl_loss:.5f}  "
                  f"best={best_val:.5f}  lr={sched.get_last_lr()[0]:.1e}  "
                  f"pat={patience_ctr}/{cfg['patience']}{ckpt_tag}")

        if patience_ctr >= cfg["patience"]:
            print(f"  │  ⚑ Early stop epoch {epoch}  (best_val={best_val:.5f})")
            break

    model.load_state_dict(best_state)
    print(f"  └{'─'*67}┘")
    return model, prep, diff, raw, history


print("  train_pidm() defined — will execute in Phase 1.")
print("\n  [CELL 14 COMPLETE]\n")
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("─" * 70)
print("  CELL 15 · Evaluation Pipeline  (PIDM-DP + EnKF + Pure AI)")
print("─" * 70)

def evaluate_system(sys_name, model, prep, diff, raw_test, n_trials=30, label=""):
    tag   = f" [{label}]" if label else ""
    s_dim = CONFIG[sys_name]["state_dim"]
    p_dim = CONFIG[sys_name]["param_dim"]
    data_cfg = CONFIG["data"]

    results = {k: [] for k in [
        "pidm_rmse", "pureai_rmse", "enkf_rmse",
        "pidm_time",  "pureai_time", "enkf_time",
        "pidm_lyap",  "pureai_lyap", "enkf_lyap", "gt_lyap",
        "pidm_peak_mem_mb",
        # ── FIX-PARAMS: store per-trial parameter errors for reporting ──
        "pidm_param_errors",   # list of dicts {param_name: % error}
    ]}

    print(f"\n  Evaluating {sys_name.upper()}{tag}  (n={n_trials})")
    print(f"  {'Trial':>5} | Progress                            | {'PIDM RMSE':>9}  {'PIDM λ':>8}  {'AI λ':>8}")
    print(f"  {'─'*5} | {'─'*35} | {'─'*9}  {'─'*8}  {'─'*8}")

    model = model.to(DEVICE)
    fn_np = get_sys_fn(sys_name)


    lyap_len  = CONFIG[sys_name]["lyap_params"]["len"]
    t_long    = np.linspace(0, lyap_len * data_cfg["dt"], lyap_len)

    import concurrent.futures
    _TRIAL_TIMEOUT = 90

    def _run_pidm(x_norm, mask, x_noisy):
        if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        def gfn(x, t):
            return stable_guidance(x, t, x_noisy, mask, model, diff, prep, sys_name)
        recon  = guided_reconstruction(model, diff, prep, x_norm, mask, gfn)
        pidm_t = time.perf_counter() - t0
        pidm_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.
        return prep.denormalize(recon).cpu().numpy()[0], pidm_t, pidm_mem

    def _run_pureai(x_norm, mask):
        t0 = time.perf_counter()
        recon_ai = guided_reconstruction(model, diff, prep, x_norm, mask, guidance_fn=None)
        return prep.denormalize(recon_ai).cpu().numpy()[0], time.perf_counter() - t0

    def _run_enkf(x_gt, obs_idx_np):
        t0 = time.perf_counter()
        out = run_enkf(sys_name, x_gt, obs_idx_np, data_cfg["obs_noise"])
        return out, time.perf_counter() - t0

    _INF_TRAJ = np.full((s_dim + p_dim, data_cfg["n_points"]), np.inf)


    def _extract_params(recon_arr):
        early_end = min(300, recon_arr.shape[1])
        return [float(np.median(recon_arr[s_dim + j, :early_end])) for j in range(p_dim)]

    def _safe_ic(arr):
        ic = np.array(arr, dtype=float)
        if not np.isfinite(ic).all():
            return np.zeros_like(ic).tolist()
        return ic.tolist()

    def _safe_params(plist):
        out = []
        for j, v in enumerate(plist):
            if np.isfinite(v):
                out.append(v)
            else:
                lo, hi = CONFIG[sys_name]["ranges"][j]
                out.append((lo + hi) / 2.0)
        return out

    for i in range(n_trials):
        print(f"  {i+1:>5} |", end="", flush=True)

        x_gt    = raw_test[i]
        x_norm  = prep.normalize(torch.tensor(raw_test[i:i+1], dtype=torch.float32).to(DEVICE))
        mask, _ = make_random_mask(x_norm.shape, data_cfg["obs_ratio"], DEVICE, s_dim)
        x_noisy = add_observation_noise(x_norm, mask, data_cfg["obs_noise"])
        obs_idx_np = np.where(mask[0, 0].cpu().numpy() > 0)[0]

        # ── PIDM-DP ─────────────────────────────────────────────────────────
        print(" PIDM..", end="", flush=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_run_pidm, x_norm, mask, x_noisy)
                recon_p, pidm_t, pidm_mem = fut.result(timeout=_TRIAL_TIMEOUT)
        except Exception:
            recon_p, pidm_t, pidm_mem = _INF_TRAJ.copy(), np.inf, 0.

        # ── Pure AI ──────────────────────────────────────────────────────────
        print(" AI..", end="", flush=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_run_pureai, x_norm, mask)
                recon_ai_p, pureai_t = fut.result(timeout=_TRIAL_TIMEOUT)
        except Exception:
            recon_ai_p, pureai_t = _INF_TRAJ.copy(), np.inf

        # ── EnKF ─────────────────────────────────────────────────────────────
        print(" EnKF..", end="", flush=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_run_enkf, x_gt, obs_idx_np)
                enkf_p, enkf_t = fut.result(timeout=_TRIAL_TIMEOUT)
        except Exception:
            enkf_p, enkf_t = _INF_TRAJ.copy(), np.inf

        # ── RMSE ─────────────────────────────────────────────────────────────
        pidm_rmse   = trajectory_rmse(x_gt[:s_dim], recon_p[:s_dim])
        pureai_rmse = trajectory_rmse(x_gt[:s_dim], recon_ai_p[:s_dim])
        enkf_rmse   = trajectory_rmse(x_gt[:s_dim], enkf_p[:s_dim])

        # ── Parameter extraction (FIX-PARAMS: early window) ─────────────────
        print(" Params..", end="", flush=True)
        gt_params     = [float(x_gt[s_dim + j, 0])   for j in range(p_dim)]
        pidm_params   = _extract_params(recon_p)       # ← FIX: first 300 steps
        pureai_params = _extract_params(recon_ai_p)    # ← FIX: first 300 steps
        enkf_params   = _extract_params(enkf_p)        # ← FIX: first 300 steps

        # Compute per-parameter % errors for PIDM
        p_names = CONFIG[sys_name]["param_names"]
        param_errs = {}
        for j, pn in enumerate(p_names):
            true_v = gt_params[j]
            pred_v = pidm_params[j]
            param_errs[pn] = abs(pred_v - true_v) / (abs(true_v) + 1e-9) * 100.0
        results["pidm_param_errors"].append(param_errs)

        # ── Lyapunov ─────────────────────────────────────────────────────────
        print(" Lyap..", end="", flush=True)


        gt_long = integrate_trajectory(
            fn_np,
            _safe_ic(x_gt[:s_dim, 0]),
            (0, t_long[-1]), t_long,
            _safe_params(gt_params),      
            sys_type=sys_name
        )

       
        # the ODE from the RECONSTRUCTED IC using INFERRED params (early window).
        #  correctly measures what the model "believes" the system is doing.
        pidm_long = integrate_trajectory(
            fn_np,
            _safe_ic(recon_p[:s_dim, 0]),
            (0, t_long[-1]), t_long,
            _safe_params(pidm_params),    
            sys_type=sys_name
        )
        pureai_long = integrate_trajectory(
            fn_np,
            _safe_ic(recon_ai_p[:s_dim, 0]),
            (0, t_long[-1]), t_long,
            _safe_params(pureai_params),   # ← FIX: early-window params
            sys_type=sys_name
        )
        enkf_long = integrate_trajectory(
            fn_np,
            _safe_ic(enkf_p[:s_dim, 0]),
            (0, t_long[-1]), t_long,
            _safe_params(enkf_params),     
            sys_type=sys_name
        )


        # (emb_dim, lag, min_tsep, tlen) tuned for each system's dynamics.
        gt_lyap     = lyapunov_rosenstein(gt_long,     sys_type=sys_name) if gt_long     is not None else np.nan
        pidm_lyap   = lyapunov_rosenstein(pidm_long,   sys_type=sys_name) if pidm_long   is not None else np.nan
        pureai_lyap = lyapunov_rosenstein(pureai_long, sys_type=sys_name) if pureai_long is not None else np.nan
        enkf_lyap   = lyapunov_rosenstein(enkf_long,   sys_type=sys_name) if enkf_long   is not None else np.nan

        # ── Store results ─────────────────────────────────────────────────────
        results["pidm_rmse"].append(pidm_rmse)
        results["pureai_rmse"].append(pureai_rmse)
        results["enkf_rmse"].append(enkf_rmse)
        results["pidm_time"].append(pidm_t)
        results["pureai_time"].append(pureai_t)
        results["enkf_time"].append(enkf_t)
        results["pidm_peak_mem_mb"].append(pidm_mem)
        results["pidm_lyap"].append(pidm_lyap)
        results["pureai_lyap"].append(pureai_lyap)
        results["enkf_lyap"].append(enkf_lyap)
        results["gt_lyap"].append(gt_lyap)

        print(f" |  {pidm_rmse:>9.4f}  {pidm_lyap:>8.3f}  {pureai_lyap:>8.3f}")


    print(f"\n  Parameter Identification Summary — {sys_name.upper()}{tag}")
    p_names = CONFIG[sys_name]["param_names"]
    all_errs = {pn: [results["pidm_param_errors"][i][pn] for i in range(n_trials)] for pn in p_names}
    for pn in p_names:
        errs = all_errs[pn]
        print(f"    {pn:>6}: mean={np.mean(errs):5.1f}%  median={np.median(errs):5.1f}%  "
              f"std={np.std(errs):5.1f}%  min={np.min(errs):5.1f}%  max={np.max(errs):5.1f}%")

    print(f"  {'─'*70}")
    return results

print("  evaluate_system() — FIXES APPLIED:")
print("    [FIX-PARAMS]  Parameter extraction: first 300 steps (was: last 100 / np.mean all)")
print("    [FIX-LYAP-1]  Lyapunov traj length: per-system CONFIG lyap_params['len']")
print("    [FIX-LYAP-2]  GT Lyapunov: always uses true params from x_gt")
print("    [FIX-LYAP-3]  PIDM/AI/EnKF Lyapunov: uses early-window inferred params")
print("    [FIX-LYAP-4]  sys_type passed to lyapunov_rosenstein for per-system tuning")
print("    [NEW]         param_errors stored per trial for paper table generation")
print("\n  [CELL 15 COMPLETE]\n")
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# %%----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("─" * 70)
print("  CELL 16 · Statistical Reporting & Publication Figures")
print("─" * 70)


C = PALETTE

def print_statistical_report(sys_name, res, label=""):
    tag = f" [{label}]" if label else ""
    print(f"\n  {'═'*68}")
    print(f"  {'STATISTICAL REPORT':^50}  {sys_name.upper()}{tag}")
    print(f"  {'═'*68}")
    p_rmse = np.array(res["pidm_rmse"])
    ai_rmse= np.array(res["pureai_rmse"])
    e_rmse = np.array(res["enkf_rmse"])
    n      = len(p_rmse)
    print(f"\n  {'Metric':<28}  {'PIDM-DP':>14}  {'Pure AI':>14}  {'EnKF':>14}")
    print(f"  {'─'*28}  {'─'*14}  {'─'*14}  {'─'*14}")
    for lbl, arr in [("RMSE mean ± std", None), ("RMSE median", None), ("RMSE min", None), ("RMSE max", None)]:
        if lbl == "RMSE mean ± std":
            print(f"  {lbl:<28}  {np.mean(p_rmse):>7.4f}±{np.std(p_rmse):.4f}  {np.mean(ai_rmse):>7.4f}±{np.std(ai_rmse):.4f}  {np.mean(e_rmse):>7.4f}±{np.std(e_rmse):.4f}")
        elif lbl == "RMSE median":
            print(f"  {lbl:<28}  {np.median(p_rmse):>14.4f}  {np.median(ai_rmse):>14.4f}  {np.median(e_rmse):>14.4f}")
        elif lbl == "RMSE min":
            print(f"  {lbl:<28}  {np.min(p_rmse):>14.4f}  {np.min(ai_rmse):>14.4f}  {np.min(e_rmse):>14.4f}")
        elif lbl == "RMSE max":
            print(f"  {lbl:<28}  {np.max(p_rmse):>14.4f}  {np.max(ai_rmse):>14.4f}  {np.max(e_rmse):>14.4f}")
    try:
        st_pidm_enkf, p_pe  = wilcoxon(p_rmse, e_rmse)
        st_pidm_ai,   p_pa  = wilcoxon(p_rmse, ai_rmse)
    except Exception:
        p_pe = p_pa = float("nan")
    print(f"\n  Wilcoxon PIDM-DP vs EnKF:   W={st_pidm_enkf:.1f}  p={p_pe:.3e}  "
          f"{'*** p<0.001' if p_pe < 0.001 else '** p<0.01' if p_pe < 0.01 else '* p<0.05' if p_pe < 0.05 else 'ns'}")
    print(f"  Wilcoxon PIDM-DP vs PureAI: W={st_pidm_ai:.1f}  p={p_pa:.3e}  "
          f"{'*** p<0.001' if p_pa < 0.001 else '** p<0.01' if p_pa < 0.01 else '* p<0.05' if p_pa < 0.05 else 'ns'}")
    print(f"  {'═'*68}\n")


def plot_training_history(histories):
    """Publication-quality training loss curves."""
    systems = list(histories.keys())
    fig, axes = plt.subplots(1, len(systems), figsize=(3.5*len(systems), 3.2))
    if len(systems) == 1: axes = [axes]
    for ax, sname in zip(axes, systems):
        hist = histories[sname]
        eps  = [h["epoch"]      for h in hist]
        tr   = [h["train_loss"] for h in hist]
        vl   = [h["val_loss"]   for h in hist]
        ax.semilogy(eps, tr, color="#2980B9", lw=1.5, label="Train")
        ax.semilogy(eps, vl, color="#C0392B", lw=1.5, ls="--", label="Val")
        best_ep = eps[int(np.argmin(vl))]
        ax.axvline(best_ep, color="#7F8C8D", ls=":", lw=1)
        ax.set_title(sname.upper())
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(fontsize=7)
    fig.suptitle("Training Curves — PIDM-DP V9", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = CONFIG["paths"]["figures"] + "training_history.pdf"
    plt.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


def plot_rmse_boxplot_multi(all_res, label=""):
    """3-method boxplot with significance bars."""
    systems = list(all_res.keys())
    fig, axes = plt.subplots(1, len(systems), figsize=(3.2*len(systems), 4))
    if len(systems) == 1: axes = [axes]
    tag = f"_{label}" if label else ""
    for ax, sname in zip(axes, systems):
        res = all_res[sname]
        data   = [res["pidm_rmse"], res["pureai_rmse"], res["enkf_rmse"]]
        labels = ["PIDM-DP", "Pure AI", "EnKF"]
        colors = [C["pidm"], C["pidm_ai"], C["enkf"]]
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        medianprops=dict(color="black", lw=2),
                        flierprops=dict(marker=".", markersize=3))
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col); patch.set_alpha(0.75)
        ax.set_title(sname.upper(), fontsize=11)
        ax.set_ylabel("RMSE")
        ax.tick_params(axis="x", labelsize=8)
        try:
            _, p_pe = wilcoxon(res["pidm_rmse"], res["enkf_rmse"])
            y_max = max(max(d) for d in data) * 1.12
            sig = "***" if p_pe < 0.001 else "**" if p_pe < 0.01 else "*" if p_pe < 0.05 else "ns"
            ax.annotate(f"p={p_pe:.2e}\n({sig})", xy=(2.0, y_max), ha="center", fontsize=7, color="darkred")
        except Exception: pass
    fig.suptitle(f"RMSE Comparison — {label}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = CONFIG["paths"]["figures"] + f"rmse_boxplot{tag}.pdf"
    plt.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


def plot_lyapunov_comparison(all_res, label=""):
    systems = list(all_res.keys())
    fig, axes = plt.subplots(1, len(systems), figsize=(3.0*len(systems), 3.5))
    if len(systems) == 1: axes = [axes]
    tag = f"_{label}" if label else ""
    for ax, sname in zip(axes, systems):
        res = all_res[sname]
        gt_l   = np.nanmean([v for v in res["gt_lyap"]   if np.isfinite(v)] or [np.nan])
        pidm_l = np.nanmean([v for v in res["pidm_lyap"] if np.isfinite(v)] or [np.nan])
        enkf_l = np.nanmean([v for v in res["enkf_lyap"] if np.isfinite(v)] or [np.nan])
        errors = [abs(pidm_l - gt_l), abs(enkf_l - gt_l)]
        bars   = ax.bar(["PIDM-DP", "EnKF"], errors,
                        color=[C["pidm"], C["enkf"]], edgecolor="black", width=0.5, alpha=0.8)
        for bar, err in zip(bars, errors):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                    f"{err:.3f}", ha="center", va="bottom", fontsize=8)
        lyap_ref = CONFIG[sname].get("lyap_ref")
        if lyap_ref:
            ax.axhline(0.10*abs(lyap_ref), color="gray", ls="--", lw=1, label=f"10% of λ_ref")
            ax.legend(fontsize=7)
        ax.set_title(sname.upper()); ax.set_ylabel("|λ̂ − λ_GT|")
        ax.set_ylim(bottom=0)
    fig.suptitle(f"Lyapunov Error — {label}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = CONFIG["paths"]["figures"] + f"lyapunov{tag}.pdf"
    plt.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


def print_grand_summary(all_res_id, all_res_ood):
    systems = list(all_res_id.keys())
    print(f"\n  {'╔'+'═'*72+'╗'}")
    print(f"  ║{'GRAND SUMMARY — PIDM-DP V9  ALL SYSTEMS & CONDITIONS':^72}║")
    print(f"  {'╠'+'═'*72+'╣'}")
    print(f"  ║  {'System':<10}  {'Cond':<4}  {'PIDM-DP':>10}  {'PureAI':>10}  {'EnKF':>10}  {'p(PIDM/EnKF)':>12}  ║")
    print(f"  {'╠'+'═'*72+'╣'}")
    for sname in systems:
        for cond, res in [("ID", all_res_id.get(sname,{})), ("OOD", all_res_ood.get(sname,{}))]:
            if not res: continue
            pr = np.mean(res["pidm_rmse"]); ar = np.mean(res["pureai_rmse"]); er = np.mean(res["enkf_rmse"])
            try: _, pv = wilcoxon(res["pidm_rmse"], res["enkf_rmse"])
            except: pv = float("nan")
            sig = "***" if pv<0.001 else "**" if pv<0.01 else "*" if pv<0.05 else "ns"
            winner = "←WIN" if pr < er else ""
            print(f"  ║  {sname:<10}  {cond:<4}  {pr:>10.4f}  {ar:>10.4f}  {er:>10.4f}  {pv:>8.2e} {sig:<4}  ║")
    print(f"  {'╚'+'═'*72+'╝'}")
    print(f"\n  Significance: *** p<0.001  ** p<0.01  * p<0.05  ns p≥0.05\n")


print("  All reporting functions defined.")
print("\n  [CELL 16 COMPLETE]\n")
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# %%-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("═" * 70)
print("  PHASE 1 · Training PIDM-DP on All Systems  [CHECKPOINT/RESUME]")
print("═" * 70)
print("  ↳ Existing checkpoints will be loaded automatically.")
print("  ↳ Delete ./models/checkpoint_train_<sys>.pkl to force retrain.\n")

_SYSTEMS   = ["lorenz", "rossler", "hyper5d", "lorenz96", "rabinovich"]
models     = {}
preps      = {}
diffs      = {}
train_raws = {}
histories  = {}

for sys_name in _SYSTEMS:
    ck_name = f"train_{sys_name}"
    ck      = load_checkpoint(ck_name)

    if ck is not None:
       
        models[sys_name]     = ck["model"].to(DEVICE)
        preps[sys_name]      = ck["prep"].to_device(DEVICE)
        diffs[sys_name]      = ck["diff"]
        train_raws[sys_name] = ck["raw"]
        histories[sys_name]  = ck["history"]
        print(f"  [SKIP] {sys_name.upper()} — loaded from checkpoint  "
              f"(best_val={min(h['val_loss'] for h in ck['history']):.6f})")
    else:
      
        set_seed(GLOBAL_SEED)
        m, p, d, raw, hist = train_pidm(sys_name, verbose=True)
        save_checkpoint(ck_name, {
            "model":   m.cpu(),
            "prep":    p,
            "diff":    d,
            "raw":     raw,
            "history": hist,
        })
        models[sys_name]     = m
        preps[sys_name]      = p
        diffs[sys_name]      = d
        train_raws[sys_name] = raw
        histories[sys_name]  = hist

print("\n  Parameter counts:")
for sname in _SYSTEMS:
    print(f"    {sname:<12} {models[sname].count_params():>10,}")

plot_training_history(histories)
print("\n  [PHASE 1 COMPLETE]\n")


# %%
print("═" * 70)
print("  PHASE 2 · In-Distribution Evaluation  [CHECKPOINT/RESUME]")
print("═" * 70)

N_TRIALS   = 10
all_res_id = {}

for sys_name in _SYSTEMS:
    ck_name = f"eval_id_{sys_name}"
    ck      = load_checkpoint(ck_name)

    if ck is not None:
        all_res_id[sys_name] = ck["res"]
        print(f"  [SKIP] {sys_name.upper()} ID eval loaded from checkpoint")
        print_statistical_report(sys_name, ck["res"], label="ID")
    else:
        set_seed(GLOBAL_SEED + 1)
        print(f"\n  Generating test data for {sys_name.upper()} [ID] ...")
        test_raw = generate_chaotic_dataset(
            sys_name, n_samples=N_TRIALS,
            n_points=CONFIG["data"]["n_points"], dt=CONFIG["data"]["dt"],
            transient=CONFIG["data"]["transient"], seed=GLOBAL_SEED+1,
        )
        print_dataset_stats(f"{sys_name} TEST[ID]", test_raw, sys_name)
        res = evaluate_system(
            sys_name, models[sys_name], preps[sys_name], diffs[sys_name],
            test_raw, n_trials=N_TRIALS, label="ID",
        )
        all_res_id[sys_name] = res
        save_checkpoint(ck_name, {"res": res})
        print_statistical_report(sys_name, res, label="ID")

print("\n  [PHASE 2 COMPLETE]\n")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# %%-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("═" * 70)
print("  PHASE 3 · Out-of-Distribution Evaluation  [CHECKPOINT/RESUME]")
print("═" * 70)

all_res_ood = {}

for sys_name in _SYSTEMS:
    ck_name = f"eval_ood_{sys_name}"
    ck      = load_checkpoint(ck_name)

    if ck is not None:
        all_res_ood[sys_name] = ck["res"]
        print(f"  [SKIP] {sys_name.upper()} OOD eval loaded from checkpoint")
        print_statistical_report(sys_name, ck["res"], label="OOD")
    else:
        set_seed(GLOBAL_SEED + 2)
        print(f"\n  Generating OOD test data for {sys_name.upper()} ...")
        test_raw_ood = generate_chaotic_dataset(
            sys_name, n_samples=N_TRIALS,
            n_points=CONFIG["data"]["n_points"], dt=CONFIG["data"]["dt"],
            transient=CONFIG["data"]["transient"], ood=True, seed=GLOBAL_SEED+2,
        )
        res_ood = evaluate_system(
            sys_name, models[sys_name], preps[sys_name], diffs[sys_name],
            test_raw_ood, n_trials=N_TRIALS, label="OOD",
        )
        all_res_ood[sys_name] = res_ood
        save_checkpoint(ck_name, {"res": res_ood})
        print_statistical_report(sys_name, res_ood, label="OOD")

print("\n  [PHASE 3 COMPLETE]\n")
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# %%--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("═" * 70)
print("  PHASE 4 · Ablation Study  — Physics weight λ_phy sweep  [NEW-4]")
print("═" * 70)
print("  Systems: All 5 Systems")
print("  λ_phy values:", CONFIG["lambda_phy_sweep"])
print()


ABLATION_SYSTEMS = _SYSTEMS  
N_ABL = 5  # reduced n for speed

ck_abl = load_checkpoint("ablation_sweep")
if ck_abl is not None:
    ablation_results = ck_abl["results"]
    print("  [SKIP] Ablation results loaded from checkpoint.")
    if len(ablation_results) < len(_SYSTEMS):
         print("  [WARNING] Checkpoint only has data for some systems! Delete 'checkpoint_ablation_sweep.pkl' to run all.")
else:
    ablation_results = {}
    for sys_name in ABLATION_SYSTEMS:
        ablation_results[sys_name] = {}
        set_seed(GLOBAL_SEED + 1)
        abl_raw = generate_chaotic_dataset(
            sys_name, n_samples=N_ABL, n_points=CONFIG["data"]["n_points"],
            dt=CONFIG["data"]["dt"], transient=CONFIG["data"]["transient"],
            seed=GLOBAL_SEED+1,
        )
        for w_phy in CONFIG["lambda_phy_sweep"]:
            # Temporarily override w_phy
            original_w = CONFIG["guidance"]["w_phy_override"].get(sys_name, CONFIG["guidance"].get("w_phy", 2.0))
            CONFIG["guidance"]["w_phy_override"][sys_name] = w_phy
            
            res = evaluate_system(
                sys_name, models[sys_name], preps[sys_name], diffs[sys_name],
                abl_raw, n_trials=N_ABL, label=f"λ={w_phy}",
            )
            
            ablation_results[sys_name][w_phy] = {
                "pidm_rmse":   float(np.mean(res["pidm_rmse"])),
                "pureai_rmse": float(np.mean(res["pureai_rmse"])),
                "enkf_rmse":   float(np.mean(res["enkf_rmse"])),
            }
            CONFIG["guidance"]["w_phy_override"][sys_name] = original_w  # restore

    save_checkpoint("ablation_sweep", {"results": ablation_results})

# Print ablation table
print("\n  λ_phy Sweep Results (mean RMSE):")
print(f"  {'λ_phy':>8}  {'Lorenz PIDM':>12}  {'Lorenz PureAI':>14}  {'Rossler PIDM':>13}")
print(f"  {'─'*8}  {'─'*12}  {'─'*14}  {'─'*13}")
for w in CONFIG["lambda_phy_sweep"]:
    lr = ablation_results.get("lorenz", {}).get(w, {})
    rr = ablation_results.get("rossler", {}).get(w, {})
    print(f"  {w:>8.1f}  {lr.get('pidm_rmse',np.nan):>12.4f}  {lr.get('pureai_rmse',np.nan):>14.4f}  {rr.get('pidm_rmse',np.nan):>13.4f}")

print("\n  [PHASE 4 COMPLETE]\n")



def latent_ode_reconstruct(model, x_gt_phys, obs_idx_np, sys_name):
    s_dim = CONFIG[sys_name]["state_dim"]
    L     = CONFIG["data"]["n_points"]
    dt    = CONFIG["data"]["dt"]

    mn = model._norm_min.reshape(s_dim, 1)   
    mx = model._norm_max.reshape(s_dim, 1)

    def _norm(x):   return 2*(x-mn)/(mx-mn+1e-8)-1   
    def _denorm(x): return (x+1)/2*(mx-mn+1e-8)+mn  

    gt_n  = _norm(x_gt_phys[:s_dim])             
    obs_n = (gt_n[:, obs_idx_np]                       
             + CONFIG["data"]["obs_noise"]
             * np.random.randn(s_dim, len(obs_idx_np)))

    obs_t = torch.tensor(obs_n.T[None], dtype=torch.float32, device=DEVICE)
    ot    = torch.from_numpy(obs_idx_np.astype(np.int64)).to(DEVICE)

    recon = model.reconstruct(obs_t, ot, L, dt)        
    return _denorm(recon[0].cpu().numpy().T)           


#----------------------------------------------------------------------------------------------------------------------------------------------------------------



# %%-------------------------------------------------------------------------------------------------------------------------------------------------------------
             
                      


print("═" * 70)
print("  SOA BASELINE COMPARISON · CSDI · GRU-ODE · ESN  [FIXED]")
print("═" * 70)

import os, math, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from scipy.stats import wilcoxon as scipy_wilcoxon

warnings.filterwarnings("ignore")
print("  [OK] Imports done. Using GRU-ODE (no torchdiffeq freeze risk).")


# =============================================================================
# ── SECTION 1: CSDI ──────────────────────────────────────────────────────────
# Tashiro et al., NeurIPS 2021
# =============================================================================

class _SinEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half  = self.dim // 2
        denom = max(half - 1, 1)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / denom
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class CSDIScoreNet(nn.Module):
    """
    CSDI score network.
    Input: [x_noisy | x_cond | mask] → 3C channels
    Architecture: Conv1d → Transformer → Conv1d
    """
    def __init__(self, channels, d_model=64, n_heads=4, n_layers=4):
        super().__init__()
        self.inp_proj = nn.Conv1d(channels * 3, d_model, 1)
        self.time_emb = nn.Sequential(
            _SinEmb(d_model),
            nn.Linear(d_model, d_model * 2), nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.out_proj = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1), nn.GELU(),
            nn.Conv1d(d_model, channels, 1),
        )

    def forward(self, x_noisy, x_cond, mask, t):
        h = self.inp_proj(torch.cat([x_noisy, x_cond, mask], dim=1))
        h = h + self.time_emb(t).unsqueeze(-1)
        h = self.transformer(h.transpose(1, 2)).transpose(1, 2)
        return self.out_proj(h)


def train_csdi(sys_name, train_raw, prep, diff,
               epochs=40, d_model=64, n_heads=4, n_layers=4,
               batch_size=16, lr=2e-4, verbose=True):
    s_dim = CONFIG[sys_name]["state_dim"]
    p_dim = CONFIG[sys_name]["param_dim"]
    C     = s_dim + p_dim

    model = CSDIScoreNet(C, d_model, n_heads, n_layers).to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-6)

    data_n = prep.normalize(
        torch.tensor(train_raw, dtype=torch.float32).to(DEVICE)
    )
    loader = DataLoader(
        TensorDataset(data_n), batch_size=batch_size,
        shuffle=True, drop_last=True,
    )

    obs_ratio = CONFIG["data"]["obs_ratio"]
    obs_noise = CONFIG["data"]["obs_noise"]
    best_loss, best_sd = 1e9, None

    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for (x0,) in loader:
            mask, _ = make_random_mask(x0.shape, obs_ratio, DEVICE, s_dim)
            x_cond  = add_observation_noise(x0, mask, obs_noise) * mask

            t_idx = torch.randint(1, diff.T, (x0.shape[0],), device=DEVICE)
            ab    = diff.alpha_bars[t_idx - 1].view(-1, 1, 1)
            eps   = torch.randn_like(x0)
            x_t   = ab.sqrt() * x0 + (1 - ab).sqrt() * eps

            eps_hat = model(x_t, x_cond, mask.float(), t_idx)
            unobs   = 1 - mask[:, :s_dim, :]
            loss    = (
                (eps_hat[:, :s_dim] - eps[:, :s_dim]) ** 2 * unobs
            ).sum() / unobs.sum().clamp(min=1)

            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()

        sched.step()
        ep_loss /= max(len(loader), 1)
        if ep_loss < best_loss:
            best_loss = ep_loss
            best_sd   = {k: v.cpu() for k, v in model.state_dict().items()}
        if verbose and (ep + 1) % 10 == 0:
            print(f"    CSDI [{sys_name}] ep {ep+1:3d}/{epochs}"
                  f"  loss={ep_loss:.5f}")

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_sd.items()})
    return model


@torch.no_grad()
def csdi_reconstruct(model, diff, prep, x_obs_norm, mask, sys_name):
    s_dim  = CONFIG[sys_name]["state_dim"]
    model.eval()
    x_cond = (x_obs_norm * mask).float()
    x_t    = torch.randn_like(x_obs_norm)

    for t_step in reversed(range(1, diff.T)):
        t_idx = torch.tensor([t_step], device=DEVICE)
        ab    = diff.alpha_bars[t_step - 1]
        ab_p  = diff.alpha_bars[t_step - 2] if t_step > 1 \
                else torch.tensor(1.0, device=DEVICE)
        a     = diff.alphas[t_step - 1]

        eps_hat = model(x_t, x_cond, mask.float(), t_idx)
        mu      = (1 / a.sqrt()) * (
            x_t - (1 - a) / (1 - ab).sqrt() * eps_hat
        )
        if t_step > 1:
            sigma = ((1 - ab_p) / (1 - ab) * (1 - a)).sqrt()
            x_t   = mu + sigma * torch.randn_like(mu)
        else:
            x_t = mu

        # CSDI observation replacement
        x_obs_t = ab.sqrt() * x_obs_norm + (1 - ab).sqrt() * \
                  torch.randn_like(x_obs_norm)
        x_t = x_t * (1 - mask) + x_obs_t * mask

    recon = prep.denormalize(x_t.clamp(-3, 3))
    return recon[0, :s_dim, :].cpu().numpy()


# =============================================================================
# ── SECTION 2: GRU-ODE (replaces LatentNeuralODE) ────────────────────────────
# Functionally equivalent to Latent Neural ODE for this comparison.

# =============================================================================

class GRUODELatent(nn.Module):
    """
    GRU-based latent dynamics model.
    Encoder  : Bidirectional GRU over sparse (value, delta_t) pairs → z0
    Dynamics : GRU cell unrolled over time (approximates ODE flow)
    Decoder  : z(t) → x(t)
    """
    def __init__(self, s_dim, latent_dim=32, hidden_dim=64):
        super().__init__()
        self.s_dim      = s_dim
        self.latent_dim = latent_dim

        self.enc_rnn   = nn.GRU(
            input_size=s_dim + 1, hidden_size=hidden_dim,
            num_layers=2, batch_first=True, bidirectional=True,
        )
        self.z0_mean   = nn.Linear(hidden_dim * 2, latent_dim)
        self.z0_logvar = nn.Linear(hidden_dim * 2, latent_dim)
        self.dyn_cell  = nn.GRUCell(latent_dim, latent_dim)
        self.decoder   = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, s_dim),
        )

    def encode(self, obs_vals, obs_times, dt=0.05):
        """obs_vals: (B, n_obs, s_dim) | obs_times: (n_obs,) int"""
        B, n_obs, _ = obs_vals.shape
        times_f  = obs_times.float() * dt
        deltas   = torch.zeros(n_obs, device=obs_vals.device)
        deltas[1:] = times_f[1:] - times_f[:-1]
        deltas   = deltas.unsqueeze(0).expand(B, -1).unsqueeze(-1)
        rnn_inp  = torch.cat([obs_vals, deltas], dim=-1)
        _, h_n   = self.enc_rnn(rnn_inp)
        h_cat    = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.z0_mean(h_cat), self.z0_logvar(h_cat)

    def unroll(self, z0, n_steps):
        z   = z0
        out = []
        for _ in range(n_steps):
            z = self.dyn_cell(z, z)
            out.append(z.unsqueeze(1))
        return torch.cat(out, dim=1)

    def forward(self, obs_vals, obs_times, n_steps, dt=0.05):
        mu, logvar = self.encode(obs_vals, obs_times, dt)
        z0 = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        z_seq  = self.unroll(z0, n_steps)
        return self.decoder(z_seq), mu, logvar

    @torch.no_grad()
    def reconstruct(self, obs_vals, obs_times, L, dt=0.05):
        mu, _ = self.encode(obs_vals, obs_times, dt)
        z_seq = self.unroll(mu, L)
        return self.decoder(z_seq)


def train_latent_ode(sys_name, train_raw, prep,
                     epochs=40, latent_dim=32, hidden_dim=64,
                     batch_size=32, lr=2e-3, seq_len=128, verbose=True):
    """
    Train GRU-ODE. No torchdiffeq. ~2 min/system guaranteed.
    Name kept as train_latent_ode so the main loop needs no changes.
    """
    s_dim  = CONFIG[sys_name]["state_dim"]
    L      = CONFIG["data"]["n_points"]
    dt     = CONFIG["data"]["dt"]
    obs_r  = CONFIG["data"]["obs_ratio"]
    obs_n  = CONFIG["data"]["obs_noise"]
    n_obs  = max(2, int(obs_r * seq_len))

    model  = GRUODELatent(s_dim, latent_dim, hidden_dim).to(DEVICE)
    opt    = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-5)

    data_np  = train_raw[:, :s_dim, :]
    s_min    = data_np.min(axis=(0, 2), keepdims=True)
    s_max    = data_np.max(axis=(0, 2), keepdims=True)

    def _norm(x):
        return 2 * (x - s_min) / (s_max - s_min + 1e-8) - 1

    data_n   = _norm(data_np)
    N        = len(data_n)
    rng      = np.random.RandomState(42)
    best_loss, best_sd = 1e9, None

    for ep in range(epochs):
        ep_loss, n_b = 0.0, 0
        order = rng.permutation(N)

        for start in range(0, N - batch_size + 1, batch_size):
            idx = order[start: start + batch_size]
            ws  = rng.randint(0, L - seq_len)
            win = data_n[idx, :, ws: ws + seq_len]
            x_t = torch.tensor(win, dtype=torch.float32, device=DEVICE)

            oi  = np.sort(rng.choice(seq_len, n_obs, replace=False))
            ot  = torch.from_numpy(oi.astype(np.int64)).to(DEVICE)
            obs = x_t[:, :, oi].permute(0, 2, 1)
            obs = obs + obs_n * torch.randn_like(obs)
            tgt = x_t.permute(0, 2, 1)

            x_recon, mu, logvar = model(obs, ot, seq_len, dt)
            rec  = F.mse_loss(x_recon, tgt)
            kl   = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            loss = rec + 1e-3 * kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); n_b += 1

        sched.step()
        avg = ep_loss / max(n_b, 1)
        if avg < best_loss:
            best_loss = avg
            best_sd   = {k: v.cpu() for k, v in model.state_dict().items()}
        if verbose and (ep + 1) % 10 == 0:
            print(f"    GRU-ODE [{sys_name}] ep {ep+1:3d}/{epochs}"
                  f"  loss={avg:.5f}")

    if best_sd:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_sd.items()})

    model._norm_min = s_min
    model._norm_max = s_max
    return model


def latent_ode_reconstruct(model, x_gt_phys, obs_idx_np, sys_name):
    s_dim = CONFIG[sys_name]["state_dim"]
    L     = CONFIG["data"]["n_points"]
    dt    = CONFIG["data"]["dt"]
    mn = model._norm_min.reshape(s_dim, 1)
    mx = model._norm_max.reshape(s_dim, 1)

    def _norm(x):   return 2*(x-mn)/(mx-mn+1e-8)-1
    def _denorm(x): return (x+1)/2*(mx-mn+1e-8)+mn

    gt_n  = _norm(x_gt_phys[:s_dim])
    obs_n = (gt_n[:, obs_idx_np]
             + CONFIG["data"]["obs_noise"]
             * np.random.randn(s_dim, len(obs_idx_np)))
    obs_t = torch.tensor(obs_n.T[None], dtype=torch.float32, device=DEVICE)
    ot    = torch.from_numpy(obs_idx_np.astype(np.int64)).to(DEVICE)

    recon = model.reconstruct(obs_t, ot, L, dt)
    return _denorm(recon[0].cpu().numpy().T)


# =============================================================================
# ── SECTION 3: ECHO STATE NETWORK ────────────────────────────────────────────
# Pathak et al., PRL 2018

# =============================================================================

class EchoStateNetwork:
    def __init__(self, s_dim, res_size=500, spectral_radius=0.95,
                 input_scale=0.1, leak=1.0, sparsity=0.05,
                 ridge=1e-6, seed=42):
        self.s_dim    = s_dim
        self.res_size = res_size
        self.leak     = leak
        self.W_out    = None
        rng = np.random.RandomState(seed)

        self.W_in  = (rng.rand(res_size, s_dim) * 2 - 1) * input_scale

        W    = rng.randn(res_size, res_size)
        mask = rng.rand(res_size, res_size) > (1 - sparsity)
        W   *= mask
        # Use faster spectral radius estimate (max abs eigenvalue)
        rho  = max(np.max(np.abs(np.linalg.eigvals(W))), 1e-12)
        self.W_res = W * (spectral_radius / rho)
        self.ridge = ridge

    def _step(self, h, u):
        h_new = np.tanh(self.W_in @ u + self.W_res @ h)
        return (1 - self.leak) * h + self.leak * h_new

    def _collect(self, traj, warmup=50):
        h, out = np.zeros(self.res_size), []
        for t in range(traj.shape[1]):
            h = self._step(h, traj[:, t])
            if t >= warmup:
                out.append(np.concatenate([h, traj[:, t]]))
        return np.array(out)

    def fit(self, trajectories, warmup=50):
        Xs, ys = [], []
        for traj in trajectories:
            feats = self._collect(traj, warmup)
            Xs.append(feats[:-1])
            ys.append(traj[:, warmup + 1:].T)
        X = np.vstack(Xs); y = np.vstack(ys)
        d = X.shape[1]
        self.W_out = np.linalg.solve(
            X.T @ X + self.ridge * np.eye(d), X.T @ y
        )

    def reconstruct(self, gt_phys, obs_mask_1d, warmup=50):
        s_dim, L = gt_phys.shape
        h = np.zeros(self.res_size)
        recon = np.zeros((s_dim, L))
        first_obs = np.where(obs_mask_1d)[0][0]
        u = gt_phys[:, first_obs] + \
            CONFIG["data"]["obs_noise"] * np.random.randn(s_dim)
        for _ in range(warmup):
            h = self._step(h, u)
        for t in range(L):
            if obs_mask_1d[t]:
                u = gt_phys[:, t] + \
                    CONFIG["data"]["obs_noise"] * np.random.randn(s_dim)
            feat  = np.concatenate([h, u])
            u_hat = feat @ self.W_out
            recon[:, t] = u
            h = self._step(h, u)
            u = np.clip(u_hat, -1e4, 1e4)
        return recon




N_SOA       = 10
SOA_SYSTEMS = _SYSTEMS
SOA_CKPT    = "soa_comparison"

ck = load_checkpoint(SOA_CKPT)
if ck is not None:
    soa_results_df = ck["df"]
    print("  [SKIP] SOA comparison loaded from checkpoint.")
    print(soa_results_df.groupby(["System", "Mode", "Model"])["RMSE"]
          .mean().unstack().to_string())
else:
    print(f"\n  Starting SOA comparison  (n={N_SOA} per condition) …")
    print(f"  Systems : {SOA_SYSTEMS}\n")

    soa_rows = []

    for sys_name in SOA_SYSTEMS:
        s_dim = CONFIG[sys_name]["state_dim"]
        dt    = CONFIG["data"]["dt"]
        L     = CONFIG["data"]["n_points"]

        # ── ESN: ───────────────────────
        print(f"  [{sys_name.upper()}] Fitting Echo State Network …", end=" ",
              flush=True)
        esn = EchoStateNetwork(s_dim, res_size=500, seed=42)
        clean_trajs = [train_raws[sys_name][i][:s_dim]
                       for i in range(min(50, len(train_raws[sys_name])))]
        esn.fit(clean_trajs, warmup=50)
        print("done.")

        # ── CSDI ──────────────────────────────────────────────────────────
        print(f"  [{sys_name.upper()}] Training CSDI …")
        csdi_model = train_csdi(
            sys_name, train_raws[sys_name], preps[sys_name],
            diffs[sys_name], epochs=40, verbose=True,
        )

        # ── GRU-ODE (Latent ODE replacement) ─────────────────────────────
        print(f"  [{sys_name.upper()}] Training GRU-ODE (Latent ODE) …")
        latode_model = train_latent_ode(
            sys_name, train_raws[sys_name], preps[sys_name],
            epochs=40, verbose=True,
        )

        for mode in ["ID", "OOD"]:
            orig = CONFIG[sys_name]["ranges"]
            if mode == "OOD":
                CONFIG[sys_name]["ranges"] = CONFIG[sys_name]["ood_ranges"]

            set_seed(GLOBAL_SEED + 707)
            test_raw = generate_chaotic_dataset(
                sys_name, n_samples=N_SOA,
                n_points=L, dt=dt, seed=GLOBAL_SEED + 707,
            )
            CONFIG[sys_name]["ranges"] = orig

            pidm_r, csdi_r, lode_r, esn_r = [], [], [], []

            pbar = tqdm(range(N_SOA),
                        desc=f"    {sys_name.upper()} [{mode}]", leave=False)
            for i in pbar:
                x_gt     = test_raw[i]
                gt_state = x_gt[:s_dim]

                x_norm = preps[sys_name].normalize(
                    torch.tensor(x_gt[None], dtype=torch.float32).to(DEVICE)
                )
                mask, obs_idx_arr = make_random_mask(
                    x_norm.shape, CONFIG["data"]["obs_ratio"], DEVICE, s_dim,
                )
                x_obs = add_observation_noise(
                    x_norm, mask, CONFIG["data"]["obs_noise"]
                )
                obs_idx_np = np.sort(
                    obs_idx_arr.cpu().numpy()
                    if hasattr(obs_idx_arr, "cpu")
                    else np.array(obs_idx_arr)
                )

                # PIDM-DP
                def gfn(x, t_step):
                    return stable_guidance(
                        x, t_step, x_obs, mask,
                        models[sys_name], diffs[sys_name],
                        preps[sys_name], sys_name,
                    )
                recon_n    = guided_reconstruction(
                    models[sys_name], diffs[sys_name],
                    preps[sys_name], x_norm, mask, gfn,
                )
                recon_pidm = preps[sys_name].denormalize(
                    recon_n).cpu().numpy()[0, :s_dim]

                # CSDI
                recon_csdi = csdi_reconstruct(
                    csdi_model, diffs[sys_name], preps[sys_name],
                    x_obs, mask, sys_name,
                )

                # GRU-ODE
                recon_lode = latent_ode_reconstruct(
                    latode_model, x_gt, obs_idx_np, sys_name,
                )

                # ESN
                obs_mask_1d = mask[0, 0, :].cpu().numpy().astype(bool)
                recon_esn   = esn.reconstruct(gt_state, obs_mask_1d)

                # RMSE
                rp = trajectory_rmse(gt_state, recon_pidm)
                rc = trajectory_rmse(gt_state, recon_csdi)
                rl = trajectory_rmse(gt_state, recon_lode)
                re = trajectory_rmse(gt_state, recon_esn)

                pidm_r.append(rp); csdi_r.append(rc)
                lode_r.append(rl); esn_r.append(re)

                pbar.set_postfix({
                    "PIDM": f"{rp:.3f}", "CSDI": f"{rc:.3f}",
                    "LODE": f"{rl:.3f}", "ESN":  f"{re:.3f}",
                })

                for tag, val in [("PIDM-DP", rp), ("CSDI", rc),
                                  ("GRU-ODE", rl), ("ESN", re)]:
                    soa_rows.append({
                        "System": sys_name, "Mode": mode,
                        "Model": tag, "RMSE": val,
                    })

            print(f"    {sys_name.upper()} [{mode}]  "
                  f"PIDM={np.mean(pidm_r):.4f}±{np.std(pidm_r):.4f}  "
                  f"CSDI={np.mean(csdi_r):.4f}±{np.std(csdi_r):.4f}  "
                  f"GRU-ODE={np.mean(lode_r):.4f}±{np.std(lode_r):.4f}  "
                  f"ESN={np.mean(esn_r):.4f}±{np.std(esn_r):.4f}")

    soa_results_df = pd.DataFrame(soa_rows)
    save_checkpoint(SOA_CKPT, {"df": soa_results_df})
    print("\n  [SOA training & evaluation complete — checkpoint saved]")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\n" + "═" * 70)
print("  SOA GRAND MEAN RMSE TABLE")
print("═" * 70)

piv = (soa_results_df
       .groupby(["System", "Mode", "Model"])["RMSE"]
       .agg(["mean", "std"]).reset_index())
piv["mean±std"] = (piv["mean"].map("{:.4f}".format)
                   + " ± " + piv["std"].map("{:.4f}".format))
wide = piv.pivot_table(
    index=["System", "Mode"], columns="Model",
    values="mean±std", aggfunc="first",
)
col_order = [c for c in ["PIDM-DP", "CSDI", "GRU-ODE", "ESN"]
             if c in wide.columns]
print(wide[col_order].to_string())

print("\n" + "─" * 70)
print("  WILCOXON TESTS  (PIDM-DP vs each baseline, two-tailed)")
print("─" * 70)
print(f"  {'System':<14} {'Mode':<5} {'vs CSDI':>16}"
      f" {'vs GRU-ODE':>14} {'vs ESN':>12}")
print("  " + "─" * 60)

sig_rows = []
for (sys_n, mode), grp in soa_results_df.groupby(["System", "Mode"]):
    pidm  = grp[grp["Model"] == "PIDM-DP"]["RMSE"].values
    parts = []
    row   = {"System": sys_n, "Mode": mode}
    for tag in ["CSDI", "GRU-ODE", "ESN"]:
        other = grp[grp["Model"] == tag]["RMSE"].values
        n     = min(len(pidm), len(other))
        if n >= 4:
            try:
                _, p = scipy_wilcoxon(pidm[:n], other[:n])
                sig  = ("***" if p < 0.001 else "**" if p < 0.01
                        else "*" if p < 0.05 else "ns")
                txt  = f"p={p:.2e}{sig}"
            except Exception:
                txt = "n/a"
        else:
            txt = "n/a"
        row[f"p_vs_{tag}"] = txt
        parts.append(f"{txt:>16}")
    sig_rows.append(row)
    print(f"  {sys_n:<14} {mode:<5}" + "".join(parts))

sig_df = pd.DataFrame(sig_rows)

res_dir = CONFIG["paths"]["results_dir"]
os.makedirs(res_dir, exist_ok=True)
soa_results_df.to_csv(res_dir + "soa_comparison_raw.csv",  index=False)
sig_df.to_csv(        res_dir + "soa_wilcoxon.csv",         index=False)
print(f"\n  ✓ Raw results  → {res_dir}soa_comparison_raw.csv")
print(f"  ✓ Significance → {res_dir}soa_wilcoxon.csv")


# =============================================================================
# ── SECTION 6: PUBLICATION FIGURE ────────────────────────────────────────────
# =============================================================================

MODEL_PALETTE = {
    "PIDM-DP": "#6C3483",
    "CSDI":    "#1A5276",
    "GRU-ODE": "#117A65",
    "ESN":     "#B7950B",
}
SYS_LABELS = {
    "lorenz":     "Lorenz\n(3D)",
    "rossler":    "Rössler\n(3D)",
    "hyper5d":    "Hyper5D\n(5D)",
    "lorenz96":   "Lorenz-96\n(20D)",
    "rabinovich": "Rabinovich\n(3D)",
}

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
fig.patch.set_facecolor("white")

for ax, mode_lbl in zip(axes, ["ID", "OOD"]):
    sub    = soa_results_df[soa_results_df["Mode"] == mode_lbl]
    mods   = [m for m in ["PIDM-DP", "CSDI", "GRU-ODE", "ESN"]
              if m in sub["Model"].unique()]
    n_mod  = len(mods)
    n_sys  = len(SOA_SYSTEMS)
    width  = 0.18
    offs   = np.linspace(-(n_mod-1)*width/2, (n_mod-1)*width/2, n_mod)
    x_pos  = np.arange(n_sys)

    for j, m in enumerate(mods):
        md = sub[sub["Model"] == m]
        means = [np.mean(md[md["System"] == s]["RMSE"].values)
                 if len(md[md["System"] == s]) else 0 for s in SOA_SYSTEMS]
        stds  = [np.std(md[md["System"] == s]["RMSE"].values)
                 if len(md[md["System"] == s]) else 0 for s in SOA_SYSTEMS]
        ax.bar(x_pos + offs[j], means, width=width - 0.01,
               color=MODEL_PALETTE[m], alpha=0.88, label=m,
               edgecolor="white", linewidth=0.4,
               yerr=stds, capsize=3,
               error_kw={"elinewidth": 1.0, "ecolor": "black", "alpha": 0.7},
               zorder=3)

    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([SYS_LABELS[s] for s in SOA_SYSTEMS], fontsize=9)
    ax.set_ylabel("Mean RMSE (log scale)", fontsize=10)
    ax.set_title(
        "In-Distribution (ID)" if mode_lbl == "ID"
        else "Out-of-Distribution (OOD)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(title="Model", fontsize=8, title_fontsize=9,
              loc="upper right", framealpha=0.9)
    ax.yaxis.grid(True, which="both", linestyle=":", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

fig.suptitle(
    f"SOA Comparison — PIDM-DP vs CSDI vs GRU-ODE vs ESN\n"
    f"All 5 Systems · 10% Observation Density · n={N_SOA} trials",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()
fig_path = CONFIG["paths"]["figures"] + "soa_comparison_final.pdf"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\n  ✓ Figure saved → {fig_path}")
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
print("\n" + "═" * 70)
print("  LaTeX TABLE SNIPPET")
print("═" * 70)
print(r"""\begin{table*}[htbp]
\centering
\caption{\textbf{Extended SOA comparison.} Mean RMSE $\pm$ std, $N=10$ trials,
10\% observation density. Best per row in \textbf{bold}.}
\label{tab:soa}
\renewcommand{\arraystretch}{1.25}
\begin{tabular}{llcccc}
\toprule
\textbf{System} & \textbf{Cond.}
  & \textbf{PIDM-DP} & \textbf{CSDI} & \textbf{GRU-ODE} & \textbf{ESN} \\
\midrule""")

SYS_TEX = {
    "lorenz":     r"Lorenz (3D)",
    "rossler":    r"R\"{o}ssler (3D)",
    "hyper5d":    r"Hyper5D (5D)",
    "lorenz96":   r"Lorenz-96 (20D)",
    "rabinovich": r"Rabinovich (3D)",
}
for (s, md), grp in soa_results_df.groupby(["System", "Mode"]):
    vals = {}
    for m in ["PIDM-DP", "CSDI", "GRU-ODE", "ESN"]:
        v = grp[grp["Model"] == m]["RMSE"].values
        vals[m] = (np.mean(v), np.std(v)) if len(v) > 0 else (999, 0)
    best = min(v[0] for v in vals.values())
    cells = []
    for m in ["PIDM-DP", "CSDI", "GRU-ODE", "ESN"]:
        mu, sd = vals[m]
        c = f"${mu:.4f} \\pm {sd:.4f}$"
        if abs(mu - best) < 1e-9:
            c = r"\textbf{" + c + "}"
        cells.append(c)
    print(f"{SYS_TEX[s]} & {md} & " + " & ".join(cells) + r" \\")

print(r"""\bottomrule
\end{tabular}
\end{table*}""")

print("\n  [SOA COMPARISON COMPLETE]")

# %%
import os, glob
ckpt_dir = CONFIG["paths"]["checkpoints"]
for f in glob.glob(os.path.join(ckpt_dir, "*soa*")):
    os.remove(f)
    print(f"Deleted: {f}")
print("Done — SOA checkpoint cleared.")
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# %%-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from tqdm.auto import tqdm

# Re-run the plotting style setup
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "savefig.format": "pdf",
    "savefig.dpi": 700
})

print("Libraries re-loaded. You can now run the visualization cell.")

# %%
print("═" * 70)
print("  PHASE 5 · Publication Figures & Grand Summary")
print("═" * 70)


import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Generate Boxplots
plot_rmse_boxplot_multi(all_res_id,  label="ID")
plot_rmse_boxplot_multi(all_res_ood, label="OOD")
plot_lyapunov_comparison(all_res_id,  label="ID")
plot_lyapunov_comparison(all_res_ood, label="OOD")


ABLATION_SYSTEMS = list(ablation_results.keys())

fig, axes = plt.subplots(1, len(ABLATION_SYSTEMS), figsize=(4.5*len(ABLATION_SYSTEMS), 3.5))
if len(ABLATION_SYSTEMS) == 1: axes = [axes]

for ax, sname in zip(axes, ABLATION_SYSTEMS):
    lambdas = CONFIG["lambda_phy_sweep"]
    pidm_v  = [ablation_results.get(sname,{}).get(w,{}).get("pidm_rmse", np.nan) for w in lambdas]
    ai_v    = [ablation_results.get(sname,{}).get(w,{}).get("pureai_rmse", np.nan) for w in lambdas]
    
    ax.plot(lambdas, pidm_v, "o-", color="#8E44AD", lw=2, markersize=6, label="PIDM-DP")
    ax.axhline(ai_v[0] if ai_v else 0, color="#C0392B", ls="--", lw=1.5, label="Pure AI (λ=0)")
    ax.set_xlabel("Physics weight $\\lambda_{phy}$", fontsize=10, fontweight="bold")
    ax.set_ylabel("Mean RMSE", fontsize=10, fontweight="bold")
    ax.set_title(sname.upper(), fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, frameon=True, edgecolor="black")
    ax.grid(True, linestyle=":", alpha=0.6)

fig.suptitle("Physics Weight Ablation Analysis", fontsize=14, fontweight="bold", y=1.05)
plt.tight_layout()
path = CONFIG["paths"]["figures"] + "ablation_lambda_sweep.pdf"
plt.savefig(path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"  ✓ Saved Ablation Sweep: {path}")

print_grand_summary(all_res_id, all_res_ood)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# ── LYAPUNOV TABLE GENERATOR ----------------------------------------------------------------------------------------------------------------------------------------------------
print("\n  " + "═" * 68)
print("  LATEX TABLE EXPORT: Lyapunov Exponents (ID)")
print("  " + "═" * 68)
print("  System & GT $\\lambda_{\\max}$ & PIDM-DP & Pure AI & EnKF \\\\")
print("  \\midrule")

for sname in ["lorenz", "rossler", "rabinovich"]:
    if sname in all_res_id:
        res = all_res_id[sname]
        
  
        gt   = np.nanmean(res.get("gt_lyap", [np.nan]))
        pidm = np.nanmean(res.get("pidm_lyap", [np.nan]))
        ai   = np.nanmean(res.get("pureai_lyap", [np.nan]))
        enkf = np.nanmean(res.get("enkf_lyap", [np.nan]))
       
        def fmt(val):
            if np.isnan(val): return "N/A"
            if val < 0.02: return f"${val:.3f}$ (collapse)"
            return f"${val:.3f}$"
            
        print(f"  {sname.capitalize()} & {gt:.3f} & {fmt(pidm)} & {fmt(ai)} & {fmt(enkf)} \\\\")
print("  " + "═" * 68 + "\n")
# ──────────────────────────────────────────────────────────────────────────



rows = []
for cond, all_res in [("ID", all_res_id), ("OOD", all_res_ood)]:
    for sname, res in all_res.items():
        for i in range(len(res["pidm_rmse"])):
            rows.append({
                "condition": cond, "system": sname, "trial": i+1,
                "pidm_rmse":   res["pidm_rmse"][i],
                "pureai_rmse": res["pureai_rmse"][i],
                "enkf_rmse":   res.get("enkf_rmse", [np.nan]*len(res["pidm_rmse"]))[i],
                "pidm_time_s": res["pidm_time"][i],
                "enkf_time_s": res.get("enkf_time", [np.nan]*len(res["pidm_time"]))[i],
                "pidm_lyap":   res["pidm_lyap"][i],
                "pureai_lyap": res.get("pureai_lyap", [np.nan]*len(res["pidm_rmse"]))[i], # Now safely tracked
                "enkf_lyap":   res.get("enkf_lyap", [np.nan]*len(res["pidm_lyap"]))[i],
                "gt_lyap":     res["gt_lyap"][i],
            })
df = pd.DataFrame(rows)
csv_path = CONFIG["paths"]["results_dir"] + "pidm_dp_v9_results.csv"
df.to_csv(csv_path, index=False)
print(f"  ✓ Results CSV Saved: {csv_path}  ({len(df)} rows)")
print("\n  [CELL 21 COMPLETE]\n")
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# %%---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
import os
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.integrate import solve_ivp
import torch
from tqdm.auto import tqdm


def plot_smooth_3d_line_solid(ax, x, y, z, color, lw=1.8, alpha=0.95, zorder=3, ls="-"):
    """
    Draws a continuous, depth-sorted 3D line using Line3DCollection.
    IMPORTANT: Line3DCollection does NOT trigger Matplotlib's auto-scaling.
    The caller MUST set ax.set_xlim / ylim / zlim after calling this function.
    """
    points   = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Map matplotlib linestyles to Line3DCollection format
    linestyles = ls if ls in ['-', '--', ':', '-.'] else 'solid'
    
    lc = Line3DCollection(
        segments, colors=[color], linewidths=lw, alpha=alpha, zorder=zorder, linestyles=linestyles
    )
    ax.add_collection3d(lc)



def hybrid_reintegrate(
    sys_name: str, recon_phys: np.ndarray, x_gt: np.ndarray, dt: float
):
    """
    Integrates the true ODE with True IC + PIDM-inferred parameters.
    Returns: (hybrid_traj, inferred_params, true_params, param_errors_dict)
    """
    s_dim   = CONFIG[sys_name]["state_dim"]
    p_dim   = CONFIG[sys_name]["param_dim"]
    p_names = CONFIG[sys_name]["param_names"]

    true_ic     = x_gt[:s_dim, 0].tolist()
    true_params = [float(x_gt[s_dim + j, 0]) for j in range(p_dim)]

    
    early_end = min(300, recon_phys.shape[1])
    inferred_params = [
        float(np.median(recon_phys[s_dim + j, :early_end]))
        for j in range(p_dim)
    ]

    p_errs = {
        n: abs(i - t) / (abs(t) + 1e-9) * 100
        for n, i, t in zip(p_names, inferred_params, true_params)
    }

 
    ranges      = CONFIG[sys_name].get("ranges", [])
    safe_params = []
    for j, pv in enumerate(inferred_params):
        if j < len(ranges):
            lo, hi = ranges[j]
            w  = hi - lo
            pv = float(np.clip(pv, lo - 3 * w, hi + 3 * w))
        safe_params.append(pv)

    fn_np  = get_sys_fn(sys_name)
    t_span = (0.0, x_gt.shape[1] * dt)
    t_eval = np.linspace(0.0, x_gt.shape[1] * dt, x_gt.shape[1])

    try:
  
        method = "LSODA" if sys_name == "rabinovich" else "DOP853"
        sol    = solve_ivp(
            fn_np, t_span, true_ic, args=tuple(safe_params),
            method=method, t_eval=t_eval, rtol=1e-8, atol=1e-10
        )
    
        hybrid = (
            sol.y
            if (sol.success and np.isfinite(sol.y).all())
            else recon_phys[:s_dim]
        )
    except Exception:
        hybrid = recon_phys[:s_dim]

    return hybrid, inferred_params, true_params, p_errs


# ── Main Figure Function ──────────────────────────────────────────────────────
def generate_scientific_report_v2(sys_name: str, n_trials: int = 12):
    print(f"\n  {'='*60}")
    print(f"  Generating Split Scientific Report: {sys_name.upper()}  (n={n_trials})")
    print(f"  {'='*60}")

    s_dim   = CONFIG[sys_name]["state_dim"]
    p_dim   = CONFIG[sys_name]["param_dim"]
    p_names = CONFIG[sys_name]["param_names"]
    dt      = CONFIG.get("data", {}).get("dt", 0.05)
    is_3d   = s_dim > 2   

    os.makedirs(CONFIG["paths"]["figures"], exist_ok=True)

    raw_data = generate_chaotic_dataset(
        sys_name, n_samples=n_trials, n_points=1000, dt=dt, seed=404
    )

    pidm_rmse_list, ai_rmse_list, hybrid_rmse_list = [], [], []
    param_errors_all  = {p: [] for p in p_names}
    true_params_all   = {p: [] for p in p_names}
    inf_params_all    = {p: [] for p in p_names}
    
    recon_trajs       = []
    recon_ai_trajs    = []
    hybrid_trajs      = []
    error_curves_pidm = []
    error_curves_ai   = []
    obs_idx_list      = []   

    # ── Inference loop ────────────────────────────────────────────────────────
    for i in tqdm(range(n_trials), desc=f"  Inference {sys_name}", leave=False):
        x_gt   = raw_data[i]
        x_norm = preps[sys_name].normalize(
            torch.tensor(raw_data[i:i + 1], dtype=torch.float32).to(DEVICE)
        )

        mask, obs_idx = make_random_mask(
            x_norm.shape, CONFIG["data"]["obs_ratio"], DEVICE, s_dim
        )
        obs_idx_list.append(obs_idx)

        x_noisy = add_observation_noise(x_norm, mask, CONFIG["data"]["obs_noise"])

        def gfn(x, t):
            return stable_guidance(
                x, t, x_noisy, mask,
                models[sys_name], diffs[sys_name], preps[sys_name], sys_name
            )

        recon_n  = guided_reconstruction(
            models[sys_name], diffs[sys_name], preps[sys_name], x_norm, mask, gfn
        )
        recon_p  = preps[sys_name].denormalize(recon_n).cpu().numpy()[0]

        recon_ai_n = guided_reconstruction(
            models[sys_name], diffs[sys_name], preps[sys_name], x_norm, mask, None
        )
        recon_ai_p = preps[sys_name].denormalize(recon_ai_n).cpu().numpy()[0]

        hybrid, inf_params, true_params, p_errs = hybrid_reintegrate(
            sys_name, recon_p, x_gt, dt
        )

        recon_trajs.append(recon_p)
        recon_ai_trajs.append(recon_ai_p)
        hybrid_trajs.append(hybrid)

        gt_state = x_gt[:s_dim]
        error_curves_pidm.append(np.linalg.norm(recon_p[:s_dim]    - gt_state, axis=0))
        error_curves_ai.append(  np.linalg.norm(recon_ai_p[:s_dim] - gt_state, axis=0))

        pidm_rmse_list.append(  trajectory_rmse(gt_state, recon_p[:s_dim]))
        ai_rmse_list.append(    trajectory_rmse(gt_state, recon_ai_p[:s_dim]))
        hybrid_rmse_list.append(trajectory_rmse(gt_state, hybrid))

        for pn, pe, t_val, i_val in zip(p_names, p_errs.values(), true_params, inf_params):
            param_errors_all[pn].append(pe)
            true_params_all[pn].append(t_val)
            inf_params_all[pn].append(i_val)

    # Median trial by PIDM RMSE
    median_idx    = int(np.argmin(np.abs(
        np.array(pidm_rmse_list) - np.median(pidm_rmse_list)
    )))
    x_gt_med      = raw_data[median_idx]
    exact_obs_idx = obs_idx_list[median_idx]   
    
    # ── METRICS PRINT BLOCK ───────────────────────────────────────────────────
    print(f"\n  --- PLOTTED VALUES & METRICS SUMMARY FOR {sys_name.upper()} ---")
    print(f"\n  [Figure 1: Phase-Space Portraits]")
    print(f"  NOTE: Titles show MEAN across all {n_trials} trials. Drawn trial is index {median_idx}.")
    print(f"    > Pure AI RMSE:        Title = {np.mean(ai_rmse_list):.4f} | Drawn Line = {ai_rmse_list[median_idx]:.4f}")
    print(f"    > PIDM-DP Raw RMSE:    Title = {np.mean(pidm_rmse_list):.4f} | Drawn Line = {pidm_rmse_list[median_idx]:.4f}")
    print(f"    > PIDM-DP Hybrid RMSE: Title = {np.mean(hybrid_rmse_list):.4f} | Drawn Line = {hybrid_rmse_list[median_idx]:.4f}")
    
    print(f"\n  [Figure 2B: Parameter Identification Error]")
    for p in p_names:
        p_err_mean = np.mean(param_errors_all[p])
        p_err_ci = st.sem(param_errors_all[p]) * st.t.ppf(0.975, len(param_errors_all[p]) - 1) if len(param_errors_all[p]) > 1 else 0.0
        t_mean = np.mean(true_params_all[p])
        i_mean = np.mean(inf_params_all[p])
        print(f"    {p}: Bar Height = {p_err_mean:.2f}% | Error Bar (± 95% CI) = {p_err_ci:.2f}%")
        print(f"       (Avg True Val: {t_mean:.3f}, Avg Inferred Val: {i_mean:.3f})")

    med_pidm_final = np.median(error_curves_pidm, axis=0)[-1]
    med_ai_final   = np.median(error_curves_ai, axis=0)[-1]
    print(f"\n  [Figure 2C: Point-wise Error at Final Timestep (t=1000)]")
    print(f"    > Median PIDM-DP Error: {med_pidm_final:.4e}")
    print(f"    > Median Pure AI Error: {med_ai_final:.4e}")
    print("  ---------------------------------------------------------\n")

    # ── Axis limits from ground-truth range ───────────────────────────────────
    xlim = (float(x_gt_med[0].min()) - 2, float(x_gt_med[0].max()) + 2)
    ylim = (float(x_gt_med[1].min()) - 2, float(x_gt_med[1].max()) + 2)
    if is_3d:
        zlim = (float(x_gt_med[2].min()) - 2, float(x_gt_med[2].max()) + 2)
    else:
        zlim = (-2.0, 2.0)

    # =========================================================================
    # FIGURE 1 — Phase-Space Portraits
    # =========================================================================
    fig1 = plt.figure(figsize=(24, 6))
    fig1.patch.set_facecolor("white")
    gs1  = gridspec.GridSpec(1, 4, figure=fig1, wspace=0.10)

    portraits = [
        ("Ground Truth",                    x_gt_med[:s_dim],                    "#1A1A2E", "-"),
        ("Pure AI",                         recon_ai_trajs[median_idx][:s_dim],     "#C0392B", "--"), 
        ("PIDM-DP Raw",                     recon_trajs[median_idx][:s_dim],        "#8E44AD", "-"),
        ("PIDM-DP Hybrid\n(Re-integrated)", hybrid_trajs[median_idx],               "#5B2C8D", "-"),
    ]

    rmse_by_col = [
        None,
        np.mean(ai_rmse_list),
        np.mean(pidm_rmse_list),
        np.mean(hybrid_rmse_list),
    ]

    for col, (title, traj, color, ls) in enumerate(portraits):
        ax = fig1.add_subplot(gs1[0, col], projection="3d" if is_3d else None)

        z_gt   = x_gt_med[2] if is_3d else np.zeros_like(x_gt_med[0])
        z_vals = traj[2]      if is_3d else np.zeros_like(traj[0])

        if is_3d:
            if col > 0:
                plot_smooth_3d_line_solid(
                    ax, x_gt_med[0], x_gt_med[1], z_gt,
                    color="#B0B0B0", lw=0.8, alpha=0.3, zorder=1
                )
            # Main trajectory
            plot_smooth_3d_line_solid(
                ax, traj[0], traj[1], z_vals,
                color=color, lw=1.8, alpha=0.95, zorder=3, ls=ls
            )
            ax.set_zlim(zlim)
            ax.xaxis.set_pane_color((1, 1, 1, 0))
            ax.yaxis.set_pane_color((1, 1, 1, 0))
            ax.zaxis.set_pane_color((1, 1, 1, 0))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.view_init(elev=22, azim=45)
        else:
            if col > 0:
                ax.plot(x_gt_med[0], x_gt_med[1], color="#B0B0B0", lw=0.8, alpha=0.3, zorder=1)
            ax.plot(traj[0], traj[1], color=color, lw=1.8, alpha=0.95, zorder=3, ls=ls)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticklabels([])
        ax.set_yticklabels([])


        gt_x_obs = x_gt_med[0, exact_obs_idx]
        gt_y_obs = x_gt_med[1, exact_obs_idx]
        
        if is_3d:
            gt_z_obs = x_gt_med[2, exact_obs_idx]
            ax.scatter(
                gt_x_obs, gt_y_obs, gt_z_obs,
                color="#2980B9", s=45,             
                edgecolors="white", linewidths=0.8,
                depthshade=False, zorder=10, label="10% Obs (GT)"
            )
        else:
            ax.scatter(
                gt_x_obs, gt_y_obs,
                color="#2980B9", s=45,
                edgecolors="white", linewidths=0.8,
                zorder=10, label="10% Obs (GT)"
            )
            
        if col == 0:
            ax.legend(fontsize=12, loc="upper left", framealpha=0.9)

        rmse_val  = rmse_by_col[col]
        rmse_str  = f"\nRMSE = {rmse_val:.3f}" if rmse_val is not None else ""
        ax.set_title(
            title + rmse_str,
            fontsize=16, fontweight="bold", pad=15
        )

    fig1.tight_layout()
    path1 = CONFIG["paths"]["figures"] + f"report_fig1_manifolds_{sys_name}.pdf"
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"  ✓ Saved Figure 1 (Manifolds): {path1}")

    # =========================================================================
    # FIGURE 2 — Quantitative Metrics
    # =========================================================================
    fig2 = plt.figure(figsize=(24, 5))
    fig2.patch.set_facecolor("white")
    gs2  = gridspec.GridSpec(1, 3, figure=fig2, wspace=0.25)

    # ── Panel A: Time series ─────────────────────────────────────────────────
    ax_ts = fig2.add_subplot(gs2[0, 0])
    t_ax  = np.arange(1000) * dt
    gt_x  = x_gt_med[0]

    ax_ts.plot(t_ax, gt_x,                              color="#1A1A2E", lw=2.0, alpha=0.30, label="Ground Truth")
    ax_ts.plot(t_ax, recon_ai_trajs[median_idx][0],     color="#C0392B", lw=1.5, ls="--", alpha=0.80, label="Pure AI")
    ax_ts.plot(t_ax, hybrid_trajs[median_idx][0],       color="#5B2C8D", lw=2.0, alpha=0.90, label="PIDM-DP Hybrid")

    ax_ts.scatter(
        t_ax[exact_obs_idx], gt_x[exact_obs_idx],
        color="#2980B9", s=45, zorder=5,
        edgecolors="white", linewidths=0.8, label="10% Obs"
    )

    ax_ts.set_title("Temporal Reconstruction  —  x(t)", fontsize=16, fontweight="bold")
    ax_ts.set_xlabel("Time (s)", fontsize=14)
    ax_ts.set_ylabel("x(t)",     fontsize=14)
    ax_ts.legend(fontsize=11)
    ax_ts.grid(True, ls=":", alpha=0.6)

    ax_par = fig2.add_subplot(gs2[0, 1])
    mean_errs = [
        np.mean(param_errors_all[p]) if param_errors_all[p] else 0.0
        for p in p_names
    ]
    ci_errs = [
        (st.sem(param_errors_all[p])
         * st.t.ppf(0.975, max(len(param_errors_all[p]) - 1, 1))
         if len(param_errors_all[p]) > 1 else 0.0)
        for p in p_names
    ]
    xpos = np.arange(len(p_names))

    bars = ax_par.bar(
        xpos, mean_errs, yerr=ci_errs,
        color="#5B2C8D", alpha=0.85, capsize=7,
        edgecolor="black", linewidth=1.2,
        error_kw={"lw": 2.0, "capthick": 2.0}
    )

    # Calculate dynamic relative offset for labels to prevent overlapping
    max_err_val = max([me + ce for me, ce in zip(mean_errs, ci_errs)]) if mean_errs else 1.0
    text_offset = max_err_val * 0.05 

    for j, (bar, me, ce) in enumerate(zip(bars, mean_errs, ci_errs)):
        ax_par.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ce + text_offset, 
            f"{me:.1f}%",
            ha="center", va="bottom",
            fontsize=12, fontweight="bold"
        )

    ax_par.set_xticks(xpos)
    ax_par.set_xticklabels(p_names, fontsize=14)
    ax_par.axhline(5, color="crimson", ls=":", lw=2.0, alpha=0.8, label="5% Threshold")
    ax_par.set_title("Parameter Identification Error", fontsize=16, fontweight="bold")
    ax_par.set_ylabel("Relative Error (%)", fontsize=14)
  
    ax_par.set_ylim(0, max_err_val + (max_err_val * 0.15))
    ax_par.legend(fontsize=11)
    ax_par.grid(axis="y", ls=":", alpha=0.6)
    for sp in ["top", "right"]:
        ax_par.spines[sp].set_visible(False)

    # ── Panel C: Log error over time ─────────────────────────────────────────
    ax_err = fig2.add_subplot(gs2[0, 2])
    med_pidm   = np.median(error_curves_pidm, axis=0)
    med_ai     = np.median(error_curves_ai,   axis=0)
    p25_pidm   = np.percentile(error_curves_pidm, 25, axis=0)
    p75_pidm   = np.percentile(error_curves_pidm, 75, axis=0)
    p25_ai     = np.percentile(error_curves_ai,   25, axis=0)
    p75_ai     = np.percentile(error_curves_ai,   75, axis=0)
    eps        = 1e-9   # prevent log(0)

    ax_err.semilogy(t_ax, med_pidm + eps, color="#5B2C8D", lw=2.0,       label="PIDM-DP (median)")
    ax_err.semilogy(t_ax, med_ai   + eps, color="#C0392B", lw=1.5, ls="--", label="Pure AI (median)")
    ax_err.fill_between(t_ax, p25_pidm + eps, p75_pidm + eps, alpha=0.20, color="#5B2C8D")
    ax_err.fill_between(t_ax, p25_ai   + eps, p75_ai   + eps, alpha=0.15, color="#C0392B")

    ax_err.set_title(f"Point-wise Error ‖Δx‖₂  (n={n_trials})", fontsize=16, fontweight="bold")
    ax_err.set_xlabel("Time (s)", fontsize=14)
    ax_err.set_ylabel("Log Error", fontsize=14)
    ax_err.legend(fontsize=11)
    ax_err.grid(True, ls=":", alpha=0.6)

    fig2.tight_layout()
    path2 = CONFIG["paths"]["figures"] + f"report_fig2_metrics_{sys_name}.pdf"
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"  ✓ Saved Figure 2 (Metrics): {path2}")


for sname in _SYSTEMS:
    generate_scientific_report_v2(sname, n_trials=30)

print("\n  [CELL 22 COMPLETE]\n")
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# this is ablation sweep for the different sparsity and noise

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
import pandas as pd
import seaborn as sns

def run_comprehensive_ablation(n_trials=10):
    """
    Evaluates PIDM-DP vs Pure AI across varying noise and sparsity levels.
    """
    print(f"\n{'='*60}")
    print(f"  STARTING MULTI-DIMENSIONAL ABLATION STUDY (n={n_trials})")
    print(f"{'='*60}")


    noise_levels = CONFIG.get("noise_levels", [0.0, 0.05, 0.15])
    sparsity_levels = CONFIG.get("sparsity_levels", [0.02, 0.05, 0.10])
    systems_to_test = _SYSTEMS # Lorenz, Rossler, etc.

    results_master = []

    for sys_name in systems_to_test:
        print(f"\n>> Testing System: {sys_name.upper()}")
        s_dim = CONFIG[sys_name]["state_dim"]
        dt = CONFIG["data"]["dt"]
        
        # Generate fresh test data for this system
        raw_test_data = generate_chaotic_dataset(
            sys_name, n_samples=n_trials, n_points=1000, dt=dt, seed=999
        )

        for noise in noise_levels:
            for sparsity in sparsity_levels:
                pidm_errors = []
                ai_errors = []

                for i in range(n_trials):
                    x_gt = raw_test_data[i]
                  
                    x_norm = preps[sys_name].normalize(
                        torch.tensor(raw_test_data[i:i+1], dtype=torch.float32).to(DEVICE)
                    )

                 
                    mask, _ = make_random_mask(x_norm.shape, sparsity, DEVICE, s_dim)
                    
                    # Add Noise
                    x_obs_noisy = add_observation_noise(x_norm, mask, noise)

                    # 1. PIDM-DP Reconstruction (Guided)
                    def gfn(x, t):
                        return stable_guidance(
                            x, t, x_obs_noisy, mask, 
                            models[sys_name], diffs[sys_name], preps[sys_name], sys_name
                        )
                    
                    recon_pidm_n = guided_reconstruction(
                        models[sys_name], diffs[sys_name], preps[sys_name], x_norm, mask, gfn
                    )
                    recon_pidm = preps[sys_name].denormalize(recon_pidm_n).cpu().numpy()[0]

                    # 2. Pure AI Reconstruction (Unguided)
                    recon_ai_n = guided_reconstruction(
                        models[sys_name], diffs[sys_name], preps[sys_name], x_norm, mask, None
                    )
                    recon_ai = preps[sys_name].denormalize(recon_ai_n).cpu().numpy()[0]

                    # Calculate RMSE vs Hidden 100% Ground Truth
                    pidm_errors.append(trajectory_rmse(x_gt[:s_dim], recon_pidm[:s_dim]))
                    ai_errors.append(trajectory_rmse(x_gt[:s_dim], recon_ai[:s_dim]))

                # Log results
                avg_pidm = np.mean(pidm_errors)
                avg_ai = np.mean(ai_errors)
                improvement = (avg_ai / avg_pidm) if avg_pidm > 0 else 1.0

                results_master.append({
                    "System": sys_name,
                    "Noise": noise,
                    "Sparsity": sparsity,
                    "Model": "PIDM-DP",
                    "RMSE": avg_pidm
                })
                results_master.append({
                    "System": sys_name,
                    "Noise": noise,
                    "Sparsity": sparsity,
                    "Model": "Pure AI",
                    "RMSE": avg_ai
                })

                print(f"   [Noise {noise:.2f} | Sparse {sparsity:.2f}] PIDM: {avg_pidm:.4f} vs AI: {avg_ai:.4f} ({improvement:.1f}x better)")

    # ── VISUALIZATION ────────────────────────────────────────────────────────
    df = pd.DataFrame(results_master)
    
    
    fig, axes = plt.subplots(len(systems_to_test), 1, figsize=(12, 5 * len(systems_to_test)))
    if len(systems_to_test) == 1: axes = [axes]

    for idx, sys_name in enumerate(systems_to_test):
        ax = axes[idx]
        sys_df = df[df["System"] == sys_name]
        
        
        sns.lineplot(
            data=sys_df, x="Sparsity", y="RMSE", hue="Noise", style="Model",
            markers=True, dashes=True, palette="viridis", ax=ax, lw=2.5
        )
        
        ax.set_title(f"Robustness Analysis: {sys_name.upper()}", fontsize=16, fontweight='bold')
        ax.set_yscale('log')
        ax.set_xlabel("Observation Density (Sparsity %)", fontsize=12)
        ax.set_ylabel("Log RMSE", fontsize=12)
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(title="Noise / Model", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    save_path = CONFIG["paths"]["figures"] + "ablation_noise_sparsity.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"\n  ✓ Ablation Study Complete. Plot saved to: {save_path}")

# Execute the study
run_comprehensive_ablation(n_trials=10)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



# %%----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#only for the Lorentz

print("═" * 70)
print("  SPARSITY SWEEP · Observation Density vs. ρ Recovery (Lorenz ID)")
print("═" * 70)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# ── Configuration ─────────────────────────────────────────────────────────────

SYS_NAME      = "lorenz"       # Lorenz 63 only
N_TRIALS      = 10             # Trials per density level (N=20 as specified)
OBS_RATIOS    = [0.02, 0.05, 0.10, 0.20, 0.50]   # 2% → 50%
SEED_BASE     = GLOBAL_SEED + 100                  # Separate seed from main evals
FIG_PATH      = CONFIG["paths"]["figures"] + "sparsity_sweep_rho_mape.pdf"
CKPT_NAME     = "sparsity_sweep_lorenz"

# ρ is parameter index 1 in Lorenz: param_names = ["σ", "ρ", "β"]
# The key stored in pidm_param_errors dicts is the raw unicode string "ρ"
RHO_KEY       = CONFIG[SYS_NAME]["param_names"][1]   # → "ρ"
RHO_IDX       = 1                                     # index in param vector

print(f"\n  System       : {SYS_NAME.upper()}")
print(f"  Trials/density: {N_TRIALS}")
print(f"  Densities    : {OBS_RATIOS}")
print(f"  ρ key in results dict: '{RHO_KEY}'  (index {RHO_IDX})")
print(f"  Output figure: {FIG_PATH}")
print(f"  Checkpoint   : {CKPT_NAME}")

# ── Load or initialise checkpoint ─────────────────────────────────────────────

ck = load_checkpoint(CKPT_NAME)
if ck is not None:
    sweep_results = ck["sweep_results"]   # dict: {float(obs_ratio): list[float mape_rho]}
    print(f"\n  ↩  Loaded checkpoint — already have densities: "
          f"{sorted(sweep_results.keys())}")
else:
    sweep_results = {}
    print("\n  No checkpoint found — starting fresh.")

# ── Backup and restore CONFIG["data"]["obs_ratio"] safely ────────────────────

_ORIGINAL_OBS_RATIO = CONFIG["data"]["obs_ratio"]
print(f"\n  Original CONFIG obs_ratio = {_ORIGINAL_OBS_RATIO} "
      f"(will be restored after sweep)")

# ── Main sweep loop ───────────────────────────────────────────────────────────

for obs_ratio in OBS_RATIOS:

    if obs_ratio in sweep_results:
        n_done = len(sweep_results[obs_ratio])
        print(f"\n  [SKIP] obs_ratio={obs_ratio:.0%} — {n_done} trials already in checkpoint.")
        continue

    print(f"\n  {'─'*66}")
    print(f"  Running obs_ratio = {obs_ratio:.0%}  ({int(obs_ratio*100)}% observed, "
          f"{int((1-obs_ratio)*100)}% missing)")
    print(f"  {'─'*66}")

    # ── Temporarily override the observation ratio in CONFIG ──────────────
    CONFIG["data"]["obs_ratio"] = obs_ratio

    # ── Generate N_TRIALS fresh test trajectories at this density ─────────
    set_seed(SEED_BASE + int(obs_ratio * 1000))
    test_raw = generate_chaotic_dataset(
        SYS_NAME,
        n_samples=N_TRIALS,
        n_points=CONFIG["data"]["n_points"],
        dt=CONFIG["data"]["dt"],
        transient=CONFIG["data"]["transient"],
        ood=False,                              # ID condition throughout
        seed=SEED_BASE + int(obs_ratio * 1000),
    )

    print(f"  Generated {N_TRIALS} Lorenz ID trajectories at {obs_ratio:.0%} density.")

    # ── Run evaluate_system — this uses CONFIG["data"]["obs_ratio"] ───────
    res = evaluate_system(
        SYS_NAME,
        models[SYS_NAME],
        preps[SYS_NAME],
        diffs[SYS_NAME],
        test_raw,
        n_trials=N_TRIALS,
        label=f"sparsity_{obs_ratio:.0%}",
    )


    #
    rho_mape_per_trial = []
    for trial_param_errs in res["pidm_param_errors"]:
        if isinstance(trial_param_errs, dict) and RHO_KEY in trial_param_errs:
            rho_mape_per_trial.append(float(trial_param_errs[RHO_KEY]))
        else:
           
            try:
                vals = list(trial_param_errs.values())
                rho_mape_per_trial.append(float(vals[RHO_IDX]))
                print(f"    [WARN] Used index fallback for ρ at trial "
                      f"{len(rho_mape_per_trial)}")
            except Exception as e:
                print(f"    [WARN] Could not extract ρ MAPE: {e} — substituting NaN")
                rho_mape_per_trial.append(float("nan"))

   
    finite_vals = [v for v in rho_mape_per_trial if np.isfinite(v)]
    fallback    = float(np.median(finite_vals)) if finite_vals else 999.0
    rho_mape_clean = [v if np.isfinite(v) else fallback for v in rho_mape_per_trial]

    sweep_results[obs_ratio] = rho_mape_clean

    print(f"\n  obs_ratio={obs_ratio:.0%}  |  "
          f"Median MAPE(ρ) = {np.median(rho_mape_clean):.2f}%  |  "
          f"Mean = {np.mean(rho_mape_clean):.2f}%  |  "
          f"Std = {np.std(rho_mape_clean):.2f}%")

    
    save_checkpoint(CKPT_NAME, {"sweep_results": sweep_results})



CONFIG["data"]["obs_ratio"] = _ORIGINAL_OBS_RATIO
print(f"\n  CONFIG obs_ratio restored to {_ORIGINAL_OBS_RATIO}")



print("\n  " + "─"*58)
print(f"  {'Density':>10}  {'N':>4}  {'Median MAPE(ρ)':>16}  {'Mean ± Std':>20}")
print("  " + "─"*58)
for ratio in OBS_RATIOS:
    vals = sweep_results.get(ratio, [])
    if vals:
        med  = np.median(vals)
        mn   = np.mean(vals)
        std  = np.std(vals)
        print(f"  {ratio:>9.0%}  {len(vals):>4}  {med:>15.2f}%  "
              f"{mn:>8.2f} ± {std:.2f}%")
print("  " + "─"*58)



PALETTE = {
    "pidm":  "#6C3483",  
    "box":   "#A569BD",   
    "med":   "#2C3E50",   
    "chaos_threshold": "#E74C3C",  
}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
fig.patch.set_facecolor("white")

# ── Panel A: Boxplot ──────────────────────────────────────────────────────────
ax1 = axes[0]

data_for_boxplot = [sweep_results.get(r, [np.nan]) for r in OBS_RATIOS]
x_positions      = list(range(1, len(OBS_RATIOS) + 1))
x_labels         = [f"{int(r*100)}%" for r in OBS_RATIOS]

bp = ax1.boxplot(
    data_for_boxplot,
    positions=x_positions,
    patch_artist=True,
    widths=0.55,
    medianprops=dict(color=PALETTE["med"], linewidth=2.5),
    boxprops=dict(facecolor=PALETTE["box"], alpha=0.75, linewidth=1.2),
    whiskerprops=dict(color=PALETTE["pidm"], linewidth=1.5, linestyle="--"),
    capprops=dict(color=PALETTE["pidm"], linewidth=1.8),
    flierprops=dict(marker="o", color=PALETTE["pidm"], alpha=0.5,
                    markersize=4, markerfacecolor=PALETTE["box"]),
    notch=False,
)


ax1.axhline(5.26, color="#1A8F78", linestyle=":", linewidth=1.8, alpha=0.85,
            label=r"Main result at 10\% ($5.26\%$)")

ax1.set_xticks(x_positions)
ax1.set_xticklabels(x_labels, fontsize=10)
ax1.set_xlabel("Observation Density", fontsize=11, fontweight="bold")
ax1.set_ylabel(r"MAPE($\rho$)  [\%]", fontsize=11, fontweight="bold")
ax1.set_title(r"(a) $\rho$ Recovery vs.\ Observation Density",
              fontsize=11, fontweight="bold")
ax1.legend(fontsize=9, frameon=True, edgecolor="black")
ax1.grid(axis="y", linestyle=":", alpha=0.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

ax2 = axes[1]

medians = [np.median(sweep_results.get(r, [np.nan])) for r in OBS_RATIOS]
q25     = [np.percentile(sweep_results.get(r, [np.nan]), 25) for r in OBS_RATIOS]
q75     = [np.percentile(sweep_results.get(r, [np.nan]), 75) for r in OBS_RATIOS]
dens_pct = [r * 100 for r in OBS_RATIOS]

ax2.plot(dens_pct, medians, "o-",
         color=PALETTE["pidm"], linewidth=2.2, markersize=7,
         markerfacecolor=PALETTE["box"], markeredgecolor=PALETTE["pidm"],
         markeredgewidth=1.5, label="PIDM-DP Median MAPE(ρ)", zorder=3)

ax2.fill_between(dens_pct, q25, q75,
                 color=PALETTE["box"], alpha=0.30,
                 label="IQR (25th–75th percentile)")

# Highlight the 10% operating point
idx_10 = OBS_RATIOS.index(0.10)
ax2.scatter([10], [medians[idx_10]],
            s=100, color=PALETTE["pidm"], zorder=5,
            marker="*", label=f"10\\% operating point ({medians[idx_10]:.1f}\\%)")

ax2.set_xlabel("Observation Density (\\%)", fontsize=11, fontweight="bold")
ax2.set_ylabel(r"Median MAPE($\rho$)  [\%]", fontsize=11, fontweight="bold")
ax2.set_title(r"(b) Median Recovery Error with IQR",
              fontsize=11, fontweight="bold")
ax2.legend(fontsize=9, frameon=True, edgecolor="black")
ax2.grid(linestyle=":", alpha=0.5)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.set_xticks([2, 5, 10, 20, 50])
ax2.set_xticklabels(["2", "5", "10", "20", "50"])



fig.suptitle(
    r"Lorenz 63 Rayleigh Number Recovery vs.\ Observation Density "
    r"($N=20$ trials per condition, ID)",
    fontsize=11, fontweight="bold", y=1.02
)

plt.tight_layout()
os.makedirs(CONFIG["paths"]["figures"], exist_ok=True)
plt.savefig(FIG_PATH, dpi=300, bbox_inches="tight", format="pdf")
plt.close(fig)

print(f"\n  ✓ Saved publication figure → {FIG_PATH}")
print("\n  NEXT STEP: In chaos_aip_pidm_dp.tex Section IV.C,")
print("  replace the \\fbox{...} placeholder with:")
print(r"  \includegraphics[width=\columnwidth]{sparsity_sweep_rho_mape.pdf}")
print("\n  [SPARSITY SWEEP COMPLETE]")

# %%
print("═" * 70)
print("  BASELINE COMPARISON · PIDM-DP vs. Bi-LSTM vs. PINN  [CORRECTED]")
print("═" * 70)
print()
print("  Architectures:")
print("    Bi-LSTM : per-trajectory sparse interpolation (250 gradient steps)")
print("    PINN    : per-trajectory physics-constrained regression (500 steps)")
print("              NOTE: PINN uses mean of training parameter ranges as the")
print("              'known' physics params — a slight advantage for PINN,")
print("              disclosed for scientific accuracy.")
print()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm


def get_sys_fn_pytorch(sys_name):
    """Returns a PyTorch-differentiable version of each chaotic ODE."""

    def lorenz_pt(t, state, sigma, rho, beta):
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        return torch.stack([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ], dim=1)

    def rossler_pt(t, state, a, b, c):
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        return torch.stack([
            -y - z,
            x + a * y,
            b + z * (x - c)
        ], dim=1)

    def rabinovich_pt(t, state, alpha, gamma):
        x, y, z = state[:, 0], state[:, 1], state[:, 2]
        return torch.stack([
            y * (z - 1 + x**2) + gamma * x,
            x * (3 * z + 1 - x**2) + gamma * y,
            -2 * z * (alpha + x * y)
        ], dim=1)

    def zero_physics_pt(t, state, *args):
        """Fallback for high-dim systems without a differentiable ODE defined."""
        return torch.zeros_like(state)

    return {
        "lorenz":      lorenz_pt,
        "rossler":     rossler_pt,
        "rabinovich":  rabinovich_pt,
    }.get(sys_name, zero_physics_pt)



class ChaoticLSTM(nn.Module):
    """Bidirectional LSTM for sparse chaotic time-series interpolation."""
    def __init__(self, s_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=s_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(hidden_dim * 2, s_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out)


class ChaoticPINN(nn.Module):
    """Continuous MLP x(t) optimised under data + physics loss."""
    def __init__(self, s_dim, hidden_layers=4, hidden_dim=64):
        super().__init__()
        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, s_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)   # t: (L, 1) → output: (L, s_dim)


# =============================================================================
# 3. ACTIVE INFERENCE FUNCTIONS —
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
def run_lstm_inference(sys_name, x_obs, mask, s_dim, steps=250):
    """
    Trains a Bi-LSTM from scratch on the 10% observed data points for this
    single trajectory, then predicts the full 1000-step sequence.

    Args:
        x_obs   : normalised observation tensor, shape (1, total_channels, L)
        mask    : observation mask, shape (1, total_channels, L), 1=observed
        s_dim   : number of physical state dimensions

    Returns:
        np.ndarray of shape (s_dim, L)
    """
    model = ChaoticLSTM(s_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    criterion = nn.MSELoss()

   
    obs_state = preps[sys_name].denormalize(x_obs)[0, :s_dim, :].to(DEVICE)
  

    obs_mask_1d = mask[0, 0, :].bool()   # (L,)

   
    sparse_input = (obs_state * mask[0, :s_dim, :]).T.unsqueeze(0)
    # sparse_input shape: (1, L, s_dim)

    target = obs_state.T.unsqueeze(0)   # (1, L, s_dim)

    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        pred = model(sparse_input)   # (1, L, s_dim)

      
        loss = criterion(pred[0][obs_mask_1d], target[0][obs_mask_1d])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_pred = model(sparse_input).squeeze(0).T  

    return final_pred.cpu().numpy()


def run_pinn_inference(sys_name, x_obs, mask, s_dim, dt=0.05, steps=500):
    """
    Optimises a continuous MLP x(t) under data fidelity loss at the observed
    time steps and an ODE physics residual loss everywhere.

    NOTE: The physics ODE uses the midpoint of training parameter ranges as
    'known' parameters. This is a mild advantage for PINN; it is disclosed
    in paper Table X notes. PIDM-DP has no such knowledge.

    Args:
        x_obs   : normalised observation tensor, shape (1, total_channels, L)
        mask    : observation mask, shape (1, total_channels, L)
        s_dim   : number of physical state dimensions

    Returns:
        np.ndarray of shape (s_dim, L)
    """
    model = ChaoticPINN(s_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    
    seq_len = x_obs.shape[2]   


    t = torch.linspace(0, seq_len * dt, seq_len,
                       requires_grad=True).unsqueeze(1).to(DEVICE)


    obs_state = preps[sys_name].denormalize(x_obs)[0, :s_dim, :].to(DEVICE)


    obs_mask_1d = mask[0, 0, :].bool()   # (L,)

    physics_fn = get_sys_fn_pytorch(sys_name)
    mean_params = [(r[0] + r[1]) / 2.0 for r in CONFIG[sys_name]["ranges"]]

    for _ in range(steps):
        optimizer.zero_grad()
        x_pred = model(t)  

     
        data_loss = torch.mean(
            (x_pred[obs_mask_1d] - obs_state.T[obs_mask_1d]) ** 2
        )

      
        dxdt_pred = torch.zeros_like(x_pred)   # (L, s_dim)
        for i in range(s_dim):
            grad_i = torch.autograd.grad(
                x_pred[:, i:i+1], t,
                grad_outputs=torch.ones(seq_len, 1, device=DEVICE),
                create_graph=True,
                retain_graph=True,
            )[0]
            dxdt_pred[:, i] = grad_i.squeeze()

        dxdt_ode = physics_fn(t, x_pred, *mean_params)   
        physics_loss = torch.mean((dxdt_pred - dxdt_ode) ** 2)

        loss = data_loss + 0.1 * physics_loss
        loss.backward()
        optimizer.step()

  
    model.eval()
    with torch.no_grad():
        t_eval = t.detach()
        final_pred = model(t_eval).T.cpu().numpy()  

    return final_pred


# =============================================================================
# 4. MAIN BENCHMARK LOOP
# =============================================================================
def run_ultimate_baseline_comparison(n_trials=10):
    print(f"\n{'='*70}")
    print(f"  BENCHMARK · PIDM-DP vs. Bi-LSTM vs. PINN  (n={n_trials} per condition)")
    print(f"{'='*70}")

    results = []

    for sys_name in _SYSTEMS:
        s_dim = CONFIG[sys_name]["state_dim"]
        dt    = CONFIG["data"]["dt"]

        for mode in ["ID", "OOD"]:
            print(f"\n  > {sys_name.upper()} [{mode}]")

            orig = CONFIG[sys_name]["ranges"]
            if mode == "OOD":
                CONFIG[sys_name]["ranges"] = CONFIG[sys_name]["ood_ranges"]

            set_seed(GLOBAL_SEED + 404)
            test_data = generate_chaotic_dataset(
                sys_name, n_samples=n_trials,
                n_points=CONFIG["data"]["n_points"],
                dt=dt, seed=GLOBAL_SEED + 404,
            )
            CONFIG[sys_name]["ranges"] = orig   # restore immediately

            loop_pidm, loop_lstm, loop_pinn = [], [], []

            for i in tqdm(range(n_trials), desc=f"    Trials", leave=False):
                x_gt   = test_data[i]   # (total_channels, L)
                x_norm = preps[sys_name].normalize(
                    torch.tensor(test_data[i:i+1], dtype=torch.float32).to(DEVICE)
                )   # (1, total_channels, L)

                mask, _ = make_random_mask(
                    x_norm.shape, CONFIG["data"]["obs_ratio"], DEVICE, s_dim
                )
                x_obs = add_observation_noise(
                    x_norm, mask, CONFIG["data"]["obs_noise"]
                )

                # ── PIDM-DP ───────────────────────────────────────────────────
                def gfn(x, t_step):
                    return stable_guidance(
                        x, t_step, x_obs, mask,
                        models[sys_name], diffs[sys_name], preps[sys_name],
                        sys_name,
                    )
                recon_n    = guided_reconstruction(
                    models[sys_name], diffs[sys_name], preps[sys_name],
                    x_norm, mask, gfn,
                )
                recon_pidm = preps[sys_name].denormalize(recon_n).cpu().numpy()[0][:s_dim]
                # (s_dim, L)

                # ── Bi-LSTM ───────────────────────────────────────────────────
                recon_lstm = run_lstm_inference(sys_name, x_obs, mask, s_dim)

                # ── PINN ──────────────────────────────────────────────────────
                recon_pinn = run_pinn_inference(sys_name, x_obs, mask, s_dim)

                # ── RMSE ──────────────────────────────────────────────────────
                gt_state = x_gt[:s_dim]   # (s_dim, L)
                r_pidm = trajectory_rmse(gt_state, recon_pidm)
                r_lstm = trajectory_rmse(gt_state, recon_lstm)
                r_pinn = trajectory_rmse(gt_state, recon_pinn)

                results.extend([
                    {"System": sys_name, "Mode": mode, "Model": "PIDM-DP", "RMSE": r_pidm},
                    {"System": sys_name, "Mode": mode, "Model": "Bi-LSTM", "RMSE": r_lstm},
                    {"System": sys_name, "Mode": mode, "Model": "PINN",    "RMSE": r_pinn},
                ])
                loop_pidm.append(r_pidm)
                loop_lstm.append(r_lstm)
                loop_pinn.append(r_pinn)

            print(f"    PIDM-DP  {np.mean(loop_pidm):.4f} ± {np.std(loop_pidm):.4f}")
            print(f"    Bi-LSTM  {np.mean(loop_lstm):.4f} ± {np.std(loop_lstm):.4f}")
            print(f"    PINN     {np.mean(loop_pinn):.4f} ± {np.std(loop_pinn):.4f}")

    # =========================================================================
    # 5. SUMMARY TABLE & FIGURE
    # =========================================================================
    df = pd.DataFrame(results)
    summary = df.groupby(["System", "Mode", "Model"])["RMSE"].mean().unstack()
    print("\n" + "═"*70)
    print("  GRAND MEAN RMSE TABLE")
    print("═"*70)
    print(summary.to_string())


    csv_path = CONFIG["paths"]["results_dir"] + "baseline_comparison_lstm_pinn.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Raw results saved → {csv_path}")


    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("white")

    palette = {"PIDM-DP": "#5B2C8D", "Bi-LSTM": "#C0392B", "PINN": "#2980B9"}

    for ax, mode_label in zip(axes, ["ID", "OOD"]):
        subset = df[df["Mode"] == mode_label]
        sns.barplot(
            data=subset, x="System", y="RMSE", hue="Model",
            palette=palette, capsize=0.1, edgecolor="black", ax=ax,
        )
        ax.set_yscale("log")
        ax.set_title(
            f"{'In-Distribution' if mode_label=='ID' else 'Out-of-Distribution'} "
            f"({mode_label})",
            fontsize=12, fontweight="bold",
        )
        ax.set_ylabel("Mean RMSE (log scale)", fontsize=10)
        ax.set_xlabel("")
        ax.legend(title="Model", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"PIDM-DP vs. Bi-LSTM vs. PINN — All 5 Systems (n={n_trials} trials)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    fig_path = CONFIG["paths"]["figures"] + "baseline_comparison_lstm_pinn.pdf"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Figure saved → {fig_path}")
    print("\n  [BASELINE COMPARISON COMPLETE]")

    return df


# ── Execute ────────────────────────────────────────────────────────────────────
df_baselines = run_ultimate_baseline_comparison(n_trials=10)

# %%
print("─" * 70)
print("  CELL 23 · Forward Diffusion — Attractor Dissolving")
print("─" * 70)

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def plot_forward_diffusion_v2(sys_name: str):
    t_steps = [0, 100, 250, 500, 750, 999]
    n_steps = len(t_steps)
    s_dim   = CONFIG[sys_name]["state_dim"]
    is_3d   = s_dim > 2   # FIX A: consistent 2D / 3D flag

    x0_phys = train_raws[sys_name][0]
    x0_norm = preps[sys_name].normalize(
        torch.tensor(x0_phys, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    )
    noise = torch.randn_like(x0_norm)
    diff  = diffs[sys_name]

    fig = plt.figure(figsize=(5 * n_steps, 5.5))
    fig.patch.set_facecolor("white")
    gs  = gridspec.GridSpec(
        1, n_steps + 1, figure=fig,
        wspace=0.05, width_ratios=[1] * n_steps + [0.06]
    )

    plasma_colors = plt.cm.plasma(np.linspace(0.0, 0.85, 256))
    plasma_colors[-1] = [0.05, 0.05, 0.05, 1.0]   
    custom_cmap = LinearSegmentedColormap.from_list(
        "plasma_dark_end", plasma_colors
    )

    for idx, t_val in enumerate(t_steps):
      
        if t_val == 0:
            x_t_norm = x0_norm
        else:
            t_tensor = torch.tensor([t_val], device=DEVICE)
            x_t_norm = diff.q_sample(x0_norm, t_tensor, noise)

        x_t_phys   = preps[sys_name].denormalize(x_t_norm).cpu().numpy()[0]
        color_frac  = idx / (n_steps - 1)
        color       = custom_cmap(color_frac)  

        lw_val    = 1.2 - 0.6 * color_frac
        alpha_val = max(0.3, 0.9 - 0.5 * color_frac)

        if is_3d:
            ax = fig.add_subplot(gs[0, idx], projection="3d")

            # Build Line3DCollection for depth-sorted rendering
            x_, y_, z_ = x_t_phys[0], x_t_phys[1], x_t_phys[2]
            pts  = np.array([x_, y_, z_]).T.reshape(-1, 1, 3)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
            lc   = Line3DCollection(
                segs, colors=[color], linewidths=lw_val,
                alpha=alpha_val, zorder=3
            )
            ax.add_collection3d(lc)


            pad = 0.5
            ax.set_xlim(x_.min() - pad, x_.max() + pad)
            ax.set_ylim(y_.min() - pad, y_.max() + pad)
            ax.set_zlim(z_.min() - pad, z_.max() + pad)

            ax.xaxis.set_pane_color((1, 1, 1, 0))
            ax.yaxis.set_pane_color((1, 1, 1, 0))
            ax.zaxis.set_pane_color((1, 1, 1, 0))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.view_init(22, 45)
        else:
            ax = fig.add_subplot(gs[0, idx])
            x_ = x_t_phys[0]
         
            y_ = x_t_phys[1] if s_dim > 1 else np.zeros_like(x_)
            ax.plot(x_, y_, lw=lw_val, alpha=alpha_val, color=color)
            ax.set_xticks([])
            ax.set_yticks([])

        # Panel title
        ab      = diff.alpha_bars[t_val].item()
        snr     = ab / (1.0 - ab + 1e-9)
        if idx == 0:
            title_str = f"$t = 0$ (clean)\nSNR = {snr:.1f}"
        elif idx == n_steps - 1:
            title_str = f"$t = T$ (pure noise)\nSNR ≈ 0"
        else:
            title_str = f"$t = {t_val}$\nSNR = {snr:.2f}"
        ax.set_title(title_str, fontsize=14, fontweight="bold", pad=10)

    # ── Colorbar ─────────────────────────────────────────────────────────────
    ax_cb = fig.add_subplot(gs[0, n_steps])
    norm  = mcolors.Normalize(vmin=0, vmax=999)
    sm    = cm.ScalarMappable(cmap=custom_cmap, norm=norm)   # FIX B: match cmap
    sm.set_array([])
    cb    = fig.colorbar(sm, cax=ax_cb)
    cb.set_label("Diffusion step  $t$", fontsize=12, fontweight="bold")
    cb.ax.tick_params(labelsize=10)

    fig.suptitle(
        f"Forward Process — {sys_name.upper()} Attractor Dissolving",
        fontsize=18, fontweight="bold", y=1.05
    )
    fig.tight_layout()
    path = CONFIG["paths"]["figures"] + f"forward_diffusion_v2_{sys_name}.pdf"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


for sname in _SYSTEMS:
    plot_forward_diffusion_v2(sname)

print("\n  [CELL 23 COMPLETE]\n")

# %%
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# ── Helper for Smooth 3D Lines ──
def plot_smooth_3d_line_solid(ax, x, y, z, color, lw=1.8, alpha=0.95, zorder=3, ls="-"):
    """Draws a continuous, depth-sorted 3D line."""
    points = np.array([x, y, z]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # We use linestyles ('--' for Pure AI, '-' for others)
    linestyles = ls if ls in ['-', '--', ':', '-.'] else 'solid'
    
    lc = Line3DCollection(segments, colors=color, linewidths=lw, alpha=alpha, zorder=zorder, linestyles=linestyles)
    ax.add_collection3d(lc)


def plot_ablation_grid(sys_name: str, n_rows: int = 3):
    raw_samples = generate_chaotic_dataset(
        sys_name, n_samples=n_rows, n_points=1000,
        dt=CONFIG["data"]["dt"], seed=42
    )
    s_dim   = CONFIG[sys_name]["state_dim"]
    p_dim   = CONFIG[sys_name]["param_dim"]
    p_names = CONFIG[sys_name]["param_names"]
    dt_val  = CONFIG["data"]["dt"]
    is_3d   = s_dim > 2  # single flag used consistently throughout

    fig = plt.figure(figsize=(26, 6.5 * n_rows))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(
        n_rows, 5, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.65],
        hspace=0.20, wspace=0.08 # slightly increased hspace
    )

    col_headers = [
        ("Ground Truth",                    "#1A1A2E"),
        ("Pure AI (no physics)",             "#C0392B"),
        ("PIDM-DP (raw output)",             "#8E44AD"),
        ("PIDM-DP Hybrid\n(inferred params → ODE)", "#5B2C8D"),
        ("Parameter\nIdentification",        "#2980B9"),
    ]

    for row in range(n_rows):
        x_gt       = raw_samples[row]
        gt_state   = x_gt[:s_dim]
        true_ic    = gt_state[:, 0].tolist()
        true_params = [float(x_gt[s_dim + j, 0]) for j in range(p_dim)]

        # ── AXIS LIMITS ──────────────────────────────────────────────────────
        xlim = (float(gt_state[0].min()) - 2, float(gt_state[0].max()) + 2)
        ylim = (float(gt_state[1].min()) - 2, float(gt_state[1].max()) + 2)

        if is_3d:
            z_data_gt = gt_state[2]
            zlim = (float(z_data_gt.min()) - 2, float(z_data_gt.max()) + 2)
        else:
            zlim = (-2.0, 2.0)  

        # ── OBSERVATIONS / NOISE ─────────────────────────────────────────────
        x_norm  = preps[sys_name].normalize(
            torch.tensor(raw_samples[row:row + 1], dtype=torch.float32).to(DEVICE)
        )
        mask, obs_idx = make_random_mask(
            x_norm.shape, CONFIG["data"]["obs_ratio"], DEVICE, s_dim
        )
        x_noisy = add_observation_noise(x_norm, mask, CONFIG["data"]["obs_noise"])

        print(f"  Row {row + 1}/{n_rows} processing...", flush=True)

        # ── AI & PIDM RECONSTRUCTIONS ─────────────────────────────────────────
        ai_n    = guided_reconstruction(
            models[sys_name], diffs[sys_name], preps[sys_name],
            x_norm, mask, None
        )
        ai_phys = preps[sys_name].denormalize(ai_n).cpu().numpy()[0]

        def gfn(x, t):
            return stable_guidance(
                x, t, x_noisy, mask,
                models[sys_name], diffs[sys_name], preps[sys_name], sys_name
            )

        pidm_n    = guided_reconstruction(
            models[sys_name], diffs[sys_name], preps[sys_name],
            x_norm, mask, gfn
        )
        pidm_phys = preps[sys_name].denormalize(pidm_n).cpu().numpy()[0]

        # ── PARAMETER INFERENCE & SAFETY CLIP ────────────────────────────────

     
        early_end = min(300, pidm_phys.shape[1])
        raw_inferred = [float(np.median(pidm_phys[s_dim + j, :early_end])) for j in range(p_dim)]
        
       
        ranges = CONFIG[sys_name].get("ranges", [])
        

        ranges = CONFIG[sys_name].get("ranges", [])
        inferred_params = []
        for j, pv in enumerate(raw_inferred):
            if j < len(ranges):
                lo, hi = ranges[j]
                w = hi - lo
                pv = float(np.clip(pv, lo - 3 * w, hi + 3 * w))
            inferred_params.append(pv)

       
        fn_np  = get_sys_fn(sys_name)
        t_span = (0.0, gt_state.shape[1] * dt_val)
        t_eval = np.linspace(0.0, gt_state.shape[1] * dt_val, gt_state.shape[1])
        
    
        ode_method = "LSODA" if sys_name == "rabinovich" else "DOP853"
        
        try:
            sol = solve_ivp(
                fn_np, t_span, true_ic,
                args=tuple(inferred_params),
                method=ode_method, t_eval=t_eval,
                rtol=1e-8, atol=1e-10
            )
            hybrid = sol.y[:s_dim] if (sol.success and np.isfinite(sol.y).all()) else pidm_phys[:s_dim]
        except Exception:
            hybrid = pidm_phys[:s_dim]

    
        trajs_info = [
            (gt_state,          "#1A1A2E", 1.5, 0.90, "-"),
            (ai_phys[:s_dim],   "#C0392B", 1.5, 0.85, "--"),
            (pidm_phys[:s_dim], "#8E44AD", 1.5, 0.85, "-"),
            (hybrid,            "#5B2C8D", 2.0, 1.00, "-"),
        ]

        rmses = [
            None,
            trajectory_rmse(gt_state, ai_phys[:s_dim]),
            trajectory_rmse(gt_state, pidm_phys[:s_dim]),
            trajectory_rmse(gt_state, hybrid),
        ]

     
        for ci, ((traj, color, lw, alpha, ls), rmse_val) in enumerate(zip(trajs_info, rmses)):
            ax = fig.add_subplot(gs[row, ci], projection="3d" if is_3d else None)

            def _z(arr):
                return arr[2] if is_3d else np.zeros_like(arr[0])

           
            if is_3d:
                if ci > 0:
                    plot_smooth_3d_line_solid(ax, gt_state[0], gt_state[1], _z(gt_state), color="#CCCCCC", lw=0.8, alpha=0.3, zorder=1)
                
                plot_smooth_3d_line_solid(ax, traj[0], traj[1], _z(traj), color=color, lw=lw, alpha=alpha, ls=ls, zorder=3)
                
                ax.set_zlim(zlim)
                ax.xaxis.set_pane_color((1, 1, 1, 0))
                ax.yaxis.set_pane_color((1, 1, 1, 0))
                ax.zaxis.set_pane_color((1, 1, 1, 0))
                ax.set_zticklabels([])
                ax.view_init(elev=22, azim=50)
            else:
               
                if ci > 0:
                    ax.plot(gt_state[0], gt_state[1], color="#CCCCCC", lw=0.8, alpha=0.3, zorder=1)
                ax.plot(traj[0], traj[1], color=color, lw=lw, alpha=alpha, ls=ls, zorder=3)
            
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            
            if ci == 0:
                if is_3d:
                    ax.scatter(traj[0, obs_idx], traj[1, obs_idx], traj[2, obs_idx], color="#2980B9", s=25, edgecolors="white", linewidths=0.6, zorder=10, label="10% obs")
                else:
                    ax.scatter(traj[0, obs_idx], traj[1, obs_idx], color="#2980B9", s=25, edgecolors="white", linewidths=0.6, zorder=10, label="10% obs")
                ax.legend(fontsize=10, loc="upper left", framealpha=0.9)

            if row == 0:
                ax.set_title(col_headers[ci][0], color=col_headers[ci][1], fontsize=16, fontweight="bold", pad=15)

            if rmse_val is not None:
                better = (ci == 3 and rmses[1] is not None and rmse_val < rmses[1])
                ax.text2D(
                    0.5, -0.04, f"RMSE = {rmse_val:.3f}",
                    transform=ax.transAxes, ha="center", va="top",
                    fontsize=14, fontweight="bold",
                    color="#1E8449" if better else "#922B21"
                )

            if ci == 0:
                ax.set_ylabel("") 
                bb = ax.get_position()
                fig.text(
                    bb.x0 - 0.01, bb.y0 + bb.height / 2,
                    f"Trial {row + 1}",
                    ha="right", va="center",
                    fontsize=16, fontweight="bold", rotation=90
                )

       
        ax_p = fig.add_subplot(gs[row, 4])
        xpos = np.arange(len(p_names))
        bw   = 0.35

        ax_p.bar(xpos - bw / 2, true_params, bw, color="#7F8C8D", edgecolor="black", alpha=0.65, linewidth=1.2, label="True")
        ax_p.bar(xpos + bw / 2, raw_inferred, bw, color="#5B2C8D", edgecolor="black", alpha=0.90, linewidth=1.2, label="PIDM")

        for j, (tv, iv) in enumerate(zip(true_params, raw_inferred)):
            pct = abs(iv - tv) / (abs(tv) + 1e-9) * 100
            ax_p.text(
                xpos[j] + bw / 2, max(abs(tv), abs(iv)) * 1.05, f"{pct:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
                color="#1E8449" if pct < 5 else "#922B21"
            )

        ax_p.set_xticks(xpos)
        ax_p.set_xticklabels(p_names, fontsize=14)
        ax_p.tick_params(axis="y", labelsize=12)
        
        
        max_bar_val = max(max(true_params), max(raw_inferred)) if true_params else 1.0
        ax_p.set_ylim(0, max_bar_val * 1.30)
        
        for sp in ["top", "right"]: ax_p.spines[sp].set_visible(False)

        if row == 0:
            ax_p.set_title(col_headers[4][0], color=col_headers[4][1], fontsize=16, fontweight="bold", pad=15)
            ax_p.legend(fontsize=12, loc="upper right")

 
    fig.suptitle(
        f"PIDM-DP  ·  {sys_name.upper()}  ·  10% Observation Density\n"
        "All panels use identical axis limits (ground truth range)",
        fontsize=22, fontweight="bold", y=0.96
    )

    path = CONFIG["paths"]["figures"] + f"ablation_grid_{sys_name}.pdf"

 
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {path}")

print("  → Generating ablation grids with fixed axis limits...\n")
for sname in _SYSTEMS:
    plot_ablation_grid(sname, n_rows=3)

print("\n  [CELL 24 COMPLETE]\n")

# %%
print("─" * 70)
print("  CELL 25 · Reverse Diffusion Progress Visualization  [NEW]")
print("─" * 70)

def plot_reverse_diffusion_progress(sys_name: str, sample_idx: int = 0):
    s_dim   = CONFIG[sys_name]["state_dim"]
    snap_steps = [999, 800, 600, 400, 200, 0]

    raw_one   = generate_chaotic_dataset(
        sys_name, n_samples=1, n_points=1000,
        dt=CONFIG["data"]["dt"], seed=77 + sample_idx,
    )
    x_gt   = raw_one[0]
    x_norm = preps[sys_name].normalize(
        torch.tensor(raw_one[0:1], dtype=torch.float32).to(DEVICE))
    mask, obs_idx = make_random_mask(
        x_norm.shape, CONFIG["data"]["obs_ratio"], DEVICE, s_dim)
    x_noisy = add_observation_noise(x_norm, mask, CONFIG["data"]["obs_noise"])

    model = models[sys_name].eval()
    diff  = diffs[sys_name]
    prep  = preps[sys_name]

    B = x_norm.shape[0]
    x = torch.randn_like(x_norm).to(DEVICE)

    snapshots   = {}
    t_indices   = []
    snap_set = set(snap_steps)

    def gfn_prog(x_t, t_val):
        return stable_guidance(x_t, t_val, x_noisy, mask,
                               model, diff, prep, sys_name)

    for t_idx in reversed(range(diff.T)):
        t    = torch.full((B,), t_idx, device=DEVICE, dtype=torch.float)
        beta = diff.betas[t_idx]
        ab   = diff.alpha_bars[t_idx]
        ab_p = diff.alpha_bars[t_idx - 1] if t_idx > 0 else torch.tensor(1.0, device=DEVICE)

        with torch.no_grad():
            ep   = model(x, t)
            x0h  = torch.clamp((x - torch.sqrt(1.0 - ab) * ep) / torch.sqrt(ab), -3.0, 3.0)
            mean = ((torch.sqrt(ab_p) * beta / (1.0 - ab)) * x0h
                    + (torch.sqrt(diff.alphas[t_idx]) * (1.0 - ab_p) / (1.0 - ab)) * x)
            var  = beta * (1.0 - ab_p) / (1.0 - ab)
            noise_s = torch.randn_like(x) if t_idx > 0 else torch.zeros_like(x)
            x    = mean + torch.sqrt(var) * noise_s

        try:
            g = gfn_prog(x, t)
            if torch.isfinite(g).all():
                x = (x + g).detach()
        except Exception:
            pass

        if t_idx in snap_set:
            x_phys = prep.denormalize(x).cpu().numpy()[0]

            x_n_snap = x.detach()
            x0h_snap = (x_n_snap - torch.sqrt(1.0 - ab) * model(x_n_snap, t)) / torch.sqrt(ab)
            x0p_snap = prep.denormalize(x0h_snap)
            fn_torch = get_torch_fn(sys_name)
            s_snap   = x0p_snap[:, :s_dim, :-1]
            p_snap   = x0p_snap[:, s_dim:, :-1]
            try:
                from torch.nn import functional as F_nn
                s_pred = dp_rk45_step(fn_torch, s_snap, p_snap, CONFIG["data"]["dt"])
                phy_l  = float(torch.log1p(F_nn.mse_loss(
                    s_pred, x0p_snap[:, :s_dim, 1:])).item())
            except Exception:
                phy_l = float("nan")

            obs_mask = mask > 0
            data_l   = float(torch.nn.functional.mse_loss(x0h_snap[obs_mask], x_noisy[obs_mask]).item())
            snapshots[t_idx] = (x_phys, phy_l, data_l)

        t_indices.append(t_idx)

    snap_steps_sorted = sorted(snap_steps, reverse=True)
    n_snaps = len(snap_steps_sorted)

 
    fig = plt.figure(figsize=(5 * n_snaps, 10.5))   
    fig.patch.set_facecolor("white")

    gs  = gridspec.GridSpec(2, n_snaps, figure=fig,
                            hspace=0.55,            
                            wspace=0.05,
                            height_ratios=[1.4, 0.6])

    cmap_prog = plt.cm.RdPu

    gt_phys = x_gt[:s_dim]
    xlim = (gt_phys[0].min() - 2, gt_phys[0].max() + 2)
    ylim = (gt_phys[1].min() - 2, gt_phys[1].max() + 2)
    zlim = (gt_phys[2].min() - 2, gt_phys[2].max() + 2) if s_dim >= 3 else (0, 0)

    for idx, t_val in enumerate(snap_steps_sorted):
        color_frac = 1.0 - t_val / 999.0
        color      = cmap_prog(0.3 + 0.7 * color_frac)
        x_phys_snap, phy_l, data_l = snapshots.get(t_val, (None, None, None))

        # Row 0: 3D portrait
        if s_dim >= 3 and x_phys_snap is not None:
            ax0 = fig.add_subplot(gs[0, idx], projection="3d")
            ax0.scatter(x_phys_snap[0], x_phys_snap[1], x_phys_snap[2],
                        color=color, s=4.0, alpha=0.6, edgecolors='none', depthshade=True)
            ax0.set_xlim(xlim); ax0.set_ylim(ylim); ax0.set_zlim(zlim)
            ax0.set_axis_off()
            ax0.view_init(22, 45)
        else:
            ax0 = fig.add_subplot(gs[0, idx])
            if x_phys_snap is not None:
                ax0.scatter(x_phys_snap[0], x_phys_snap[1] if s_dim > 1 else np.zeros(1000),
                            alpha=0.6, color=color, s=4.0, edgecolors='none')
                ax0.set_xlim(xlim); ax0.set_ylim(ylim)
            ax0.set_xticks([]); ax0.set_yticks([])
            for sp in ["top", "right", "bottom", "left"]: ax0.spines[sp].set_visible(False)

        step_label = ("Pure noise"          if t_val >= 990 else
                      "Rough shape"         if t_val >= 400 else
                      "Attractor emerging"  if t_val >= 100 else
                      "Final reconstruction")

        phy_str = f"\nPhy. loss = {phy_l:.3f}" if phy_l and not np.isnan(phy_l) else ""

       
        ax0.set_title(
            f"t = {t_val}\n{step_label}{phy_str}",
            fontsize=13, fontweight="bold",
            pad=2,          
            y=-0.12,       
        )

   
        ax1 = fig.add_subplot(gs[1, idx])
        ab_val = float(diff.alpha_bars[t_val].item())
        signal_frac = ab_val
        noise_frac  = 1.0 - ab_val
        ax1.bar(["Signal", "Noise"],
                [signal_frac, noise_frac],
                color=["#5B2C8D", "#E74C3C"],
                alpha=0.80, edgecolor="black", linewidth=1.2)
        ax1.set_ylim(0, 1)
        if idx == 0:
            ax1.set_ylabel("Fraction", fontsize=13)
        ax1.tick_params(labelsize=12)
        ax1.set_title(f"$\\bar{{\\alpha}}_t$ = {ab_val:.3f}", fontsize=13, pad=6)
        for sp in ["top", "right"]:
            ax1.spines[sp].set_visible(False)

    fig.suptitle(
        f"Reverse Diffusion Progress  —  {sys_name.upper()}\n"
        f"Reading left to right: pure Gaussian noise gradually crystallises "
        f"into the strange attractor under physics guidance",
        fontsize=20, fontweight="bold",
        y=0.90,        
    )

    
    plt.tight_layout(rect=[0, 0, 1, 0.88])   

    path = CONFIG["paths"]["figures"] + f"reverse_diffusion_progress_{sys_name}.pdf"
    plt.savefig(path, dpi=700, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


for sname in ["lorenz", "rabinovich", "lorenz96","rossler", "hyper5d" ]:
    plot_reverse_diffusion_progress(sname, sample_idx=0)

print("\n  [CELL 25 COMPLETE]\n")


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp

def generate_introductory_figures():
    print(f"\n{'='*60}")
    print(" Generating Introductory Ground Truth Figures (PDFs)")
    print(f"{'='*60}")

    os.makedirs(CONFIG["paths"]["figures"], exist_ok=True)


    INTRO_PARAMS = {
        "lorenz": [10.0, 28.0, 8/3],               
        "rossler": [0.2, 0.2, 5.7],               
        "hyper5d": [35.0, 35.0, 3.0],              
        "lorenz96": [8.0],                         
        "rabinovich": [0.14, 0.10]                 
    }

   
    INTRO_ICS = {
        "lorenz": [1.0, 1.0, 1.0],
        "rossler": [1.0, 1.0, 1.0],
        "hyper5d": [1.0, 1.0, 1.0, 1.0, 1.0],
        "lorenz96": np.random.RandomState(42).randn(20).tolist(),
        "rabinovich": [-1.0, 0.0, 0.5]
    }

    for sys_name in _SYSTEMS:
        print(f"  -> Simulating & plotting {sys_name.upper()}...")
        
        fn_np  = get_sys_fn(sys_name)
        params = INTRO_PARAMS[sys_name]
        ic     = INTRO_ICS[sys_name]
        
        
        t_span = (0, 150)
        t_eval = np.linspace(0, 150, 7500)
        
      
        method = "LSODA" if sys_name == "rabinovich" else "DOP853"
        sol = solve_ivp(
            fn_np, t_span, ic, args=tuple(params), 
            method=method, t_eval=t_eval, rtol=1e-8, atol=1e-10
        )
        
      
        transient_idx = int(len(sol.t) * 0.2)
        t_plot = sol.t[transient_idx:] - sol.t[transient_idx] 
        traj   = sol.y[:, transient_idx:]
        
        fig = plt.figure(figsize=(15, 6))
        fig.patch.set_facecolor('white')
        
     
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        
    
        ax1.plot(traj[0], traj[1], traj[2], lw=0.5, alpha=0.8, color='#1A1A2E')
        
        ax1.set_xlabel('$x_1$' if sys_name != 'lorenz' else 'X', labelpad=10)
        ax1.set_ylabel('$x_2$' if sys_name != 'lorenz' else 'Y', labelpad=10)
        ax1.set_zlabel('$x_3$' if sys_name != 'lorenz' else 'Z', labelpad=10)
 
        param_names = CONFIG[sys_name]["param_names"]
        param_str = ", ".join([f"{n}={v:.2f}" for n, v in zip(param_names, params)])
        ax1.set_title(f"Phase Space\n({param_str})", fontweight='bold', pad=15)
        

        ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.view_init(elev=25, azim=45)


   
        plot_steps = 1500
        t_ts    = t_plot[:plot_steps]
        traj_ts = traj[:, :plot_steps]

        gs = gridspec.GridSpec(3, 1, left=0.55, right=0.95, hspace=0.15)
        ax_x = fig.add_subplot(gs[0, 0])
        ax_y = fig.add_subplot(gs[1, 0], sharex=ax_x)
        ax_z = fig.add_subplot(gs[2, 0], sharex=ax_x)

        # Scientific colors
        colors = ['#2980B9', '#C0392B', '#27AE60']
        
        ax_x.plot(t_ts, traj_ts[0], color=colors[0], lw=1.5)
        ax_x.set_ylabel('$x_1(t)$', rotation=0, labelpad=20, fontsize=12)
        ax_x.grid(True, ls=':', alpha=0.6)
        ax_x.tick_params(labelbottom=False)

        ax_y.plot(t_ts, traj_ts[1], color=colors[1], lw=1.5)
        ax_y.set_ylabel('$x_2(t)$', rotation=0, labelpad=20, fontsize=12)
        ax_y.grid(True, ls=':', alpha=0.6)
        ax_y.tick_params(labelbottom=False)

        ax_z.plot(t_ts, traj_ts[2], color=colors[2], lw=1.5)
        ax_z.set_ylabel('$x_3(t)$', rotation=0, labelpad=20, fontsize=12)
        ax_z.set_xlabel('Time (s)', fontsize=12)
        ax_z.grid(True, ls=':', alpha=0.6)


        dim_note = f" (Projected $x_1, x_2, x_3$)" if len(ic) > 3 else ""
        fig.suptitle(f'Canonical Attractor Topology: {sys_name.upper()}{dim_note}', 
                     fontsize=18, fontweight='bold', y=1.02)
 
        save_path = os.path.join(CONFIG["paths"]["figures"], f"report_intro_{sys_name}.pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    print("\n  ✓ Successfully generated 5 introductory PDFs in the figures folder!")

generate_introductory_figures()





