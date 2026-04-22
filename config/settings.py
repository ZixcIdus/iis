"""Централизованные настройки проекта анализа версий ИИС."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"

SUPPORTED_DATASETS = ("wesad", "dreamer", "deap", "case", "eva_med", "ds002722", "ds002724")
PROCESSING_MODES = ("strict", "proxy", "hybrid")

DATASET_DEFAULTS = {
    "wesad": {
        "window_seconds": 30,
        "step_seconds": 30,
        "default_root": DATA_DIR / "WESAD",
    },
    "dreamer": {
        "window_seconds": 10,
        "step_seconds": 10,
        "default_root": DATA_DIR / "DREAMER",
    },
    "deap": {
        "window_seconds": 10,
        "step_seconds": 10,
        "default_root": DATA_DIR / "DEAP",
    },
    "case": {
        "window_seconds": 10,
        "step_seconds": 10,
        "default_root": DATA_DIR / "CASE",
    },
    "eva_med": {
        "window_seconds": 4,
        "step_seconds": 4,
        "default_root": DATA_DIR / "EVA-MED",
    },
    "ds002722": {
        "window_seconds": 21,
        "step_seconds": 21,
        "default_root": DATA_DIR / "DS002722",
    },
    "ds002724": {
        "window_seconds": 21,
        "step_seconds": 21,
        "default_root": DATA_DIR / "DS002724",
    },
}

DYNAMIC_WINDOW_DEFAULTS = {
    "wesad": {"window_seconds": 10.0, "step_seconds": 5.0},
    "dreamer": {"window_seconds": 5.0, "step_seconds": 2.5},
    "deap": {"window_seconds": 5.0, "step_seconds": 2.5},
    "case": {"window_seconds": 5.0, "step_seconds": 2.0},
    "eva_med": {"window_seconds": 4.0, "step_seconds": 2.0},
    # Для динамической V4 на BCMI более длинное окно даёт устойчивее HRV и меньше ложных скачков.
    "ds002722": {"window_seconds": 8.0, "step_seconds": 4.0},
    "ds002724": {"window_seconds": 8.0, "step_seconds": 4.0},
}

EPSILON = 1e-8
MIN_SIGNAL_SAMPLES = 256
MIN_RR_COUNT = 6
RR_INTERPOLATION_FREQUENCY = 4.0
PLOT_DPI = 160
HISTOGRAM_BINS = 30
SENSITIVITY_DELTA = 0.05
SUMMARY_FLOAT_DIGITS = 6
EEG_POWER_UV2_SCALE = 1e12
DYNAMIC_ROLLING_WINDOW = 5
DYNAMIC_MAX_RECORDS = 4
DYNAMIC_MIN_POINTS = 6
DYNAMIC_EWMA_ALPHA = 0.35
DYNAMIC_VOLATILITY_WINDOW = 4
DYNAMIC_VOLATILITY_SCALE = 0.035
DYNAMIC_SMOOTHING_BLEND = 0.60
DYNAMIC_SMOOTH_WEIGHT_MIN = 0.10
DYNAMIC_SMOOTH_WEIGHT_MAX = 0.70
DYNAMIC_FAST_ALPHA = 0.42
DYNAMIC_RESPONSE_ALPHA_BASE = 0.48
DYNAMIC_RESPONSE_ALPHA_SCALE = 0.22
DYNAMIC_RECOVERY_ALPHA_BASE = 0.12
DYNAMIC_RECOVERY_ALPHA_SCALE = 0.10
DYNAMIC_GAIN_MIN = 0.08
DYNAMIC_GAIN_MAX = 0.85

EEG_BANDS = {
    "full": (1.0, 45.0),
    "alpha": (8.0, 13.0),
    "gamma": (30.0, 45.0),
}

HRV_BANDS = {
    "lf": (0.04, 0.15),
    "hf": (0.15, 0.40),
}

WESAD_SAMPLING_RATES = {
    "ecg": 700,
    "resp": 700,
    "eda": 700,
    "emg": 700,
    "temp": 700,
}

WESAD_LABEL_MAP = {
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}

DREAMER_EEG_CHANNELS = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]

DREAMER_LEFT_CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1"]
DREAMER_RIGHT_CHANNELS = ["AF4", "F8", "F4", "FC6", "T8", "P8", "O2"]

DEAP_EEG_CHANNELS = [
    "Fp1",
    "AF3",
    "F3",
    "F7",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "CP5",
    "CP1",
    "P3",
    "P7",
    "PO3",
    "O1",
    "Oz",
    "Pz",
    "Fp2",
    "AF4",
    "Fz",
    "F4",
    "F8",
    "FC6",
    "FC2",
    "Cz",
    "C4",
    "T8",
    "CP6",
    "CP2",
    "P4",
    "P8",
    "PO4",
    "O2",
]

DEAP_LEFT_CHANNELS = [
    "Fp1",
    "AF3",
    "F3",
    "F7",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "CP5",
    "CP1",
    "P3",
    "P7",
    "PO3",
    "O1",
]

DEAP_RIGHT_CHANNELS = [
    "Fp2",
    "AF4",
    "F4",
    "F8",
    "FC6",
    "FC2",
    "C4",
    "T8",
    "CP6",
    "CP2",
    "P4",
    "P8",
    "PO4",
    "O2",
]

DEAP_PERIPHERAL_CHANNELS = [
    "hEOG",
    "vEOG",
    "zEMG",
    "tEMG",
    "GSR",
    "Respiration",
    "Plethysmograph",
    "Temperature",
]

BCMI_EEG_CHANNELS = [
    "FP1",
    "FPz",
    "FP2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FT9",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "FT10",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "TP9",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "TP10",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
]

BCMI_LEFT_CHANNELS = [
    "FP1",
    "F7",
    "F3",
    "FT9",
    "FC5",
    "FC1",
    "T7",
    "C3",
    "TP9",
    "CP5",
    "CP1",
    "P7",
    "P3",
    "O1",
]

BCMI_RIGHT_CHANNELS = [
    "FP2",
    "F8",
    "F4",
    "FT10",
    "FC6",
    "FC2",
    "T8",
    "C4",
    "TP10",
    "CP6",
    "CP2",
    "P8",
    "P4",
    "O2",
]

BCMI_TARGET_LABELS = {
    1: "LVLA",
    2: "NVLA",
    3: "HVLA",
    4: "LVNA",
    5: "NVNA",
    6: "HVNA",
    7: "LVHA",
    8: "NVHA",
    9: "HVHA",
}

CASE_SAMPLING_RATES = {
    "ecg": 1000.0,
    "ppg": 1000.0,
    "gsr": 1000.0,
    "resp": 1000.0,
    "temperature": 1000.0,
    "annotation": 20.0,
}

CASE_VIDEO_LABEL_MAP = {
    "startvid": "baseline",
    "endvid": "baseline",
    "bluvid": "baseline",
    "relaxed-1": "baseline",
    "relaxed-2": "baseline",
    "scary-1": "stress",
    "scary-2": "stress",
    "amusing-1": "disbalance",
    "amusing-2": "disbalance",
    "boring-1": "disbalance",
    "boring-2": "disbalance",
}

EVA_MED_LEFT_CHANNELS = [
    "FP1",
    "AF3",
    "AF7",
    "F1",
    "F3",
    "F5",
    "F7",
    "FC1",
    "FC3",
    "FC5",
    "FT7",
    "C1",
    "C3",
    "C5",
    "T7",
    "CP1",
    "CP3",
    "CP5",
    "TP7",
    "P1",
    "P3",
    "P5",
    "P7",
    "PO3",
    "PO5",
    "O1",
]

EVA_MED_RIGHT_CHANNELS = [
    "FP2",
    "AF4",
    "AF8",
    "F2",
    "F4",
    "F6",
    "F8",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "C2",
    "C4",
    "C6",
    "T8",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO4",
    "PO6",
    "O2",
]

LABEL_ENCODING = {
    "baseline": 0.0,
    "rest": 0.0,
    "calm": 0.0,
    "amusement": 0.35,
    "neutral": 0.40,
    "disbalance": 0.60,
    "activated": 0.70,
    "stress": 1.0,
    "stress_like": 1.0,
}

STATE_THRESHOLDS = {
    "dreamer": {"low": 2.5, "high": 3.5},
    "deap": {"low": 4.5, "high": 5.5},
    "case": {"low": 3.5, "high": 6.5},
    "eva_med": {"low": 2.5, "high": 3.5},
    "ds002722": {"low": 3.5, "high": 5.5},
    "ds002724": {"low": 3.5, "high": 5.5},
}

SELF_REPORT_RANGES = {
    "dreamer": (1.0, 5.0),
    "deap": (1.0, 9.0),
    "case": (0.5, 9.5),
    "eva_med": (1.0, 9.0),
    "ds002722": (0.0, 9.0),
    "ds002724": (0.0, 9.0),
}

NORMALIZATION_RANGES = {
    "alpha_asymmetry": (-1.2, 1.2),
    "gamma_log_power": (-3.0, 2.0),
    "gamma_alpha_ratio": (0.0, 6.0),
    "lf_hf": (0.2, 6.0),
    "hf_lf": (0.1, 4.0),
    "heart_rate": (45.0, 140.0),
    "rmssd": (10.0, 150.0),
    "sdnn": (10.0, 180.0),
    "stress_label": (0.0, 1.0),
}

MODEL_COMPONENTS = ("A", "Gamma", "H", "V", "Q", "K")
SENSITIVITY_COMPONENTS = ("A", "Gamma", "H", "V")

MODEL_CONFIGS = {
    "IISVersion1": {
        "weights": {
            "A": 0.35,
            "Gamma": 0.25,
            "H": 0.30,
            "V": 0.10,
        },
    },
    "IISVersion2": {
        "weights": {
            "A": 0.35,
            "Gamma": 0.25,
            "H": 0.30,
            "V": 0.10,
        },
    },
    "IISVersion3": {
        "weights": {
            "A": 0.35,
            "Gamma": 0.25,
            "H": 0.30,
            "V": 0.10,
        },
    },
    "IISVersion4": {
        "weights": {
            "A": 0.10,
            "Gamma": 0.05,
            "V": 0.25,
            "Q": 0.60,
        },
    },
    "IISVersion5": {
        "weights": {
            "A": 0.12,
            "Gamma": 0.07,
            "V": 0.24,
            "Q": 0.57,
        },
    },
    "IISVersion6": {
        "weights": {
            "A": 0.14,
            "Gamma": 0.08,
            "V": 0.24,
            "Q": 0.54,
        },
    },
    "IISVersion7": {
        "weights": {
            "A": 0.15,
            "Gamma": 0.08,
            "V": 0.24,
            "Q": 0.53,
        },
    },
}

PROXY_RANGES = {
    "dopamine": (25.0, 80.0),
    "cortisol": (8.0, 25.0),
    "hf_reference": (20.0, 120.0),
    "gamma_reference": (20.0, 100.0),
}

VERSION3_K_PARAMS = {
    "alpha": 1.2,
    "beta": 0.24,
    "fatigue_k": 5.0,
    "fatigue_s0": 0.6,
    "valence_beta": 2.0,
}

VERSION4_CALIBRATION = {
    "alpha_asymmetry_center": 0.0006126058755668,
    "alpha_asymmetry_width": 0.0005296436866331,
    "total_asymmetry_center": -0.029885183821534496,
    "total_asymmetry_width": 0.028709543894737545,
    "gamma_log_center": 1.5434688675501878,
    "gamma_log_width": 0.8496713486963336,
    "gamma_alpha_log_center": 0.0003674266913414713,
    "gamma_alpha_log_width": 0.00043434484108849444,
    "heart_rate_center": 118.74019555606847,
    "heart_rate_width": 46.71845481469046,
    "hf_lf_log_center": 0.9675457417400027,
    "hf_lf_log_width": 1.2220370048971366,
    "q_proxy_weight": 0.35,
}

VERSION5_CALIBRATION = {
    "q_linear_gain": 1.4,
    "q_coherence_gain": 0.55,
    "q_mismatch_penalty": 0.50,
    "q_gamma_penalty": 0.18,
    "q_proxy_weight": 0.25,
    "q_signed_power": 0.82,
    "q_synergy_gain": 0.42,
    "q_energy_gain": 0.28,
    "q_conflict_gain": 0.24,
    "q_gamma_v_penalty": 0.16,
    "output_center": 0.08,
    "output_gain": 2.6,
    "output_curve": 0.80,
    "output_balance_coupling": 0.18,
    "output_gamma_brake": 0.06,
    "output_contrast_mix": 0.35,
    "output_signed_power": 0.84,
    "output_regime_gain": 0.52,
    "output_energy_gain": 0.34,
    "output_conflict_gain": 0.22,
    "output_gamma_cross_penalty": 0.10,
}

VERSION6_CALIBRATION = {
    "shape_power": 0.78,
    "gamma_shape_power": 1.05,
    "gate_temperature": 1.65,
    "q_proxy_weight": 0.18,
    "reg_q_base": 0.76,
    "reg_q_amp": 0.18,
    "mob_q_base": 0.50,
    "mob_q_amp": 0.14,
    "dep_q_base": 0.18,
    "dep_q_amp": 0.16,
    "reg_iis_base": 0.80,
    "reg_iis_amp": 0.16,
    "mob_iis_base": 0.54,
    "mob_iis_amp": 0.12,
    "dep_iis_base": 0.20,
    "dep_iis_amp": 0.16,
    "transition_entropy_weight": 0.08,
    "regime_balance_weight": 0.05,
    "conflict_penalty": 0.10,
}

VERSION7_CALIBRATION = {
    "sumigron_temperature": 1.75,
    "sumigron_level_weight": 1.00,
    "sumigron_structure_weight": 0.42,
    "sumigron_energy_weight": 0.32,
    "component_gain": 1.15,
    "q_gain": 1.85,
    "q_proxy_weight": 0.15,
    "q_coherence_penalty": 0.48,
    "q_gamma_penalty": 0.20,
    "gate_temperature": 1.55,
    "reg_iis_base": 0.80,
    "reg_iis_amp": 0.16,
    "mob_iis_base": 0.53,
    "mob_iis_amp": 0.12,
    "dep_iis_base": 0.20,
    "dep_iis_amp": 0.15,
    "transition_entropy_weight": 0.08,
    "regime_balance_weight": 0.05,
    "conflict_penalty": 0.09,
}

VERSION7_DATASET_OVERRIDES = {
    "dreamer": {
        "sumigron_temperature": 0.9298847019451314,
        "sumigron_structure_weight": 0.32447089642619953,
        "sumigron_energy_weight": 0.2318303047775378,
        "component_gain": 1.1105332584571976,
        "q_gain": 2.6049273216939297,
        "q_coherence_penalty": 0.7319484783158632,
        "q_gamma_penalty": 0.25866560912379244,
        "gate_temperature": 1.136308633470851,
        "reg_iis_base": 0.8654101037471308,
        "reg_iis_amp": 0.17480463463286533,
        "mob_iis_base": 0.45896753729627837,
        "mob_iis_amp": 0.13777127163378194,
        "dep_iis_base": 0.1967023282007362,
        "dep_iis_amp": 0.1415090086598715,
        "transition_entropy_weight": 0.035697242886134434,
        "regime_balance_weight": 0.06219651310141984,
        "conflict_penalty": 0.05399045325613091,
        "weights": {
            "A": 0.17120302234851265,
            "Gamma": 0.05182915508006668,
            "V": 0.27522905070616915,
            "Q": 0.5017387718652515,
        },
    }
}

UTILITY_WEIGHTS = {
    "effect_size": 0.24,
    "inverse_overlap": 0.18,
    "relative_sensitivity": 0.18,
    "arousal_correlation": 0.12,
    "valence_correlation": 0.12,
    "stability": 0.08,
    "coverage": 0.08,
}

RELIABILITY_THRESHOLDS = {
    "min_valid_segments": 20,
    "min_direct_ratio_high": 0.6,
    "min_direct_ratio_medium": 0.3,
}


def ensure_runtime_directories() -> None:
    """Создаёт рабочие каталоги проекта, если они отсутствуют."""

    for path in (DATA_DIR, OUTPUT_DIR, PLOTS_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def dataset_root(dataset_name: str, data_root: Path | None = None) -> Path:
    """Возвращает корневой каталог конкретного датасета."""

    dataset_name = dataset_name.lower()
    if dataset_name not in DATASET_DEFAULTS:
        raise ValueError(f"Неподдерживаемый датасет: {dataset_name}")

    if data_root is None:
        return DATASET_DEFAULTS[dataset_name]["default_root"]

    return Path(data_root).expanduser().resolve()
