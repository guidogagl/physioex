"""NightCleaner: one night in, brain-only EEG, eye-only EOG and muscle-only EMG out.

Order (Croft & Barry: the EOG is cleaned of brain before it is used to clean the EEG):

0. EEG/EOG/extra: mains notch, 0.3-35 Hz; EMG kept at its native rate;
1. heart: R peaks on the ECG (or on the chin when there is no ECG), template
   subtraction on every channel at its native rate; then EEG/EOG to ``fs``;
2. EOG derivation: already in the recipe (E1 - E2 where possible);
3. source separation (rich nights only): SOBI on all EEG/EOG channels, ocular
   components by topography + spectrum + state (bss.classify_ocular);
4. brain left in the EOG: band regression on all clean EEG, fitted on the
   epochs with still eyes, separately for slow-wave and other epochs;
5. eyes left in the EEG: band regression on the clean EOG, fitted on the
   epochs with moving eyes;
6. muscle in the EEG (rich nights only): AMUSE components with low
   autocorrelation whose 20-35 Hz envelope follows the chin envelope;
7. EMG: bipolar chin and legs, 10-100 Hz, notch, cardiac template/blanking,
   spindle regression when the chin carries EEG sigma, RMS envelope.

Label-free: the cleaning never reads the hypnogram, so a night is cleaned the
same way in training and in evaluation. The epochs each fit uses come from
:func:`states.state_masks` (eye, delta and sigma indices of the night); its
quantiles are calibrated against the stages of training nights only.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from physioex.data.cleaning.bss import amuse, classify_ocular, sobi
from physioex.data.cleaning.cardiac import (
    cardiac_residual,
    detect_r_peaks,
    subtract_cardiac,
)
from physioex.data.cleaning.emg import clean_emg, notch_mains, rms_envelope
from physioex.data.cleaning.montage import (
    CleanMontage,
    ResolvedMontage,
    read_derivation,
    resolve_montage,
)
from physioex.data.cleaning.regression import band_regression, fft_band
from physioex.data.cleaning.states import state_masks
Signal = Tuple[np.ndarray, float]

#: Names of the clean outputs, as requested from a dataset.
CLEAN_CHANNELS = (
    "EEG_clean",
    "EOG_clean",
    "EMG_clean",
    "EMG_ENV_clean",
    "LEG_L_clean",
    "LEG_R_clean",
)


def resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if fs_in == fs_out:
        return np.asarray(x, dtype=np.float64)
    r = Fraction(fs_out / fs_in).limit_denominator(1000)
    return resample_poly(np.asarray(x, dtype=np.float64), r.numerator, r.denominator)


@dataclass
class CleanNight:
    outputs: Dict[str, Signal]  # keys in CLEAN_CHANNELS that could be produced
    level: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NightCleaner:
    montage: CleanMontage
    fs: float = 100.0
    band: Tuple[float, float] = (0.3, 35.0)
    sobi_lags: Tuple[int, ...] = tuple(range(1, 101, 5))
    sobi_subsample: int = 2
    lf_share: float = 0.6
    activity_ratio: float = 1.5
    muscle_autocorr_max: float = 0.6
    muscle_env_corr_min: float = 0.3
    spindle_corr_max: float = 0.1
    env_fs: float = 10.0
    # state_masks quantiles (calibrated on training nights)
    quiet_q: float = 0.4
    active_q: float = 0.8
    deep_q: float = 0.75
    sigma_q: float = 0.7

    def spec(self) -> str:
        p = asdict(self)
        p["montage"] = self.montage.spec()
        h = hashlib.sha1(
            json.dumps(p, sort_keys=True, default=str).encode()
        ).hexdigest()
        return "NightCleaner:" + h[:12]

    # ------------------------------------------------------------------ reading
    def read(
        self, edf_path
    ) -> Tuple[Dict[str, Signal], Dict[str, Signal], ResolvedMontage]:
        """Resolve the recipe on an EDF and read every derivation (µV, native rate)."""
        import pyedflib

        with pyedflib.EdfReader(str(edf_path)) as f:
            labels = [s.strip() for s in f.getSignalLabels()]
            fs = {lab: float(f.getSampleFrequency(i)) for i, lab in enumerate(labels)}
            r = resolve_montage(self.montage, labels, fs)
            sig = {
                o: read_derivation(f, d)
                for o, d in r.derivations.items()
                if d is not None
            }
            extra = {k: read_derivation(f, d) for k, d in r.extra.items()}
        return sig, extra, r

    def clean_edf(self, edf_path) -> CleanNight:
        sig, extra, r = self.read(edf_path)
        night = self.clean(sig, extra, level=r.level)
        night.diagnostics["resolved"] = {
            o: (d.label if d else None) for o, d in r.derivations.items()
        }
        night.diagnostics["level_reasons"] = r.reasons
        return night

    # ------------------------------------------------------------------ cleaning
    def _prep(
        self, x: np.ndarray, fs: float, peaks: Optional[np.ndarray]
    ) -> np.ndarray:
        x = notch_mains(np.asarray(x, dtype=np.float64), fs, self.montage.mains)
        hi = min(self.band[1], 0.45 * fs)
        x = sosfiltfilt(
            butter(4, [self.band[0], hi], btype="band", fs=fs, output="sos"), x
        )
        if peaks is not None:
            x = subtract_cardiac(x, fs, peaks)
        return resample(x, fs, self.fs)

    def clean(
        self,
        signals: Dict[str, Signal],
        extra: Dict[str, Signal],
        level: Optional[str] = None,
    ) -> CleanNight:
        """``signals`` keyed by montage outputs (eeg, eog, chin, leg_l, leg_r, ecg), ``extra`` by recipe extra names."""
        level = level or self.montage.level
        diag: Dict[str, Any] = {"spec": self.spec()}
        if "eeg" not in signals or "eog" not in signals:
            return CleanNight({}, "missing", {**diag, "error": "no EEG or EOG"})

        # 1. R peaks: ECG first, chin as surrogate
        peaks, src, q = None, None, 0.0
        for k in ("ecg", "chin"):
            if k in signals and signals[k][1] >= 50:
                peaks, q = detect_r_peaks(*signals[k])
                if peaks is not None:
                    src = k
                    break
        diag["cardiac"] = {
            "source": src,
            "quality": q,
            "n_peaks": 0 if peaks is None else len(peaks),
        }
        if peaks is None:
            diag["cardiac"]["note"] = "no usable heartbeat: cardiac field not removed"

        # 0 + 1. EEG / EOG / extras at self.fs, cardiac removed at native rate
        eeg_x = {"eeg": self._prep(*signals["eeg"], peaks)}
        ocu_x = {"eog": self._prep(*signals["eog"], peaks)}
        for k, (x, fs) in extra.items():
            target = eeg_x if k in self.montage.extra_eeg else ocu_x
            target[k] = self._prep(x, fs, peaks)
        n = min(len(v) for v in {**eeg_x, **ocu_x}.values())
        eeg_x = {k: v[:n] for k, v in eeg_x.items()}
        ocu_x = {k: v[:n] for k, v in ocu_x.items()}
        if peaks is not None:
            diag["cardiac"]["residual_uV"] = {
                "eeg": cardiac_residual(eeg_x["eeg"], self.fs, peaks),
                "eog": cardiac_residual(ocu_x["eog"], self.fs, peaks),
            }
        st = self.states(eeg_x["eeg"], ocu_x["eog"])
        scored, quiet, active = st["ok"], st["quiet"], st["active"]
        diag["states"] = {k: round(float(v.mean()), 3) for k, v in st.items()}

        # 3. source separation (rich)
        eog_c, eeg_c = ocu_x["eog"], dict(eeg_x)
        if level == "rich":
            keys = list(ocu_x) + list(eeg_x)
            X = np.stack([ocu_x[k] if k in ocu_x else eeg_x[k] for k in keys])
            X = X - X.mean(1, keepdims=True)
            idx = np.where(scored)[0][:: self.sobi_subsample]
            U, A = sobi(X[:, idx], lags=self.sobi_lags)
            S = U @ X
            ocu, crit = classify_ocular(
                S,
                A,
                keys,
                list(ocu_x),
                self.fs,
                active,
                quiet,
                lf_share=self.lf_share,
                activity_ratio=self.activity_ratio,
            )
            if ocu.any():
                Xo, Xc = A[:, ocu] @ S[ocu], A[:, ~ocu] @ S[~ocu]
                eog_c = Xo[keys.index("eog")]
                eeg_c = {k: Xc[keys.index(k)] for k in eeg_x}
            else:  # no ocular component: the EOG would be zero, keep the poor path
                diag["sobi_note"] = "no ocular component: separation not applied"
            diag["sobi"] = {
                "n_components": int(S.shape[0]),
                "n_ocular": int(ocu.sum()),
                "lf_share": crit["lf_share"].round(3).tolist(),
                "activity_ratio": np.round(crit["activity_ratio"], 3).tolist(),
            }

        # 4. brain left in the EOG: coupling differs in slow-wave epochs, own coefficients there
        if quiet.any():
            deep = st["deep"]
            fit_d, fit_o = quiet & deep, quiet & ~deep
            reg = lambda fit: band_regression({"eog": eog_c}, eeg_c, fit, self.fs)[0]["eog"]  # noqa: E731
            if fit_d.sum() > 10 * self.fs and fit_o.sum() > 10 * self.fs:
                eog_c = np.where(deep, reg(fit_d), reg(fit_o))
            else:
                eog_c = reg(quiet)
        else:
            diag["eog_regression"] = "skipped: no epoch with still eyes"
        # brain-eye coupling before step 5 (step 5 subtracts the EOG from the EEG)
        diag["r_eog_eeg_before_step5"] = {
            name: float(
                np.corrcoef(
                    fft_band(eog_c, 0.3, 4.0, self.fs)[m],
                    fft_band(eeg_c["eeg"], 0.3, 4.0, self.fs)[m],
                )[0, 1]
            )
            for name, m in (("quiet", quiet & ~st["deep"]), ("deep", quiet & st["deep"]))
            if m.sum() > 10
        }
        # 5. eyes left in the EEG
        if active.any():
            eeg_c["eeg"] = band_regression(
                {"eeg": eeg_c["eeg"]}, {"eog": eog_c}, active, self.fs
            )[0]["eeg"]
        else:
            diag["eeg_regression"] = "skipped: no epoch with moving eyes"

        out: Dict[str, Signal] = {"EOG_clean": (eog_c, self.fs)}

        # 7. EMG (before 6: the chin envelope guides the muscle components)
        chin_env = None
        if "chin" in signals:
            x, fs = signals["chin"]
            if fs < 25:  # an envelope already (Sleep-EDF): passed through
                out["EMG_ENV_clean"] = (np.asarray(x, dtype=np.float64), fs)
                diag["chin"] = {"note": f"envelope at {fs} Hz, not cleaned"}
            else:
                y, d = clean_emg(x, fs, self.montage.mains, peaks)
                y, d["spindle"] = self._spindles(y, fs, eeg_c["eeg"], st["sigma"])
                out["EMG_clean"] = (y, fs)
                out["EMG_ENV_clean"] = (
                    rms_envelope(y, fs, 1.0, self.env_fs),
                    self.env_fs,
                )
                chin_env = rms_envelope(y, fs, 1.0, 1.0)
                diag["chin"] = d
        for k, name in (("leg_l", "LEG_L_clean"), ("leg_r", "LEG_R_clean")):
            if k in signals:
                try:
                    y, d = clean_emg(*signals[k], self.montage.mains, peaks)
                    out[name] = (y, signals[k][1])
                    diag[k] = d
                except ValueError as e:
                    diag[k] = {"note": str(e)}

        # 6. muscle in the EEG (rich)
        if level == "rich" and chin_env is not None:
            eeg_c["eeg"], diag["muscle"] = self._muscle(eeg_c, chin_env)
        else:
            diag["muscle"] = {"note": "not removed (poor night or no chin)"}
        out["EEG_clean"] = (eeg_c["eeg"], self.fs)
        diag["n_samples"] = n
        return CleanNight(out, level, diag)

    def _muscle(
        self, eeg_c: Dict[str, np.ndarray], chin_env: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        keys = list(eeg_c)
        X = np.stack([eeg_c[k] for k in keys])
        mu = X.mean(1, keepdims=True)
        U, A, ac = amuse(X - mu)
        S = U @ (X - mu)
        sec = int(self.fs)
        m = min(len(chin_env), S.shape[1] // sec)
        corr = np.zeros(S.shape[0])
        for i in range(S.shape[0]):
            hb = fft_band(S[i], 20.0, 35.0, self.fs)[: m * sec].reshape(m, sec)
            env = np.sqrt((hb**2).mean(1))
            corr[i] = np.corrcoef(env, chin_env[:m])[0, 1] if env.std() > 0 else 0.0
        drop = (ac < self.muscle_autocorr_max) & (corr > self.muscle_env_corr_min)
        x = X[keys.index("eeg")] - (
            A[keys.index("eeg"), drop] @ S[drop] if drop.any() else 0.0
        )
        return x, {
            "n_components": len(ac),
            "n_removed": int(drop.sum()),
            "autocorr": ac.round(3).tolist(),
            "env_corr": corr.round(3).tolist(),
        }

    def states(self, eeg: np.ndarray, eog: np.ndarray) -> Dict[str, np.ndarray]:
        """Label-free epoch masks of the night (see :func:`states.state_masks`)."""
        return state_masks(
            eeg,
            eog,
            self.fs,
            quiet_q=self.quiet_q,
            active_q=self.active_q,
            deep_q=self.deep_q,
            sigma_q=self.sigma_q,
        )

    def _spindles(
        self, chin: np.ndarray, fs: float, eeg: np.ndarray, sigma: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Regress the EEG sigma band out of the chin when they correlate in spindle-rich epochs."""
        e = resample(fft_band(eeg, 11.0, 16.0, self.fs), self.fs, fs)
        n = min(len(e), len(chin))
        c = fft_band(chin[:n], 11.0, 16.0, fs)
        idx = np.floor(np.arange(n) * self.fs / fs).astype(int)
        m = np.zeros(n, bool)
        ok = idx < len(sigma)
        m[ok] = sigma[idx[ok]]
        if not m.any():
            return chin, {"note": "no spindle-rich epoch"}
        r = float(np.corrcoef(c[m], e[:n][m])[0, 1])
        d = {"corr_sigma": r}
        if abs(r) > self.spindle_corr_max:
            b = float(np.dot(e[:n][m], c[m]) / np.dot(e[:n][m], e[:n][m]))
            chin = chin.copy()
            chin[:n] -= b * e[:n]
            d["regressed"] = b
        return chin, d
