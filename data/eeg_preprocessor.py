import mne
import numpy as np
from typing import Optional, List


class EEGPreprocessor:
    """
    纯信号处理型 EEG 预处理器
    只接受 Raw 对象，不负责文件加载
    """

    def __init__(
        self,
        l_freq: float = 1.0,
        h_freq: float = 40.0,
        resample_sfreq: float = 250.0,
        use_ica: bool = True,
        drop_non_eeg: bool = False,
    ):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.resample_sfreq = resample_sfreq
        self.use_ica = use_ica
        self.drop_non_eeg = drop_non_eeg

    # ==========================================================
    # 主预处理流程
    # ==========================================================
    def preprocess(
        self,
        raw: mne.io.BaseRaw,
        montage: Optional[str] = "standard_1020",
        manual_ica_exclude: Optional[List[int]] = None,
    ) -> mne.io.BaseRaw:

        if raw is None:
            raise ValueError("preprocess() 接收到 raw=None")

        print("[EEG] 开始预处理...")

        # ----------------------------------------
        # 设置通道类型
        # ----------------------------------------
        ch_types = raw.get_channel_types()
        mapping = {}
        for ch_name, ch_type in zip(raw.ch_names, ch_types):
            if ch_type.upper() in ["ECG", "EOG", "EMG"]:
                mapping[ch_name] = ch_type.lower()
        if mapping:
            raw.set_channel_types(mapping)

        # ----------------------------------------
        # 设置 Montage
        # ----------------------------------------
        if montage is not None:
            raw.set_montage(montage, on_missing="ignore")

        # ----------------------------------------
        # 滤波
        # ----------------------------------------
        raw.filter(l_freq=self.l_freq, h_freq=self.h_freq)

        # 平均参考
        raw.set_eeg_reference("average", projection=True)
        raw.apply_proj()

        # ----------------------------------------
        # ICA 去伪迹
        # ----------------------------------------
        if self.use_ica:
            print("[EEG] 运行 ICA...")
            n_comp = min(20, len(raw.ch_names) - 1)

            ica = mne.preprocessing.ICA(
                n_components=n_comp,
                random_state=97,
                max_iter="auto",
            )
            ica.fit(raw)

            # 自动 EOG 检测
            eog_picks = mne.pick_types(raw.info, meg=False, eeg=False, eog=True)
            if eog_picks.size > 0:
                eog_indices, _ = ica.find_bads_eog(raw)
                ica.exclude.extend(eog_indices)

            # 手动排除
            if manual_ica_exclude:
                ica.exclude.extend(manual_ica_exclude)

            raw = ica.apply(raw.copy())

        # ----------------------------------------
        # 删除 ECG
        # ----------------------------------------
        if "ECG" in raw.ch_names:
            raw.drop_channels(["ECG"])

        # ----------------------------------------
        # 重采样
        # ----------------------------------------
        if self.resample_sfreq is not None:
            raw.resample(self.resample_sfreq)

        print(f"[EEG] 预处理完成: {len(raw.ch_names)} 通道, {raw.n_times} 采样点")

        return raw

    # ==========================================================
    # Spike 平滑
    # ==========================================================
    def _smooth_spikes(
        self,
        data: np.ndarray,
        percentile: float = 99.0,
        taper_pct: float = 0.05,
    ) -> np.ndarray:

        orig_shape = data.shape

        if data.ndim == 2:
            data = data[np.newaxis, ...]

        n_epochs, n_ch, n_times = data.shape
        n_taper = int(n_times * taper_pct)

        # 头尾 taper
        if n_taper > 0:
            window = np.hanning(n_taper * 2)
            fade_in = window[:n_taper]
            fade_out = window[-n_taper:]

            for ep in range(n_epochs):
                for ch in range(n_ch):
                    data[ep, ch, :n_taper] *= fade_in
                    data[ep, ch, -n_taper:] *= fade_out

        # spike 裁剪
        for ep in range(n_epochs):
            for ch in range(n_ch):
                x = data[ep, ch]
                threshold = np.percentile(np.abs(x), percentile)
                spike_idx = np.where(np.abs(x) > threshold)[0]

                if spike_idx.size > 0:
                    valid_idx = np.setdiff1d(np.arange(n_times), spike_idx)
                    if valid_idx.size > 1:
                        x[spike_idx] = np.interp(
                            spike_idx,
                            valid_idx,
                            x[valid_idx],
                        )
                    data[ep, ch] = x

        if orig_shape == (n_ch, n_times):
            return data[0]

        return data

    # ==========================================================
    # 切分 Epoch
    # ==========================================================
    def extract_epochs(
        self,
        raw: mne.io.BaseRaw,
        epoch_length: float = 2.0,
        smooth_spike: bool = True,
    ) -> np.ndarray:

        events = mne.make_fixed_length_events(raw, duration=epoch_length)

        epochs = mne.Epochs(
            raw,
            events,
            tmin=0,
            tmax=epoch_length,
            baseline=None,
            preload=True,
        )

        data = epochs.get_data()

        if smooth_spike:
            data = self._smooth_spikes(data)

        return data

    # ==========================================================
    # 源定位
    # ==========================================================
    def compute_source_localization(
        self,
        raw: mne.io.BaseRaw,
        fwd_file: str,
        noise_cov_file: Optional[str] = None,
        method: str = "dSPM",
    ):

        fwd = mne.read_forward_solution(fwd_file)

        if noise_cov_file is None:
            noise_cov = mne.compute_raw_covariance(raw)
        else:
            noise_cov = mne.read_cov(noise_cov_file)

        inv = mne.minimum_norm.make_inverse_operator(
            raw.info,
            fwd,
            noise_cov,
        )

        stc = mne.minimum_norm.apply_inverse_raw(
            raw,
            inv,
            lambda2=1.0 / 9.0,
            method=method,
        )

        return stc
