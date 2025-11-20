import numpy as np
import pandas as pd

def feature_engineer(df, window=100):
    """
    Feature engineering for waveform-based fault classification.
    Generates time-domain, frequency-domain, and phase-relationship features
    using only Va, Vb, Vc, Ia, Ib, Ic.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: Va, Vb, Vc, Ia, Ib, Ic.
    window : int
        Rolling-window size for time-domain features.

    Returns
    -------
    pandas.DataFrame
        DataFrame with engineered features.
    """

    df = df.copy()

    volt_cols = ['Va', 'Vb', 'Vc']
    curr_cols = ['Ia', 'Ib', 'Ic']
    all_cols = volt_cols + curr_cols

    # -----------------------------
    # 1. TIME-DOMAIN FEATURES
    # -----------------------------
    # Rolling RMS
    for col in all_cols:
        df[f'{col}_rms'] = df[col].rolling(window).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)

    # Rolling Standard Deviation
    for col in all_cols:
        df[f'{col}_std'] = df[col].rolling(window).std()

    # Rolling Peak-to-Peak
    for col in all_cols:
        df[f'{col}_ptp'] = df[col].rolling(window).apply(lambda x: np.ptp(x), raw=True)

    # Rolling Mean
    for col in all_cols:
        df[f'{col}_mean'] = df[col].rolling(window).mean()

    # -----------------------------
    # 2. FREQUENCY-DOMAIN FEATURES
    # -----------------------------
    def fft_dominant_freq(arr, sampling_rate=1000):
        """Return dominant frequency magnitude."""
        yf = np.abs(np.fft.rfft(arr))
        xf = np.fft.rfftfreq(len(arr), d=1/sampling_rate)
        return xf[np.argmax(yf)] if len(arr) > 0 else np.nan

    def fft_energy(arr):
        """Total signal energy in frequency domain."""
        yf = np.abs(np.fft.rfft(arr))
        return np.sum(yf**2)

    for col in all_cols:
        df[f'{col}_dom_freq'] = df[col].rolling(window).apply(
            lambda x: fft_dominant_freq(x), raw=False
        )
        df[f'{col}_fft_energy'] = df[col].rolling(window).apply(
            lambda x: fft_energy(x), raw=False
        )

    # -----------------------------
    # 3. PHASE RELATIONSHIP FEATURES
    # -----------------------------
    # Voltage imbalance (IEEE definition)
    df['voltage_imbalance'] = df[volt_cols].std(axis=1) / df[volt_cols].mean(axis=1)

    # Current imbalance
    df['current_imbalance'] = df[curr_cols].std(axis=1) / df[curr_cols].mean(axis=1)

    # Phase-to-phase voltage differences
    df['Vab'] = df['Va'] - df['Vb']
    df['Vbc'] = df['Vb'] - df['Vc']
    df['Vca'] = df['Vc'] - df['Va']

    # Phase-to-phase current differences
    df['Iab'] = df['Ia'] - df['Ib']
    df['Ibc'] = df['Ib'] - df['Ic']
    df['Ica'] = df['Ic'] - df['Ia']

    # -----------------------------
    # 4. POWER FEATURES
    # -----------------------------
    df['P_total'] = df['Va']*df['Ia'] + df['Vb']*df['Ib'] + df['Vc']*df['Ic']
    df['S_mag'] = np.sqrt((df['Va']**2 + df['Vb']**2 + df['Vc']**2) *
                          (df['Ia']**2 + df['Ib']**2 + df['Ic']**2))

    df['power_factor'] = df['P_total'] / (df['S_mag'] + 1e-9)

    # Drop rows without enough rolling history
    df = df.dropna().reset_index(drop=True)

    return df
