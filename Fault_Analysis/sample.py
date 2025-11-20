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
        Rolling-window size for time-domain and frequency-domain features.

    Returns
    -------
    pandas.DataFrame
        DataFrame with engineered features.
    """

    df = df.copy()  # Make a copy to avoid modifying the original dataframe

    # Define voltage and current column groups
    volt_cols = ['Va', 'Vb', 'Vc']  # Phase voltages
    curr_cols = ['Ia', 'Ib', 'Ic']  # Phase currents
    all_cols = volt_cols + curr_cols  # All waveform measurement columns

    # -----------------------------
    # 1. TIME-DOMAIN FEATURES
    # -----------------------------
    
    # Rolling RMS: measures effective power level / wave magnitude
    for col in all_cols:
        df[f'{col}_rms'] = df[col].rolling(window).apply(
            lambda x: np.sqrt(np.mean(x**2)), raw=True
        )

    # Rolling Standard Deviation: measures waveform fluctuation and instability
    for col in all_cols:
        df[f'{col}_std'] = df[col].rolling(window).std()

    # Rolling Peak-to-Peak: max - min value within the window (detects spikes)
    for col in all_cols:
        df[f'{col}_ptp'] = df[col].rolling(window).apply(
            lambda x: np.ptp(x), raw=True
        )

    # Rolling Mean: local average magnitude of the signal
    for col in all_cols:
        df[f'{col}_mean'] = df[col].rolling(window).mean()

    # -----------------------------
    # 2. FREQUENCY-DOMAIN FEATURES
    # -----------------------------

    # Function to compute dominant frequency from FFT of a window
    def fft_dominant_freq(arr, sampling_rate=1000):
        """
        Computes dominant frequency from FFT magnitude spectrum.
        arr: window of waveform values
        sampling_rate: assumed sampling rate (Hz)
        """
        yf = np.abs(np.fft.rfft(arr))  # FFT magnitudes
        xf = np.fft.rfftfreq(len(arr), d=1/sampling_rate)  # Frequency bins
        
        # Return the frequency corresponding to highest FFT amplitude
        return xf[np.argmax(yf)] if len(arr) > 0 else np.nan

    # Function to compute total spectral energy
    def fft_energy(arr):
        """
        Computes total FFT energy (sum of squared magnitudes).
        Indicates harmonic content and disturbance level.
        """
        yf = np.abs(np.fft.rfft(arr))
        return np.sum(yf**2)

    # Apply FFT-based features using rolling windows
    for col in all_cols:
        # Dominant frequency in each window
        df[f'{col}_dom_freq'] = df[col].rolling(window).apply(
            lambda x: fft_dominant_freq(x), raw=False
        )

        # Total frequency-domain energy of the window
        df[f'{col}_fft_energy'] = df[col].rolling(window).apply(
            lambda x: fft_energy(x), raw=False
        )

    # -----------------------------
    # 3. PHASE RELATIONSHIP FEATURES
    # -----------------------------

    # Voltage imbalance: measures deviation among phase voltages
    df['voltage_imbalance'] = df[volt_cols].std(axis=1) / df[volt_cols].mean(axis=1)

    # Current imbalance: deviation among phase currents
    df['current_imbalance'] = df[curr_cols].std(axis=1) / df[curr_cols].mean(axis=1)

    # Phase-to-phase voltage differences (line-line voltages)
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

    # Total real power: sum of per-phase instantaneous power
    df['P_total'] = (
        df['Va']*df['Ia'] +
        df['Vb']*df['Ib'] +
        df['Vc']*df['Ic']
    )

    # Apparent power magnitude: sqrt(V^2 sum * I^2 sum)
    df['S_mag'] = np.sqrt(
        (df['Va']**2 + df['Vb']**2 + df['Vc']**2) *
        (df['Ia']**2 + df['Ib']**2 + df['Ic']**2)
    )

    # Power factor: ratio of real to apparent power
    df['power_factor'] = df['P_total'] / (df['S_mag'] + 1e-9)  # avoid divide-by-zero

    # Drop initial rows that don't have enough rolling-window history
    df = df.dropna().reset_index(drop=True)

    return df




