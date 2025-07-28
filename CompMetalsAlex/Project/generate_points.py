import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, norm

def generate_points(samples_main, samples_tail, in_file: str, a_fit: float, b_fit: float, out_file):   
    
    ''' Helper Function Definitions '''

    def sample_Sij(n_samples):
        #Draw Sij values from the KDE.
        out = []
        batch = n_samples            # draw in batches of ≈desired size
        while len(out) < n_samples:
            s = kde_S.resample(batch)[0]           # 1‑D NumPy array
            valid = s[(s > 0) & (s < 0.32)]        # keeps ONLY positives
            out.extend(valid.tolist())             # ***append only valid***
            batch = n_samples - len(out)           # how many still needed?
        return np.array(out[:n_samples])

    def sigma_for_S(S):
        idx = np.digitize(S, bins) - 1
        idx = np.clip(idx, 0, len(bins)-2)  # ensure 0…9
        return sigma_by_bin.values[idx]

    def generate_synthetic(n_samples, a_fit, b_fit):
        S_syn = sample_Sij(n_samples) #A list of random samples
        # Predicted Rij from log fit
        R_mu  = log_model(S_syn, a_fit, b_fit)
        # Add Gaussian jitter with S-dependent σ
        sig   = sigma_for_S(S_syn)
        R_syn = R_mu + norm.rvs(scale=sig)
        return pd.DataFrame({"Sij": S_syn, "Rij": R_syn})

    def bootstrap_tail(n_samples, eps_sigma=0.1):
        tail = df_pairs[df_pairs["Sij"] > 0.10]
        if tail.empty:
            return pd.DataFrame(columns=["Sij", "Rij"])
        idx  = np.random.choice(tail.index, size=n_samples, replace=True)
        boot = tail.loc[idx].copy()
        # Add small noise so duplicates aren’t identical
        boot["Sij"] += norm.rvs(scale=eps_sigma*0.01, size=n_samples)
        boot["Rij"] += norm.rvs(scale=eps_sigma*0.1,  size=n_samples)
        # Keep within physical bounds
        #boot["Sij"] = boot["Sij"].clip(1e-6, 0.30)
        # boot["Rij"] = boot["Rij"].clip(2.0, 21.0)
        return boot



    def sample_tail_kde(n_samples):
        """Smoothly sample (Sij,Rij) in the outlier region via 2-D KDE."""
        S_tail, R_tail = kde_tail.resample(n_samples)
        # Enforce physical limits
        mask = (S_tail > 0) & (S_tail < 0.30) & (R_tail > 2.0) & (R_tail < 25.0)
        return pd.DataFrame({"Sij": S_tail[mask], "Rij": R_tail[mask]}).reset_index(drop=True)

    '''End of helper function definitions'''

    df_pairs = pd.read_csv(in_file)   # DataFrame with columns "Sij" and "Rij"

    #Find variance with log line
    def log_model(S, a, b):          # helper
        return a * (-np.log(S)) + b

    #Calculate log fit on current Sij values in database, and residuals
    df_pairs["R_pred"] = log_model(df_pairs["Sij"], a_fit, b_fit)
    df_pairs["resid"]  = df_pairs["Rij"] - df_pairs["R_pred"]

    # Estimate σ of residuals as a function of Sij (optional: bin-wise)
    bins = np.linspace(0.0, 0.30, 11)     # 0.03-wide bins
    df_pairs["S_bin"] = pd.cut(df_pairs["Sij"], bins, labels=False)

    #sigma_by_bin = df_pairs.groupby("S_bin")["resid"].std()   # NaNs in empty bins
    #sigma_global = df_pairs["resid"].std()                    # fallback value

    sigma_by_bin = (
    df_pairs.groupby("S_bin")["resid"].std()
    .reindex(range(len(bins)-1))           # ensure full length 10
    .interpolate(method="linear", limit_direction="both")
    .fillna(method="bfill")                # just in case
    )

    print(sigma_by_bin)
    # for i, sigma in sigma_by_bin.items():
    #     bin_low  = bins[int(i)]
    #     bin_high = bins[int(i) + 1]
    #     print(f"Sij ∈ [{bin_low:.2f}, {bin_high:.2f}) → stddev_resid ≈ {sigma:.10f} Å")

    kde_S = gaussian_kde(df_pairs["Sij"].values, bw_method="scott")

    tail_real = df_pairs[df_pairs["Sij"] > 0.1][["Sij", "Rij"]].values.T
    kde_tail  = gaussian_kde(tail_real, bw_method="silverman")

    main_syn = generate_synthetic(samples_main, a_fit, b_fit)
    #tail_syn = bootstrap_tail(samples_tail)
    tail_syn = sample_tail_kde(samples_tail)
    EPS_S = 2e-4      # jitter for Sij  (≈0.0002)
    EPS_R = 1e-2      # jitter for Rij  (≈0.01 Å)

    tail_syn["Sij"] += np.random.uniform(-EPS_S, EPS_S,size=len(tail_syn))
    tail_syn["Rij"] += np.random.uniform(-EPS_R, EPS_R,size=len(tail_syn))

    df_synthetic = pd.concat([main_syn, tail_syn], ignore_index=True)

    

    df_synthetic.to_csv(out_file, index = False)

    
    



if __name__ == "__main__":
    #generate_points(70000, 750, "/mnt/c/c++_tests/games_work/ml_work/MLModel/SymbolicReg/data/test1/all_damped_data.csv", 1.0540, 1.7658,"/mnt/c/c++_tests/games_work/ml_work/synth_data/70kplus750on90k_01.csv")
    # df = pd.read_csv("/mnt/c/c++_tests/games_work/ml_work/synth_data/70kplus750on90k_01.csv")
    # df_filtered = df.drop(df[df['Rij'] < 2].index)
    # df_filtered.to_csv("/mnt/c/c++_tests/games_work/ml_work/synth_data/70kplus750on90k_filtered01.csv", index=False)
