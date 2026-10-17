import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pathlib
import jax
import io, contextlib
import jax.numpy as jnp
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.stats.stattools import jarque_bera
from scipy.stats import shapiro, anderson, levene, bartlett
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, het_breuschpagan
from statsmodels.tsa.stattools import adfuller

jax.config.update("jax_enable_x64", True)

from statsmodels.tools.numdiff import approx_fprime, approx_hess
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import minimize, Bounds
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

pd.set_option("display.max_columns", None)     
pd.set_option("display.width",        None)    
pd.set_option("display.max_colwidth", None) 


def model_computation(company, ema_spans, analisys, model = None, split_date = None, group = None): # stock data, company name, ema_span, analisys (Y yes, Y_P yes with plots), model (garch, garch+sentiment), date to split train data

    COMPANY_ISIN = {
    "Apple":     "US0378331005",
    "Exxon":     "US30231G1022",
    "JPM":       "US46625H1005",  
    "PFE":       "US7170811035",
    }

    if company == "Apple":
        df = pd.read_csv("AAPL_historical_data.csv")
    elif company == "Exxon":
        df = pd.read_csv("XOM_historical_data.csv")
    elif company == "JPM":
        df = pd.read_csv("JPM_historical_data.csv")
    elif company == "PFE":
        df = pd.read_csv("PFE_historical_data.csv")

    sentiment = pd.read_csv("sentiment_volume_scores.csv")
    macro = pd.read_csv("macro_data.csv")

    sent_filtered = sentiment.loc[sentiment["AssetId"] == COMPANY_ISIN[company]].copy()

    df["date"] = pd.to_datetime(df["Date"])
    sent_filtered["date"] = pd.to_datetime(sent_filtered["NewsDate"])
    macro["date"] = pd.to_datetime(macro["date"])

    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna(subset=["Log_Returns"]).copy()

    mean_ret = df["Log_Returns"].mean()
    df["residuals"] = (df["Log_Returns"] - mean_ret) * 100

# --------------------------------- Functions to obtain volatility -------------------------------------------------------------------    
    def realized_volatility(window, kernel="triangular", bandwidth=None):
        x = np.asarray(window, dtype=float)
        x = x[~np.isnan(x)]                      
        n = x.size
        if n == 0:
            return 0.0                        
        if n == 1:
            return abs(x[0])                   

        centre = (n - 1) / 2.0
        d = np.abs(np.arange(n) - centre)

        if kernel == "triangular":
            dmax = d.max()
           
            w = 1.0 - d / (dmax if dmax != 0 else 1.0)
        elif kernel == "gaussian":
            if bandwidth is None:
                bandwidth = n / 6.0
            w = np.exp(-(d**2) / (2.0 * bandwidth**2))
        else:
            raise ValueError("kernel must be 'triangular' or 'gaussian'")

        w_sum = w.sum()
        if not np.isfinite(w_sum) or w_sum == 0:
            return 0.0

        mu_w  = np.dot(w, x) / w_sum
        var_w = np.dot(w, (x - mu_w)**2) / w_sum
        return float(np.sqrt(max(var_w, 0.0)))

    def real_volatility_window(returns, window_size, kernel="triangular", bandwidth=None):
        returns = np.asarray(returns) * 100
        T = len(returns)
        vols = np.empty(T)

        for i in range(T):
            start = max(0, i - (window_size - 1))  
            end   = i + 1
            window = returns[start:end]
            vols[i] = realized_volatility(window, kernel=kernel, bandwidth=bandwidth)
        return vols

    realized_vol = real_volatility_window(df['Log_Returns'], 7)

    df['realized_vol'] = realized_vol   

    # Merge of datasets by date
    merged = (
        df
        .merge(sent_filtered, on="date", how="inner")
        .merge(macro,         on="date", how="inner")
        .drop(columns=["Date", "NewsDate"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Group by aggregation subtype and doc aggregation. Can call it with merge_df[("count", "Mean")]

    merged_df = {
    (docagg, subtype): grp.copy()
    for (docagg, subtype), grp in merged.groupby(
        ["DocAgg", "AggregationSubtype"], sort=False
        )
    }

    for key, df_slice in merged_df.items():
        
        for span in ema_spans:
            finb_col = f"PolarityFinb_EMA_{span}"
            rob_col  = f"PolarityRob_EMA_{span}"
            
            df_slice[finb_col] = (
                df_slice["PolarityFinb"]
                .ewm(span=span, adjust=False, min_periods=1).mean()
            )
            df_slice[rob_col] = (
                df_slice["PolarityRob"]
                .ewm(span=span, adjust=False, min_periods=1).mean()
            )

        merged_df[key] = df_slice

    
     
# --------------------------------- Function for analisys -------------------------------------------------------------------
    def docagg_analysis(merged_df):

        columns_corr = [
        "residuals",
        "PolarityFinb",
        "PolarityRob",
        "RCV",
        "RVT",
        "Volume_y",
        "realized_vol",
        "EPU"
        ]
        
        columns_stats = [
        "PolarityFinb",
        "PolarityRob",
        ]
        
        # Add ema columns
        for span in ema_spans:
            columns_corr.append(f"PolarityFinb_EMA_{span}")
            columns_corr.append(f"PolarityRob_EMA_{span}")

            columns_stats.append(f"PolarityFinb_EMA_{span}")
            columns_stats.append(f"PolarityRob_EMA_{span}")
        
       

        stats = ["mean", "std", "min", "max", "25%", "50%", "75%"]

        # Correlation
        corr_matrices = {
            key: df_slice[columns_corr].corr()
            for key, df_slice in merged_df.items()
            }

        # Stats
        frames = []
        for (docagg, subtype), df_slice in merged_df.items():
            df_num = df_slice[columns_stats].apply(pd.to_numeric, errors='coerce')
            desc = df_num.describe().T.loc[:, stats].reset_index()
            desc = desc.rename(columns={'index': 'Polarity type'})
            desc.insert(0, 'DocAgg', docagg)
            desc.insert(1, 'AggregationSubtype', subtype)
            desc['Polarity type'] = pd.Categorical(desc['Polarity type'],
                                                categories=columns_stats, ordered=True)
            desc = desc.sort_values('Polarity type')
            frames.append(desc)

        comparison_df = (pd.concat(frames, ignore_index=True)
                        .sort_values(["DocAgg", "AggregationSubtype", "Polarity type"])
                        .reset_index(drop=True))

        corr_dir = Path(f"analisys_{company}")
        corr_dir.mkdir(exist_ok=True)

        for (docagg, subtype), mat in corr_matrices.items():
            mat.to_csv(corr_dir / f"corr_{docagg}_{subtype}.csv", index=True)

        comparison_df.to_csv(corr_dir / "comparison_df.csv", index=False)

        return corr_matrices, comparison_df
# --------------------------------- Plots -------------------------------------------------------------------    
    def plot_vol_correlations(grouped_dict,
                            features=None,
                            save_dir=f"plots/{company}/corr_bars"):
       
        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for key, df_slice in grouped_dict.items():
            docagg, subtype = key
            if features is None:
                features = [c for c in df_slice.select_dtypes("number").columns
                            if c != "realized_vol"]

            # Pearson r for each feature vs the target
            corr = df_slice[features + ["realized_vol"]].corr()["realized_vol"]\
                                                        .drop("realized_vol")


            plt.figure(figsize=(8, 4))
            corr.sort_values().plot(kind="barh")
            plt.title(f"Correlation with realized_vol\n{docagg} – {subtype}")
            plt.xlabel("Pearson r")
            plt.tight_layout()

            fname = save_dir / f"corr_{docagg}_{subtype}.png"
            plt.savefig(fname, dpi=150)
            plt.close()
            print(f"Wrote  {fname}")

    # SCATTER PLOTS
    def scatter_vol_vs_features(grouped_dict,
                                features=None,
                                save_dir=f"plots/{company}/scatter"):

        save_dir = pathlib.Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for key, df_slice in grouped_dict.items():
            docagg, subtype = key
            if features is None:
                features = [c for c in df_slice.select_dtypes("number").columns
                            if c != "realized_vol"]

            for feat in features:
                plt.figure(figsize=(5, 4))
                plt.scatter(df_slice[feat], df_slice["realized_vol"], alpha=0.4, s=10)
                plt.xlabel(feat)
                plt.ylabel("realized_vol")
                plt.title(f"{docagg} – {subtype}")
                plt.tight_layout()

                fname = save_dir / f"scatter_{feat}_{docagg}_{subtype}.png"
                plt.savefig(fname, dpi=150)
                plt.close()
                print(f"Wrote  {fname}")
# ----------------------------------------------------------------------------------------------------    
    if analisys == "Y_P":
    
        plot_vol_correlations(merged_df)
        scatter_vol_vs_features(merged_df)
        corr_matrices, comparison_df = docagg_analysis(merged_df)
        return corr_matrices, comparison_df

    elif analisys == "Y":
        corr_matrices, comparison_df = docagg_analysis(merged_df)

        return corr_matrices, comparison_df
    else: # Run the models if the analisys is not chosen
        param_map = {
            "garch":                                ["omega", "alpha", "beta"],
            "garch+sentiment":                      ["omega", "alpha", "beta", "kappa", "gamma"],
            "garch+attention":                      ["omega", "alpha", "beta", "tau"],
            "garch+macro":                          ["omega", "alpha", "beta", "ep1"],
            "garch+sentiment+macro":                ["omega", "alpha", "beta", "kappa", "gamma","ep1"],
            "garch+attention+macro":                ["omega", "alpha", "beta", "tau", "ep1"],
            "garch+sentiment+attention":            ["omega","alpha","beta","kappa","gamma","tau"],
            "garch+sentiment+attention+macro":      ["omega", "alpha", "beta", "kappa", "gamma","tau","ep1"]

        }

        all_names = ["omega","alpha","beta","kappa","gamma","tau","ep1"]

        def active_names(model):
            return param_map[model.lower()]
        
        
        def to_full_vector(p_small, model):
            """Expand p_small (k×1) into an 8×1 full vector."""
            full = np.zeros(8)
            for name, val in zip(active_names(model), p_small):
                full[all_names.index(name)] = val
            return full
        
        
        
        def statistical_tests(white_noise, name_prop, polarity):
            buff = io.StringIO()                   # <-- new
            with contextlib.redirect_stdout(buff):
                 
                print(f"== {name_prop}+{polarity} ==")
        
                print("== Normality Tests ==")

                # 1) Jarque-Bera Test
                jb_stat, jb_pvalue, _, _ = jarque_bera(white_noise)
                print(f"Jarque-Bera test: statistic={jb_stat:.4f}, p-value={jb_pvalue:.4g} "
                    f"(fail to reject normality if p>0.05)")

                # 2) Shapiro-Wilk Test
                shapiro_stat, shapiro_p = shapiro(white_noise)
                print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4g} "
                    f"(fail to reject normality if p>0.05)")

                # 3) Anderson-Darling Test
                ad_result = anderson(white_noise)
                print(f"Anderson-Darling test: statistic={ad_result.statistic:.4f}")
                for i, sl in enumerate(ad_result.significance_level):
                    cv = ad_result.critical_values[i]
                    print(f"  Significance level {sl}%, Critical Value {cv:.4f}, "
                        f"{'Fail to reject' if ad_result.statistic < cv else 'Reject'} normality")

                print("\n== Autocorrelation Test ==")

                # 4) Ljung-Box test
                ljung_result = acorr_ljungbox(white_noise, lags=[20], return_df=True)
                print("Ljung-Box test (lag=10):")
                print(ljung_result, "p-value > 0.05 => no autocorrelation")
                

                print("\n== Stationarity Test ==")

                # 5) Augmented Dickey-Fuller
                adf_stat, adf_pval, _, _, _, _ = adfuller(white_noise)
                print(f"ADF test: statistic={adf_stat:.4f}, p-value={adf_pval:.4g} "
                    f"(stationary if p<0.05)")

                print("\n== Heteroscedasticity Tests ==")

                # 6) Engle’s ARCH test
                arch_stat, arch_pval, _, _ = het_arch(white_noise)
                print(f"Engle’s ARCH test: statistic={arch_stat:.4f}, p-value={arch_pval:.4g} "
                    f"(constant variance if p>0.05)")

                # 7) Breusch-Pagan test
                import statsmodels.api as sm
                X = sm.add_constant(np.arange(len(white_noise)))
                bp_stat, bp_pval, _, _ = het_breuschpagan(white_noise, X)
                print(f"Breusch-Pagan test: statistic={bp_stat:.4f}, p-value={bp_pval:.4g} "
                    f"(constant variance if p>0.05)")

                print("\n== Homoscedasticity Across Groups (Levene / Bartlett) ==")
                
                # Test variance across groups
                group1 = white_noise[0:len(white_noise)//3]
                group2 = white_noise[len(white_noise)//3:2*len(white_noise)//3]
                group3 = white_noise[2*len(white_noise)//3:]

                # 8) Levene Test, (does not assume normality)
                lev_stat, lev_pval = levene(group1, group2, group3)
                print(f"Levene test: statistic={lev_stat:.4f}, p-value={lev_pval:.4g} "
                    f"(constant variance across groups if p>0.05)")

                # 9) Bartlett Test (assumes normality)
                bart_stat, bart_pval = bartlett(group1, group2, group3)
                print(f"Bartlett test: statistic={bart_stat:.4f}, p-value={bart_pval:.4g} "
                    f"(constant variance across groups if p>0.05)")
            return buff.getvalue()
        
#-----------------------------------------------

        def garch_negloglike(params, model, residuals, polarity = None, volume = None, var = None, mu_sigma = None):

            T = residuals.shape[0]
            
            if model == 'garch':
                omega, alpha, beta = params
                inputs = residuals[:-1]

            elif model == 'garch+sentiment':
                omega, alpha, beta, kappa, gamma = params

                inputs = (residuals[:-1], polarity[:-1])

            elif model == 'garch+attention':
                omega, alpha, beta, tau = params

                volume       = jnp.asarray(volume,       dtype=jnp.float32)

                growth = 100.0 * (volume[1:] - volume[:-1]) / volume[:-1]

                growth = np.concatenate(([0.0], growth))

                growth = growth / np.nanmax(np.abs(growth))
                growth = growth.astype(np.float32)

                inputs = (residuals[:-1], growth[:-1])
            
            elif model == 'garch+macro':
                omega, alpha, beta, ep1 = params

                epu = var

                inputs = (residuals[:-1], epu[:-1])
            
            elif model == 'garch+sentiment+attention':
                omega, alpha, beta, kappa, gamma, tau = params

                volume       = jnp.asarray(volume,       dtype=jnp.float32)

                growth = 100.0 * (volume[1:] - volume[:-1]) / volume[:-1]

                growth = np.concatenate(([0.0], growth))

                growth = growth / np.nanmax(np.abs(growth))
                growth = growth.astype(np.float32)

                inputs = (residuals[:-1], polarity[:-1],growth[:-1])
            
            elif model == 'garch+sentiment+macro':
                omega, alpha, beta, kappa, gamma, ep1 = params

                epu = var

                inputs = (residuals[:-1], polarity[:-1], epu[:-1])
            
            elif model == 'garch+attention+macro':
                omega, alpha, beta, tau, ep1 = params
                
                volume       = jnp.asarray(volume,       dtype=jnp.float32)

                growth = 100.0 * (volume[1:] - volume[:-1]) / volume[:-1]

                growth = np.concatenate(([0.0], growth))

                growth = growth / np.nanmax(np.abs(growth))
                growth = growth.astype(np.float32)

                epu = var

                inputs = (residuals[:-1],growth[:-1], epu[:-1])

            elif model == 'garch+sentiment+attention+macro':
                omega, alpha, beta, kappa, gamma, tau, ep1 = params

                volume       = jnp.asarray(volume,       dtype=jnp.float32)

                growth = 100.0 * (volume[1:] - volume[:-1]) / volume[:-1]

                growth = np.concatenate(([0.0], growth))

                growth = growth / np.nanmax(np.abs(growth))
                growth = growth.astype(np.float32)

                epu = var

                inputs = (residuals[:-1], polarity[:-1],growth[:-1], epu[:-1])

            sigma0 = jnp.var(residuals[0:20])

            # Define garch ecuation at time t
            def garch_update(sigma_tm1, inputs):

                if model == 'garch':
                    eps_tm1 = inputs
                    sigma_t = omega + alpha*(eps_tm1**2) + beta*sigma_tm1
                    
                elif model == 'garch+sentiment':
                    eps_tm1, pol_tm1 = inputs
                    delta = (pol_tm1 < 0).astype(float) # if true delta = 1

                    f_tm1  = gamma * (1 - delta) * pol_tm1**2 + kappa * delta * pol_tm1**2
                    sigma_t = omega + alpha*(eps_tm1**2) + beta*sigma_tm1 + f_tm1

                elif model == 'garch+attention':
                    eps_tm1, growth = inputs
                    #d = (growth > 0).astype(float)
                    d = (sigma_tm1 > mu_sigma).astype(float)

                    sigma_t = omega + alpha*(eps_tm1**2) + beta*sigma_tm1+tau*d*growth
                
                elif model == 'garch+macro':
                    eps_tm1, epu = inputs

                    om = (epu < 0).astype(float)
                    mac_tm1 = ep1 * (1 - om) * epu**2
                    sigma_t = omega + alpha*(eps_tm1**2) + beta*sigma_tm1 + mac_tm1
                
                elif model == 'garch+sentiment+attention':
                    eps_tm1, pol_tm1, growth = inputs
                    #d = (growth > 0).astype(float)
                    d = (sigma_tm1 > mu_sigma).astype(float)

                    delta = (pol_tm1 < 0).astype(float)

                    f_tm1  = gamma * (1 - delta) * pol_tm1**2 + kappa * delta * pol_tm1**2
                    sigma_t = omega + alpha*(eps_tm1**2) + beta*sigma_tm1+tau*d*growth + f_tm1
                
                elif model == 'garch+sentiment+macro':
                    eps_tm1, pol_tm1, epu = inputs

                    om = (epu < 0).astype(float)
                    mac_tm1 = ep1 * (1 - om) * epu**2

                    delta = (pol_tm1 < 0).astype(float)

                    f_tm1  = gamma * (1 - delta) * pol_tm1**2 + kappa * delta * pol_tm1**2
                    sigma_t = omega + alpha*(eps_tm1**2) + beta*sigma_tm1 + f_tm1 + mac_tm1
                
                elif model == 'garch+attention+macro':
                    eps_tm1, growth, epu = inputs
                    #d = (growth > 0).astype(float)
                    d = (sigma_tm1 > mu_sigma).astype(float)

                    om = (epu < 0).astype(float)
                    mac_tm1 = ep1 * (1 - om) * epu**2
                    sigma_t = omega + alpha*(eps_tm1**2) + beta*sigma_tm1+tau*d*growth + mac_tm1

                elif model == 'garch+sentiment+attention+macro':
                    eps_tm1, pol_tm1, growth, epu = inputs
                    d = (sigma_tm1 > mu_sigma).astype(float)

                    om = (epu < 0).astype(float)
                    mac_tm1 = ep1 * (1 - om) * epu**2

                    delta = (pol_tm1 < 0).astype(float)

                    f_tm1  = gamma * (1 - delta) * pol_tm1**2 + kappa * delta * pol_tm1**2
                    sigma_t = omega + alpha*(eps_tm1**2) + beta*sigma_tm1+tau*d*growth + f_tm1 + mac_tm1

                sigma_t = jnp.maximum(sigma_t, 1e-8)
                return sigma_t, sigma_t # One value is the input for next iteration, the other is the output

            _, sigma2_array = jax.lax.scan(garch_update, sigma0, inputs)
            sigma2_full = jnp.concatenate((jnp.array([sigma0]), sigma2_array))

            loglike = -0.5 * (
                T*jnp.log(2*jnp.pi)
                + jnp.sum(jnp.log(sigma2_full))
                + jnp.sum(residuals**2 / sigma2_full)
            )
            return -loglike


        def garch_negloglike_per_obs(params, model, residuals, polarity = None, volume = None, var = None, mu_sigma = None):

            T = residuals.shape[0]
            
            if model == 'garch':
                omega, alpha, beta = params
                inputs = residuals[:-1]

            elif model == 'garch+sentiment':
                omega, alpha, beta, kappa, gamma = params

                inputs = (residuals[:-1], polarity[:-1])

            elif model == 'garch+attention':
                omega, alpha, beta, tau = params

                volume       = jnp.asarray(volume,       dtype=jnp.float32)

                growth = 100.0 * (volume[1:] - volume[:-1]) / volume[:-1]

                growth = np.concatenate(([0.0], growth))

                growth = growth / np.nanmax(np.abs(growth))
                growth = growth.astype(np.float32)

                inputs = (residuals[:-1], growth[:-1])
            
            elif model == 'garch+macro':
                omega, alpha, beta, ep1 = params

                epu = var

                inputs = (residuals[:-1], epu[:-1])
            
            elif model == 'garch+sentiment+attention':
                omega, alpha, beta, kappa, gamma, tau = params

                volume       = jnp.asarray(volume,       dtype=jnp.float32)

                growth = 100.0 * (volume[1:] - volume[:-1]) / volume[:-1]

                growth = np.concatenate(([0.0], growth))

                growth = growth / np.nanmax(np.abs(growth))
                growth = growth.astype(np.float32)

                inputs = (residuals[:-1], polarity[:-1],growth[:-1])
            
            elif model == 'garch+sentiment+macro':
                omega, alpha, beta, kappa, gamma, ep1 = params

                epu = var

                inputs = (residuals[:-1], polarity[:-1], epu[:-1])
            
            elif model == 'garch+attention+macro':
                omega, alpha, beta, tau, ep1 = params
                
                volume       = jnp.asarray(volume,       dtype=jnp.float32)

                growth = 100.0 * (volume[1:] - volume[:-1]) / volume[:-1]

                growth = np.concatenate(([0.0], growth))

                growth = growth / np.nanmax(np.abs(growth))
                growth = growth.astype(np.float32)

                epu = var

                inputs = (residuals[:-1],growth[:-1], epu[:-1])

            elif model == 'garch+sentiment+attention+macro':
                omega, alpha, beta, kappa, gamma, tau, ep1 = params

                volume       = jnp.asarray(volume,       dtype=jnp.float32)

                growth = 100.0 * (volume[1:] - volume[:-1]) / volume[:-1]

                growth = np.concatenate(([0.0], growth))

                growth = growth / np.nanmax(np.abs(growth))
                growth = growth.astype(np.float32)

                epu = var

                inputs = (residuals[:-1], polarity[:-1],growth[:-1], epu[:-1])

            sigma0 = jnp.var(residuals[0:20])

            # Define garch ecuation at time t
            def garch_update(sigma_tm1, inputs):

                if model == 'garch':
                    eps_tm1 = inputs
                    sigma_t = omega + alpha*(eps_tm1**2) + beta*sigma_tm1
                    
                elif model == 'garch+sentiment':
                    eps_tm1, pol_tm1 = inputs
                    delta = (pol_tm1 < 0).astype(float)

                    f_tm1  = gamma * (1 - delta) * pol_tm1**2 + kappa * delta * pol_tm1**2
                    sigma_t = omega + alpha*(eps_tm1**2) + beta*sigma_tm1 + f_tm1

                elif model == 'garch+attention':
                    eps_tm1, growth = inputs
                    d = (sigma_tm1 > mu_sigma).astype(float)

                    sigma_t = omega + alpha*(eps_tm1**2) + beta*sigma_tm1+tau*d*growth
                
                elif model == 'garch+macro':
                    eps_tm1, epu = inputs

                    om = (epu < 0).astype(float)
                    mac_tm1 = ep1 * (1 - om) * epu**2
                    sigma_t = omega + alpha*(eps_tm1**2) + beta*sigma_tm1 + mac_tm1
                
                elif model == 'garch+sentiment+attention':
                    eps_tm1, pol_tm1, growth = inputs
                    d = (sigma_tm1 > mu_sigma).astype(float)

                    delta = (pol_tm1 < 0).astype(float)

                    f_tm1  = gamma * (1 - delta) * pol_tm1**2 + kappa * delta * pol_tm1**2
                    sigma_t = omega + alpha*(eps_tm1**2) + beta*sigma_tm1+tau*d*growth + f_tm1
                
                elif model == 'garch+sentiment+macro':
                    eps_tm1, pol_tm1, epu = inputs

                    om = (epu < 0).astype(float)
                    mac_tm1 = ep1 * (1 - om) * epu**2

                    delta = (pol_tm1 < 0).astype(float)

                    f_tm1  = gamma * (1 - delta) * pol_tm1**2 + kappa * delta * pol_tm1**2
                    sigma_t = omega + alpha*(eps_tm1**2) + beta*sigma_tm1 + f_tm1 + mac_tm1
                
                elif model == 'garch+attention+macro':
                    eps_tm1, growth, epu = inputs
                    d = (sigma_tm1 > mu_sigma).astype(float)

                    om = (epu < 0).astype(float)
                    mac_tm1 = ep1 * (1 - om) * epu**2
                    sigma_t = omega + alpha*(eps_tm1**2) + beta*sigma_tm1+tau*d*growth + mac_tm1

                elif model == 'garch+sentiment+attention+macro':
                    eps_tm1, pol_tm1, growth, epu = inputs
                    d = (sigma_tm1 > mu_sigma).astype(float)

                    om = (epu < 0).astype(float)
                    mac_tm1 = ep1 * (1 - om) * epu**2

                    delta = (pol_tm1 < 0).astype(float)

                    f_tm1  = gamma * (1 - delta) * pol_tm1**2 + kappa * delta * pol_tm1**2
                    sigma_t = omega + alpha*(eps_tm1**2) + beta*sigma_tm1+tau*d*growth + f_tm1 + mac_tm1

                sigma_t = jnp.maximum(sigma_t, 1e-8)
                return sigma_t, sigma_t # One value is the input for next iteration, the other is the output

            _, sigma2_array = jax.lax.scan(garch_update, sigma0, inputs)

            sigma2_full = jnp.concatenate((jnp.array([sigma0]), sigma2_array))

            # Per-observation negative log-likelihood:
            neg_ll_obs = 0.5 * (
                jnp.log(2 * jnp.pi) 
                + jnp.log(sigma2_full) 
                + residuals**2 / sigma2_full
            )

            return neg_ll_obs

        
        def fit_garch(params, model, residuals, polarity=None, volume=None, var=None, mu_sigma = None):
            
            names = active_names(model)          # e.g. ["omega","alpha","beta"]
            k     = len(names)
            idx   = [all_names.index(n) for n in names]

            lb = np.array([1e-8, 1e-8, 1e-8, -5, -5, 1e-8, -5],  dtype=float)[idx] # lower bound omega, alpha, beta, kappa, gamma, tau,ep1
            ub = np.array([1   , 1   , 1   , 100, 100, 100, 100], dtype=float)[idx] # upper bound omega, alpha, beta, kappa, gamma, tau, ep1
            bounds = Bounds(lb, ub)

            # JAX value-and-grad wrapper
            def nll(p):
                return garch_negloglike(p, model, residuals, polarity, volume, var, mu_sigma)

            nll_and_grad = jax.value_and_grad(nll)

            res = minimize(
                fun      = lambda p: nll_and_grad(p)[0],
                x0       = params,
                jac      = lambda p: np.array(nll_and_grad(p)[1]),
                method   = "L-BFGS-B",
                bounds   = bounds,
                tol      = 1e-15
            )
            return res


        def compute_param_cov(
            params,
            model,
            negloglike_fn,
            residuals,
            polarity = None,
            volume = None,
            var = None,
            mu_sigma = None,
            robust=True
        ):

            nobs = len(residuals)
            
            # 2) Approximate Hessian of the *negative* log-likelihood (summed, not per observation)
            def negloglike_summed(p):
                return negloglike_fn(p, model, residuals, polarity, volume, var, mu_sigma)
            
            H = approx_hess(params, negloglike_summed)
            
            # H is an average Hessian.
            H /= nobs
            
            k          = H.shape[0]
            eigvals    = np.linalg.eigvalsh(H)
            cond_num   = np.abs(eigvals[-1] / eigvals[0]) if eigvals[0] != 0 else np.inf

            print("\n=== Hessian diagnostics ===")
            print(f"model={model}  nobs={nobs}")
            print("shape :", H.shape)
            print("rank  :", np.linalg.matrix_rank(H), "/", k)
            print("eigs  :", np.round(eigvals, 6))
            print("cond# :", f"{cond_num:.2e}")
            print("params:", np.round(params, 6))

            # Invert the Hessian
            # inv_H = np.linalg.inv(H)

            try:
                inv_H = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                # fallback: Moore–Penrose pseudo-inverse
                print("⚠️  Hessian singular – using pseudo-inverse")
                inv_H = np.linalg.pinv(H)
            
            if robust:
                # 3) For robust, need the per-observation gradients of the negative log-likelihood
                def negloglike_per_obs(p):
                    return garch_negloglike_per_obs(p, model, residuals, polarity, volume, var, mu_sigma)
                
                # approx_fprime will produce an (nobs x n_params) array of scores
                scores = approx_fprime(params, negloglike_per_obs)
                # scores shape: (nobs, n_params)
                
                # 4) np.cov by default treats rows as variables, columns as observations, need to traspose
                score_cov = np.cov(scores.T, bias=False) # shape: (n_params, n_params)
                
                # Sandwich formula: inv_H * score_cov * inv_H
                cov = inv_H @ score_cov @ inv_H

                cov /= nobs
            else:
                # Classic: inverse Hessian scaled by 1/nobs
                cov = inv_H / nobs
            
            return cov
        
        def volatility_garch (params, model, residuals, polarity = None, volume = None, var = None, mu_sigma = None):

            T = residuals.shape[0]
            
            if model == 'garch':
                omega, alpha1, beta1 = params

            elif model == 'garch+sentiment':
                omega, alpha1, beta1, kappa, gamma = params

            elif model == 'garch+attention':
                omega, alpha1, beta1, tau = params

                volume       = jnp.asarray(volume,       dtype=jnp.float32)

                growth = 100.0 * (volume[1:] - volume[:-1]) / volume[:-1]

                growth = np.concatenate(([0.0], growth))

                growth = growth / np.nanmax(np.abs(growth))
                growth = growth.astype(np.float32)
            
            elif model == 'garch+macro':
                omega, alpha1, beta1, ep1 = params

                epu = var
            
            elif model == 'garch+sentiment+attention':
                omega, alpha1, beta1, kappa, gamma, tau = params

                volume       = jnp.asarray(volume,       dtype=jnp.float32)

                growth = 100.0 * (volume[1:] - volume[:-1]) / volume[:-1]

                growth = np.concatenate(([0.0], growth))

                growth = growth / np.nanmax(np.abs(growth))
                growth = growth.astype(np.float32)
            
            elif model == 'garch+sentiment+macro':
                omega, alpha1, beta1, kappa, gamma, ep1 = params

                epu = var
            
            elif model == 'garch+attention+macro':
                omega, alpha1, beta1, tau, ep1 = params
                
                volume       = jnp.asarray(volume,       dtype=jnp.float32)

                growth = 100.0 * (volume[1:] - volume[:-1]) / volume[:-1]

                growth = np.concatenate(([0.0], growth))

                growth = growth / np.nanmax(np.abs(growth))
                growth = growth.astype(np.float32)

                epu = var

            elif model == 'garch+sentiment+attention+macro':
                omega, alpha1, beta1, kappa, gamma, tau, ep1 = params

                volume       = jnp.asarray(volume,       dtype=jnp.float32)

                growth = 100.0 * (volume[1:] - volume[:-1]) / volume[:-1]

                growth = np.concatenate(([0.0], growth))

                growth = growth / np.nanmax(np.abs(growth))
                growth = growth.astype(np.float32)
                epu = var

            sigma2 = np.zeros(T)
            sigma2[0] = np.var(residuals[0:20])

            for t in range(1, T):

                if model == 'garch':
                    sigma2[t] = (omega + alpha1 * residuals[t-1]**2 + beta1 * sigma2[t-1])
                    sigma2[t] = max(sigma2[t], 1e-8)

                elif model == 'garch+sentiment':

                    delta = (polarity[t-1] < 0).astype(float)
                    f_t = gamma * (1 - delta) * polarity[t-1]**2 + kappa * delta * polarity[t-1]**2

                    sigma2[t] = (f_t + omega + alpha1 * residuals[t-1]**2 + beta1 * sigma2[t-1])
                    sigma2[t] = max(sigma2[t], 1e-8)

                elif model == 'garch+attention':
                    #d = (growth[t-1] > 0).astype(float)
                    d = (sigma2[t-1] > mu_sigma).astype(float)
                    sigma2[t] = (omega + alpha1 * residuals[t-1]**2 + beta1 * sigma2[t-1] + tau*d*growth[t-1])
                    sigma2[t] = max(sigma2[t], 1e-8)
                
                elif model == 'garch+macro':
                    om = float(epu[t-1] < 0)            
                    mac_tm1 = ep1 * (1 - om) * (epu[t-1]**2)
                    sigma2[t] = (omega + alpha1 * residuals[t-1]**2 + beta1 * sigma2[t-1] + mac_tm1)
                    sigma2[t] = max(sigma2[t], 1e-8)
                
                elif model == 'garch+sentiment+attention':
                    delta = (polarity[t-1] < 0).astype(float)
                    f_t = gamma * (1 - delta) * polarity[t-1]**2 + kappa * delta * polarity[t-1]**2

                    d = (sigma2[t-1] > mu_sigma).astype(float)
                    sigma2[t] = (f_t + omega + alpha1 * residuals[t-1]**2 + beta1 * sigma2[t-1] + tau*d*growth[t-1])
                    sigma2[t] = max(sigma2[t], 1e-8)
                
                elif model == 'garch+sentiment+macro':
                    om = float(epu[t-1] < 0)            
                    mac_tm1 = ep1 * (1 - om) * (epu[t-1]**2)

                    delta = (polarity[t-1] < 0).astype(float)
                    f_t = gamma * (1 - delta) * polarity[t-1]**2 + kappa * delta * polarity[t-1]**2

                    sigma2[t] = (f_t + omega + alpha1 * residuals[t-1]**2 + beta1 * sigma2[t-1] + mac_tm1)
                    sigma2[t] = max(sigma2[t], 1e-8)
                
                elif model == 'garch+attention+macro':
                    om = float(epu[t-1] < 0)            
                    mac_tm1 = ep1 * (1 - om) * (epu[t-1]**2)

                    d = (sigma2[t-1] > mu_sigma).astype(float)
                    sigma2[t] = (omega + alpha1 * residuals[t-1]**2 + beta1 * sigma2[t-1] + tau*d*growth[t-1]+mac_tm1)
                    sigma2[t] = max(sigma2[t], 1e-8)

                elif model == 'garch+sentiment+attention+macro':
                    om = float(epu[t-1] < 0)            
                    mac_tm1 = ep1 * (1 - om) * (epu[t-1]**2)

                    delta = (polarity[t-1] < 0).astype(float) 
                    f_t = gamma * (1 - delta) * polarity[t-1]**2 + kappa * delta * polarity[t-1]**2

                    d = (sigma2[t-1] > mu_sigma).astype(float)
                    sigma2[t] = (f_t + omega + alpha1 * residuals[t-1]**2 + beta1 * sigma2[t-1] + tau*d*growth[t-1]+mac_tm1)
                    sigma2[t] = max(sigma2[t], 1e-8)

            return np.sqrt(sigma2)

        params = [0.05, 0.006, 0.9, 0.5, 0.5, 0.1, 0.2] #omega, alpha, beta, kappa, gamma, tau, ep1

        param_dict = {
            "garch":                                [0.05, 0.006, 0.9],
            "garch+sentiment":                      [0.05, 0.006, 0.9, 0.5, 0.5],
            "garch+attention":                      [0.05, 0.006, 0.9, 0.1],
            "garch+macro":                          [0.05, 0.006, 0.9, 0.2],
            "garch+sentiment+macro":                [0.05, 0.006, 0.9, 0.5, 0.5,0.2],
            "garch+attention+macro":                [0.05, 0.006, 0.9, 0.1, 0.2],
            "garch+sentiment+attention":            [0.05, 0.006, 0.9, 0.5, 0.5, 0.1],
            "garch+sentiment+attention+macro":      [0.05, 0.006, 0.9, 0.5, 0.5, 0.1, 0.2]

        }

        def model_data(train_df, test_df, model, polarity, name_prop):
            #train_df = df[df['date'] < split_date]
            
            #test_df = df[df['date'] >= split_date]

            residuals = train_df['residuals'].to_numpy(dtype=np.float64)


            polarity_d = train_df[polarity].to_numpy(dtype=np.float64)


            volume = train_df['Volume_y'].to_numpy(dtype=np.float64)


            var = train_df['EPU'].to_numpy(dtype=np.float64)


            realized_vol = train_df['realized_vol'].to_numpy(dtype=np.float64)

            mu_sigma = (float(train_df['realized_vol'].mean()))/2

            params = param_dict[model]

            if model == "garch":
                result = fit_garch(params, model, residuals)

                cov_robust = compute_param_cov(
                    params=result.x,
                    model = model,
                    negloglike_fn=garch_negloglike,
                    residuals=residuals,
                    robust=True
                )
            
            elif model == "garch+sentiment":
                result = fit_garch(params = params, model = model, residuals = residuals, polarity = polarity_d)

                cov_robust = compute_param_cov(
                    params=result.x,
                    model = model,
                    negloglike_fn=garch_negloglike,
                    residuals=residuals,
                    polarity = polarity_d,
                    robust=True
                )
            
            elif model == "garch+attention":
                result = fit_garch(params = params, model = model, residuals = residuals, volume = volume, mu_sigma = mu_sigma)
                
                cov_robust = compute_param_cov(
                    params=result.x,
                    model = model,
                    negloglike_fn=garch_negloglike,
                    residuals=residuals,
                    volume = volume,
                    mu_sigma = mu_sigma,
                    robust=True
                )

            elif model == "garch+macro":
                result = fit_garch(params = params, model = model, residuals = residuals, var = var)
                
                cov_robust = compute_param_cov(
                    params=result.x,
                    model = model,
                    negloglike_fn=garch_negloglike,
                    residuals=residuals,
                    var = var,
                    robust=True
                )

            elif model == "garch+sentiment+macro":
                result = fit_garch(params = params, model = model, residuals = residuals, polarity = polarity_d, var = var)
                print(result)
                cov_robust = compute_param_cov(
                    params=result.x,
                    model = model,
                    negloglike_fn=garch_negloglike,
                    residuals=residuals,
                    polarity = polarity_d,
                    var = var,
                    robust=True
                )
            
            elif model == "garch+attention+macro":
                result = fit_garch(params = params, model = model, residuals = residuals, volume = volume, var = var, mu_sigma = mu_sigma)
                
                cov_robust = compute_param_cov(
                    params=result.x,
                    model = model,
                    negloglike_fn=garch_negloglike,
                    residuals=residuals,
                    volume = volume,
                    var = var,
                    mu_sigma = mu_sigma,
                    robust=True
                )

            elif model == "garch+sentiment+attention":
                result = fit_garch(params = params, model = model, residuals = residuals, polarity = polarity_d, volume = volume, mu_sigma = mu_sigma)
                
                cov_robust = compute_param_cov(
                    params=result.x,
                    model = model,
                    negloglike_fn=garch_negloglike,
                    residuals=residuals,
                    polarity = polarity_d,
                    volume = volume,
                    mu_sigma = mu_sigma,
                    robust=True
                )

            elif model == "garch+sentiment+attention+macro":
                result = fit_garch(params = params, model = model, residuals = residuals, polarity = polarity_d, volume = volume, var = var, mu_sigma = mu_sigma)
                print(result)
                cov_robust = compute_param_cov(
                    params=result.x,
                    model = model,
                    negloglike_fn=garch_negloglike,
                    residuals=residuals,
                    polarity = polarity_d,
                    volume = volume,
                    var = var,
                    mu_sigma = mu_sigma,
                    robust=True
                )

            std_err_robust = jnp.sqrt(jnp.diag(cov_robust))

            names_active = active_names(model)

            t_stats = result.x / std_err_robust

            p_vals = 2 * (1 - norm.cdf(jnp.abs(t_stats)))

            residuals = test_df['residuals'].to_numpy(dtype=np.float64)


            polarity_d = test_df[polarity].to_numpy(dtype=np.float64)


            volume = test_df['Volume_y'].to_numpy(dtype=np.float64)

            var = test_df['EPU'].to_numpy(dtype=np.float64)
            
            realized_vol = test_df['realized_vol']

            mu_sigma = (float(test_df['realized_vol'].mean()))/2


            relative_volat = volatility_garch (result.x, model, residuals, polarity_d, volume, var, mu_sigma)
            relative_volat = np.nan_to_num(relative_volat, nan=0.000001)

            nLL = result.fun
            n = len(realized_vol)
            k = len(result.x)
            garch_AIC = 2*k + 2*nLL
            garch_BIC = np.log(n)*k + 2*nLL

            # MAPE, SMAPE
            def mape(y_true, y_pred):
                return np.mean(np.abs((y_true - y_pred) / y_true))

            def rmape(y_true, y_pred):
                numerator = np.abs(y_true - y_pred)
                denominator = (np.abs(y_true) + np.abs(y_pred))
                mask = denominator == 0
                numerator[mask] = 0
                denominator[mask] = 1
                return 2.0 * np.mean(numerator / denominator)


            garch_mae_value = mean_absolute_error(realized_vol, relative_volat)
            garch_mse_value = mean_squared_error(realized_vol, relative_volat)
            garch_rmse_value = np.sqrt(garch_mse_value)
            garch_mape_val = mape(realized_vol, relative_volat)
            garch_rmape_val = rmape(realized_vol, relative_volat)

            summary = pd.DataFrame({
                'coef': result.x,
                'std err': std_err_robust,
                't': t_stats,
                'P>|t|': p_vals,
                '95% Conf. Int.': ['[{:.3e},{:.3e}]'.format(low, high) 
                                        for low, high in zip(result.x - 1.96 * std_err_robust, result.x + 1.96 * std_err_robust)]
                
            }, index=names_active)

            metrics = pd.DataFrame(
                {
                    "LogLikelyhood"     : nLL,
                    "AIC"     : garch_AIC,
                    "BIC"     : garch_BIC,
                    "MAE"     : garch_mae_value,
                    "MSE"     : garch_mse_value,
                    "RMSE"    : garch_rmse_value,
                    "MAPE"    : garch_mape_val,
                    "RMAPE"   : garch_rmape_val,
                },
                index=[model]                      
            )

            coef_table = summary.T.stack().to_frame().T      
            coef_table.index = [model] 
           

            # Plot realized vs fitted
            save_dir = Path(f"plots/{company}/realized_vs_fitted")
            save_dir.mkdir(parents=True, exist_ok=True)



            full_table = pd.concat([metrics, coef_table], axis=1)
       

            return full_table, result.x
        

        def white_data(df, model, polarity, params, name_prop):
            # everything from the same df
            residuals   = df['residuals'].to_numpy(dtype=np.float64)
            volume      = df['Volume_y'].to_numpy(dtype=np.float64)
            var         = df['EPU'].to_numpy(dtype=np.float64)
            polarity_d  = df[polarity].to_numpy(dtype=np.float64)
            mu_sigma    = float(df['realized_vol'].mean()) / 2.0  # needed by *attention* variants

            # compute fitted vol
            relative_volat = volatility_garch(params, model, residuals,
                                            polarity_d, volume, var, mu_sigma)
            relative_volat = np.nan_to_num(relative_volat, nan=1e-6)

            # white-noise tests (use the same df / polarity array you just built)
            white_noise = residuals / relative_volat
            white_test  = statistical_tests(white_noise, name_prop, polarity)

            # persist comparable-length table (all from df)
            volat_dir = Path(f"volat_table/{company}")
            volat_dir.mkdir(exist_ok=True)

            volat_t = pd.DataFrame({
                "date"          : df["date"].to_numpy(),          
                "realized_vol"  : df["realized_vol"].to_numpy(),  
                "relative_volat": relative_volat
            })

            volat_t.to_csv(volat_dir / f"{model}+{name_prop}+{polarity}.csv", index=True)

            return white_test




            
        def all_polarity_columns(ema_spans): #To get all polarity columns
            cols = ["PolarityFinb", "PolarityRob"]
            for span in ema_spans:
                cols.append(f"PolarityFinb_EMA_{span}")
                cols.append(f"PolarityRob_EMA_{span}")
            return cols
        
        row_tables   = []
        row_tables_mean = []
        white_tests  = {}   

        N_SPLITS   = 3
        TEST_SIZE  = 400
        tscv = TimeSeriesSplit(n_splits=N_SPLITS,
                            test_size=TEST_SIZE,
                            gap=0) 
        
        if group is None:
            items = merged_df.items()  # all groups
        elif isinstance(group, tuple):  
            if group not in merged_df:
                raise KeyError(f"{group} not in merged_df keys")
            items = [(group, merged_df[group])]

        if model == "garch":
            # only one run
            df = merged_df[("count", "Mean")]
            train = df_slice[df_slice['date'] < split_date]
            
            test = df_slice[df_slice['date'] >= split_date]  

            for fold, (train_idx, test_idx) in enumerate(tscv.split(train)):
                train_df = df_slice.iloc[train_idx].reset_index(drop=True)
                test_df  = df_slice.iloc[test_idx ].reset_index(drop=True)

                full_table, params = model_data(train_df, test_df, model, "PolarityRob", "prop")
                key = ('count','Mean')
                run_label = f"{model}+('count','Mean')+PolarityRob"
                full_table = full_table.copy()
                full_table.index = [run_label]

                row_tables.append(full_table)

            stacked      = pd.concat(row_tables, axis=0)
            num_cols     = stacked.select_dtypes(np.number).columns
            mean_tbl = stacked[num_cols].groupby(level=0).mean()
            row_tables_mean.append(mean_tbl)

            white_test = white_data(test, model, "PolarityRob", params, key)
            white_tests[run_label] = white_test
            

        else:  # every other model
            for key, df_slice in merged_df.items():
                train = df_slice[df_slice['date'] < split_date]
            
                test = df_slice[df_slice['date'] >= split_date]        
                for pol_col in all_polarity_columns(ema_spans):
                    for fold, (train_idx, test_idx) in enumerate(tscv.split(train)):

                        train_df = train.iloc[train_idx].reset_index(drop=True)
                        test_df  = train.iloc[test_idx ].reset_index(drop=True)

                        full_table, params = model_data(train_df, test_df, model, pol_col, key)

                        run_label = f"{model}+{key}+{pol_col}"
                        full_table = full_table.copy()
                        full_table.index = [run_label]

                        row_tables.append(full_table)

                    stacked      = pd.concat(row_tables, axis=0)
                    num_cols     = stacked.select_dtypes(np.number).columns
                    mean_tbl = stacked[num_cols].groupby(level=0).mean()
                    row_tables_mean.append(mean_tbl)
                    white_test = white_data(test, model, pol_col,params, key)
                    white_tests[run_label] = white_test



        # CSV with all the data
        result_dir = Path(f"result_table_{company}")
        result_dir.mkdir(exist_ok=True)

        big_table = pd.concat(row_tables_mean, axis=0)
        big_table = big_table.sort_values("AIC", ascending=False)
        big_table.to_csv(result_dir / f"results_{model}.csv",index=True)

        # white-noise report in its own .txt file
        wt_dir = Path(f"white_tests_{company}")
        wt_dir.mkdir(exist_ok=True)

        for label, report in white_tests.items():
            (wt_dir / f"{label}.txt").write_text(report, encoding="utf-8")
            
        return row_tables  