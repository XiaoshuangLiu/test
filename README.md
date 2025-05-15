# test

import pandas as pd
from scipy.signal import lfilter   # C-level implementation, fast

# ── 假设你的 DataFrame 叫 media ──────────────────────────────────
# 列: WEEK_NBR, DMA_ID, TTL_FS_IMPR, TTL_GROWTH_IMPR, TTL_ACQ_IMPR, TTL_Retarget_IMPR
# 如果还没有读进来，请先:
# media = pd.read_csv("xxx.csv")   # or other source

# 1. 定义 geometric adstock 函数  y_t = x_t + λ · y_{t-1}
def adstock_geometric(series: pd.Series, lmbda: float = 0.6) -> pd.Series:
    """
    Parameters
    ----------
    series : 1-D pandas Series, the raw impressions for *one* DMA
    lmbda  : float, 0 ≤ λ < 1, carry-over rate

    Returns
    -------
    pandas Series, same index, adstocked values
    """
    return pd.Series(
        lfilter([1.0], [1.0, -lmbda], series.values),
        index=series.index,
        name=f"{series.name}_AD"
    )

# 2. Impression columns to transform
impr_cols = [
    "TTL_FS_IMPR",
    "TTL_GROWTH_IMPR",
    "TTL_ACQ_IMPR",
    "TTL_Retarget_IMPR",
]

# 3. 保证按 DMA + WEEK_NBR 排序，再按组做 adstock
media = (
    media.sort_values(["DMA_ID", "WEEK_NBR"])     # critical for carry-over
          .reset_index(drop=True)
)

for col in impr_cols:
    media[f"{col}_AD"] = (
        media
        .groupby("DMA_ID", group_keys=False)[col]     # 每个 DMA 独立递归
        .apply(lambda s: adstock_geometric(s, lmbda=0.6))
    )

# 4. 结果示例
display_cols = ["DMA_ID", "WEEK_NBR"] + impr_cols + [f"{c}_AD" for c in impr_cols]
print(media[display_cols].head(10))


import statsmodels.formula.api as smf
formula = (
    "sales ~ C(DMA_ID) + TTL_FS_IMPR_AD + TTL_GROWTH_IMPR_AD "
    "+ TTL_ACQ_IMPR_AD + TTL_Retarget_IMPR_AD"
)
model = smf.ols(formula, data=media).fit()
print(model.summary())

import matplotlib.pyplot as plt
import pandas as pd

# ╭─▼─ 必填：要绘制的 impression 列 ───────────────────────────────╮
impr_cols = [
    "TTL_FS_IMPR",
    "TTL_GROWTH_IMPR",
    "TTL_ACQ_IMPR",
    "TTL_Retarget_IMPR",
]
# ╰──────────────────────────────────────────────────────────────╯

# ── 选填：指定某个 DMA；若置为 None 则汇总全部 DMA ────────────
dma_to_plot = None          # 例如 500；None => all DMA aggregated
# ----------------------------------------------------------------

# 1. 准备绘图 DataFrame
if dma_to_plot is not None:
    df_plot = media.loc[media["DMA_ID"] == dma_to_plot].copy()
    title_suffix = f" (DMA {dma_to_plot})"
else:
    df_plot = (
        media
        .groupby("WEEK_NBR", as_index=False)[impr_cols + [f"{c}_AD" for c in impr_cols]]
        .sum()
    )

    
    title_suffix = " (All DMA)"




    

# 2. 按列绘图：raw vs. adstock
for col in impr_cols:
    ad_col = f"{col}_AD"

    df_sub = df_plot[["WEEK_NBR", col, ad_col]].sort_values("WEEK_NBR")

    plt.figure()                           # 每个指标独立一张图
    plt.plot(df_sub["WEEK_NBR"], df_sub[col], label="raw")
    plt.plot(df_sub["WEEK_NBR"], df_sub[ad_col], label="adstock", linestyle="--")
    plt.title(f"{col}{title_suffix}")
    plt.xlabel("Week")
    plt.ylabel("Impressions")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    



import matplotlib.pyplot as plt
import pandas as pd

# ── parameters you might tweak ─────────────────────────────────
impr_cols   = ["TTL_FS_IMPR",
               "TTL_GROWTH_IMPR",
               "TTL_ACQ_IMPR",
               "TTL_Retarget_IMPR"]
dma_to_plot = None          # e.g. 500 → single DMA, None → aggregate
# ───────────────────────────────────────────────────────────────

# --- build plotting frame -------------------------------------
if dma_to_plot is not None:
    df_plot = media.loc[media["DMA_ID"] == dma_to_plot].copy()
    title_suffix = f"(DMA {dma_to_plot})"
else:
    agg_cols   = impr_cols + [f"{c}_AD" for c in impr_cols]
    df_plot    = (
        media.groupby("WEEK_NBR", as_index=False)[agg_cols]
             .sum()
    )
    title_suffix = "(All DMA)"
df_plot = df_plot.sort_values("WEEK_NBR")

# --- create 4-row × 1-col subplot figure ----------------------
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=True)

for ax, col in zip(axes, impr_cols):
    ad_col = f"{col}_AD"
    ax.plot(df_plot["WEEK_NBR"], df_plot[col],      label="raw")
    ax.plot(df_plot["WEEK_NBR"], df_plot[ad_col],   label="adstock", linestyle="--")
    ax.set_title(f"{col} {title_suffix}")
    ax.set_ylabel("Impressions")
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Week")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



import pandas as pd
from scipy.signal import lfilter
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# 0. 基本参数
impr_cols   = ["TTL_FS_IMPR",
               "TTL_GROWTH_IMPR",
               "TTL_ACQ_IMPR",
               "TTL_Retarget_IMPR"]
λ           = 0.6               # carry-over rate
dma_to_plot = None              # 聚合全部 DMA；指定 500 看单 DMA
# ──────────────────────────────────────────────────────────────

# 1. adstock 函数 (geometric)
def adstock_geometric(series, lmbda=λ):
    return lfilter([1.0], [1.0, -lmbda], series.values)

# 2. 生成 adstock 列，并在 **每个 DMA 内 renormalize**
media = media.sort_values(["DMA_ID", "WEEK_NBR"]).reset_index(drop=True)

for col in impr_cols:
    ad_name = f"{col}_AD"
    rn_name = f"{col}_AD_RN"

    # step-A: adstock by DMA
    media[ad_name] = (
        media.groupby("DMA_ID", group_keys=False)[col]
             .apply(lambda s: adstock_geometric(s))
    )

    # step-B: renormalize inside each DMA
    def _renorm(g):
        factor = g[col].sum() / g[ad_name].sum()
        return g[ad_name] * factor

    media[rn_name] = (
        media.groupby("DMA_ID", group_keys=False)[[col, ad_name]]
             .apply(lambda g: _renorm(g))
             .reset_index(level=0, drop=True)
    )

# 3. 为绘图准备数据（聚合或单 DMA）
if dma_to_plot is not None:
    df_plot = media.loc[media["DMA_ID"] == dma_to_plot].copy()
    title_suffix = f"(DMA {dma_to_plot})"
else:
    agg_cols = (impr_cols +
                [f"{c}_AD_RN" for c in impr_cols])
    df_plot = (
        media.groupby("WEEK_NBR", as_index=False)[agg_cols]
             .sum()
    )
    title_suffix = "(All DMA)"

df_plot = df_plot.sort_values("WEEK_NBR")

# 4. 画 4 行 × 1 列子图：raw vs adstock_renorm
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12), sharex=True)

for ax, col in zip(axes, impr_cols):
    rn_col = f"{col}_AD_RN"
    ax.plot(df_plot["WEEK_NBR"], df_plot[col],   label="raw")
    ax.plot(df_plot["WEEK_NBR"], df_plot[rn_col],label="adstock renorm", linestyle="--")
    ax.set_title(f"{col} {title_suffix}")
    ax.set_ylabel("Impressions")
    ax.legend(loc="upper right")

axes[-1].set_xlabel("Week")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





from scipy.signal import lfilter
import pandas as pd

def adstock_geometric(series: pd.Series, decay: float = 0.6, normalize: bool = True) -> pd.Series:
    """
    Geometric adstock with optional peak normalization.
    """
    # 1. 用 lfilter 做高效递归
    vals = lfilter([1.0], [1.0, -decay], series.values)
    
    # 2. 可选的峰值归一化
    if normalize and vals.max() != 0:
        vals = vals * (series.max() / vals.max())
    
    return pd.Series(vals, index=series.index, name=f"{series.name}_AD")

# groupby 示例
media = media.sort_values(["DMA_ID","WEEK_NBR"])
for col in impr_cols:
    media[f"{col}_AD"] = (
        media
        .groupby("DMA_ID", group_keys=False)[col]
        .apply(lambda s: adstock_geometric(s, decay=0.6, normalize=True))
    )
