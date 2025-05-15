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
    

