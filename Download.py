import time
import tushare as ts
import pandas as pd

# 1. 初始化tushare
TOKEN = "768f55e6ede1f29cc6d472d244d78677a1bae0692cad8367df274893"
ts.set_token(TOKEN)
pro = ts.pro_api()

def download_index_data(name, ts_code, api_func, years=range(2016, 2021), delay=0):
    """通用下载函数：按年分批获取指数日线数据"""
    all_data = []
    for year in years:
        s_date = f"{year}0101"
        e_date = f"{year}1231"
        print(f"  正在下载 {year} 年...")
        df_year = api_func(ts_code=ts_code, start_date=s_date, end_date=e_date)
        if not df_year.empty:
            all_data.append(df_year)
        if delay > 0:
            time.sleep(delay)

    df = pd.concat(all_data)
    df = df.sort_values("trade_date").reset_index(drop=True)

    # 保存
    df.to_csv(f"{name}.csv", index=False, encoding="utf-8-sig")
    df.to_parquet(f"{name}.parquet", index=False)

    print(f"  {name} 下载完成，共 {len(df)} 条日线数据")
    return df


# # 2. 下载沪深300
# print("【沪深300】")
# df_hs300 = download_index_data(
#     name="沪深300_2021-2026.7年日线",
#     ts_code="000300.SH",
#     api_func=pro.index_daily,
# )

# # 3. 下载标普500
# print("【标普500】")
# df_spx = download_index_data(
#     name="标普500_2021-2026.7年日线",
#     ts_code="SPX",
#     api_func=pro.index_global,
#     delay=7,  # index_global 限频 10次/分钟，间隔7秒
# )

# # 4. 下载沪金期货
# print("【沪金期货】")
# df_gold = download_index_data(
#     name="沪金期货_2021-2026.7年日线",
#     ts_code="AU.SHF",
#     api_func=pro.fut_daily,
# )

# 5. 下载黄金ETF (华安黄金ETF 518880)
print("【黄金ETF】")
df_gold_etf = download_index_data(
    name="黄金ETF_2016-2020年日线",
    ts_code="518880.SH",
    api_func=pro.fund_daily,
)

print("\n全部下载完成！")
