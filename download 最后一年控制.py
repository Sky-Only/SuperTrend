import time
import tushare as ts
import pandas as pd

# 1. 初始化tushare
TOKEN = "768f55e6ede1f29cc6d472d244d78677a1bae0692cad8367df274893"
ts.set_token(TOKEN)
pro = ts.pro_api()

def download_index_data(name, ts_code, api_func, start_year=2016, end_date="20260731", delay=0):
    """
    通用下载函数：按年分批获取日线数据

    参数:
        name:       保存文件名（不含扩展名）
        ts_code:    tushare 品种代码
        api_func:   tushare API 函数
        start_year: 起始年份
        end_date:   截止日期，格式 "YYYYMMDD"（最后一年只下载到该日期）
        delay:      每次请求间隔秒数
    """
    end_year = int(end_date[:4])
    all_data = []
    years = list(range(start_year, end_year + 1))

    for year in years:
        s_date = f"{year}0101"
        if year < end_year:
            e_date = f"{year}1231"
        else:
            e_date = end_date  # 最后一年使用指定的截止日期
        print(f"  正在下载 {year} 年 ({s_date} ~ {e_date})...")
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
    name="黄金ETF_10年日线",
    ts_code="518880.SH",
    api_func=pro.fund_daily,
)

# 6. 下载创业板指 (399006.SZ)
print("【创业板指】")
df_cyb = download_index_data(
    name="创业板指_10年日线",
    ts_code="399006.SZ",
    api_func=pro.index_daily,
)

# 7. 下载中证1000 (000852.SH)
print("【中证1000】")
df_zz1000 = download_index_data(
    name="中证1000_10年日线",
    ts_code="000852.SH",
    api_func=pro.index_daily,
)

# 8. 下载中证2000 (932000.CSI)
print("【中证2000】")
df_zz2000 = download_index_data(
    name="中证2000_10年日线",
    ts_code="932000.CSI",
    api_func=pro.index_daily,
)

# 9. 下载科创50 (000688.SH)
print("【科创50】")
df_kc50 = download_index_data(
    name="科创50_10年日线",
    ts_code="000688.SH",
    api_func=pro.index_daily,
)

print("\n全部下载完成！")
