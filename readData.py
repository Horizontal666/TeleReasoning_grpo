import pandas as pd
df = pd.read_parquet("/workspace/wbh/202509_InferenceModel/data/gsm8k/test.parquet")
print(df.columns)
print(df.head())

# 查看所有列名
print("列名:", df.columns.tolist())

# 查看数据行数（可选）
print("总样本数:", len(df))

# 查看第一个样本（索引 0）
first = df.iloc[0]
print("第一个样本内容:")
for col in df.columns:
    print(f"{col}: {first[col]}")