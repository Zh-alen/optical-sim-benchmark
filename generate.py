import pandas as pd
import numpy as np

# --- 自定义配置 ---
min_val = 2000          # 最小值
max_val = 2300       # 最大值
count = 28        # 生成的总个数
file_name = 'random_numbers5.xlsx'  # 导出的文件名

# 1. 使用 numpy 生成大量随机整数（速度极快）
# np.random.randint 的范围是 [low, high)，所以 max_val 需要 +1
random_data = np.random.randint(min_val, max_val + 1, size=count)

# 2. 将数据转换为 pandas 的 DataFrame
# 我们给这列取个名字叫 "Random Numbers"
df = pd.DataFrame(random_data, columns=['Random Numbers'])
# 3. 导出到 Excel
# index=False 表示不保存行索引（即左侧的 0, 1, 2...）
df.to_excel(file_name, index=False)

print(f"成功！已生成 {count} 个随机整数并保存到 '{file_name}'。")