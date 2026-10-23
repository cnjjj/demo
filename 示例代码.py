#变量
patient_name = "张某"       # 字符串：患者姓名
age = 65                   # 整数：年龄
#pulse_rate = 72.5          # 浮点数：脉率（次/分）
is_qi_deficiency = True    # 布尔值：是否气虚证

#列表
symptoms = ["乏力", "气短", "自汗", "乏力"]  # 症状序列（可重复）
herbs = ["黄芪", "白术", "茯苓", "甘草"]     # 方剂组成
pulse_records = [72, 75, 71]              # 三次脉率测量

#字典
patients_data = {
    "姓名": "张某",
    "年龄": 65,
    "舌色": "淡白",
    "脉率": 72,
    "脉象": "细弱",
    "主诉": ["乏力", "气短", "自汗"],
    "证型": "气虚证"
}


#变量赋值
#pulse_rate = 101
tongue_color = "淡白"

#条件判断判断，and 代表”并且“，or 代表”或者“
if pulse_rate < 60 and tongue_color == "淡白":
    preliminary = "阳虚证"
elif pulse_rate > 90 and tongue_color == "红绛":
    preliminary = "阴虚火旺证"
else:
    preliminary = "需进一步四诊合参"

# 输出：需进一步四诊合参
print(preliminary)


#循环
# 遍历患者名单
patients = ["张某", "李某", "王某"]
for name in patients:
    print(f"正在处理患者：{name}")


#循环
# 统计"乏力"出现次数
symptoms = ["乏力", "气短", "乏力", "自汗", "乏力"]
fatigue_count = 0
for s in symptoms:
    if s == "乏力":
        fatigue_count += 1     # 每次看到乏力，计数器加 1
print(f"乏力出现次数：{fatigue_count}")  # 输出：3


# 现有三位患者字典库
patients_data = [
    {"name": "张三", "age": 45, "syndrome": "气虚证", "gender": "男"},
    {"name": "李四", "age": 30, "syndrome": "阴虚证", "gender": "女"},
    {"name": "王五", "age": 62, "syndrome": "气虚证", "gender": "男"}
]
#列表推导式
qi_patients = [p["name"] for p in patients_data if p["syndrome"] == "气虚证"]
print(qi_patients)   # 输出 ['张三','王五']

#函数
def diagnose_by_pulse(pulse_rate, pulse_type):
    """
    根据脉率和脉象类型输出初步证型提示
    """
    if pulse_rate < 60 and pulse_type == "细弱":
        return "阳虚证"
    else:
        return "需进一步辨证"

#调用
result = diagnose_by_pulse(68, "细弱")
print(result)  # 输出：需进一步辨证

#模块导入
#import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. 创建一个数组
numbers = np.array([1, 2, 3, 4, 5])

# 2. 快速进行数学计算 (比如求平均值)
average = np.mean(numbers)
print(f"平均值是: {average}")


import pandas as pd
# 提取表格，转化为 DataFrame 数据结构
df = pd.read_csv("二维数据表.csv", encoding="gbk")

# 三行代码完成数据"望诊"
print(df.head())      # 查看行，确认数据格式
print(df.info())      # 查看列名、数据类型、非空计数
print(df.describe())  # 数值列的统计描述（均值、标准差、最值）


# 剔除中文单位，并将该列转化为系统可计算的数值型数据
df['pulse_rate'] = df['pulse_rate'].astype(str).str.replace('次/分', '')
df['pulse_rate'] = pd.to_numeric(df['pulse_rate'], errors='coerce')


# 利用 IQR 原则过滤脉率列的极端异常值
Q1 = df['pulse_rate'].quantile(0.25)
Q3 = df['pulse_rate'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 仅保留处于正常生理或病理范围内的患者数据
df = df[(df['pulse_rate'] >= lower_bound) & (df['pulse_rate'] <= upper_bound)]

# 按证型分组，计算气虚与非气虚群体的均值，精准填补乏力评分的空白
df['fatigue_score'] = df.groupby('syndrome')['fatigue_score'].transform(lambda x: x.fillna(x.mean()))


# 将中医有序变量映射为数值特征，有序类别
tongue_mapping = {'淡白舌': 1, '淡红舌': 2, '红绛舌': 3}
df['tongue_color_score'] = df['tongue_color_score'].map(tongue_mapping)

coating_mapping = {'薄苔': 1, '厚苔': 2, '腻苔': 3}
df['coating_thickness'] = df['coating_thickness'].map(coating_mapping)

# 将中医有序变量映射为数值特征，无序类别
# 将脉象和证型拆为独立的0/1特征列，指定dtype=int确保输出整数0/1
# 如"脉象"列中的各类别自动拆分为"脉象_浮""脉象_沉"等0/1特征列，由原始数据中的唯一值自动决定
df = pd.get_dummies(df, columns=["脉象"], prefix="脉象", dtype=int)
df = pd.get_dummies(df, columns=["证型"], prefix="证型", dtype=int)


# 提取特征矩阵 X，对应患者各项客观化四诊指标
X = df[['tongue_color_score', 'pulse_rate', 'coating_thickness', 'fatigue_score']]

# 提取标签向量 y
y = df['syndrome']

# 将清洗完毕的特征矩阵 X 和标签向量 y 重新拼合成一个完整的标准数据表
clean_df = pd.concat([X, y], axis=1)

# 将表格保存为 CSV 文件
# index=False 表示不保存行号，encoding='utf-8' 确保中文在绝大多数系统中不乱码
clean_df.to_csv("模拟临床四诊数据.csv", index=False, encoding='utf-8')


#运行完整代码
# ===== 2. 数据准备 =====
X = df[['tongue_color_score', 'pulse_rate', 'coating_thickness', 'fatigue_score']]
y = df['syndrome']

# 分层抽样切分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ===== 3. 模型训练 (随机森林) =====
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ===== 4. 评估与解释 =====
y_pred = model.predict(X_test)
print("【分类报告】:")
print(classification_report(y_test, y_pred))

# 绘制混淆矩阵热力图
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('证型判别混淆矩阵')
plt.xlabel('模型预测结果')
plt.ylabel('医生真实诊断')
plt.show()  # 注意：程序会在这里暂停，关闭弹出的图表后才会继续运行！

# 绘制特征重要性排序图
importances = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 4))
importances.sort_values().plot(kind='barh', color='teal')
plt.title('四诊特征对证型判别的重要性排序')
plt.show()  # 关闭此图后继续

# ===== 5. 交叉验证 =====
scores = cross_val_score(model, X, y, cv=5)
print(f"【5折交叉验证准确率】: {scores.mean():.3f} ± {scores.std():.3f}"