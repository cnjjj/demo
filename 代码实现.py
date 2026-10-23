import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ===== 前置设置：解决图表中文乱码 =====
# Windows 用户默认使用 SimHei，Mac 用户如果报错，请改为 'Arial Unicode MS'
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== 1. 生成【符合临床规律】的模拟数据 =====
# 输入真实文件路径，并在最前面加上转义字符r
csv_path = r"/home/mw/project/模拟临床四诊数据.csv"
# 如果读取报错乱码，把 utf-8 换成 gbk 试试
df = pd.read_csv(csv_path, encoding='gbk')

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
print(f"【5折交叉验证准确率】: {scores.mean():.3f} ± {scores.std():.3f}")