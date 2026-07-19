import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)

# =========================================================
# LAB 1: Dataset Exploration
# =========================================================
print("=" * 60)
print("LAB 1: Dataset Exploration")
print("=" * 60)

# ---- Load Dataset ----
df = pd.read_csv("one_piece_characters.csv")
print("\nโหลดข้อมูลสำเร็จ")
print(df.head())

# ---- Display Shape ----
print(f"\nShape ของข้อมูล (แถว, คอลัมน์): {df.shape}")

# ---- Display Data Types ----
print("\nData Types ของแต่ละคอลัมน์:")
print(df.dtypes)

# ---- Display Summary Statistics ----
print("\nSummary Statistics:")
print(df.describe(include="all"))

# ---- Display Missing Values ----
print("\nจำนวน Missing Values ต่อคอลัมน์:")
print(df.isnull().sum())

# ---- Display Duplicate Records ----
print(f"\nจำนวนแถวที่ซ้ำกัน (Duplicate Records): {df.duplicated().sum()}")

# ---- Display Class Distribution ----
# ใช้คอลัมน์ Status เป็น target/class (Alive / Deceased)
print("\nClass Distribution ของ Status:")
print(df["Status"].value_counts())
print(df["Status"].value_counts(normalize=True).round(3) * 100, "%")


# =========================================================
# LAB 2: Data Visualization
# =========================================================
print("\n" + "=" * 60)
print("LAB 2: Data Visualization")
print("=" * 60)

# ---- Histogram ----
# ดูการกระจายตัวของตัวแปรตัวเลขทั้งหมดในทีเดียว (สังเกต Height_cm จะมี bar ที่ผิดปกติ
# จากค่า 6660 ที่พิมพ์ผิด - นี่คือจุดที่เราจะไปแก้ใน Part 3)
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols].hist(figsize=(12, 8), bins=15, edgecolor="black")
plt.tight_layout()
plt.savefig("histogram.png")
plt.close()
print("\nบันทึกกราฟ histogram.png แล้ว")

# ---- Correlation Heatmap ----
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.close()
print("บันทึกกราฟ correlation_heatmap.png แล้ว")


# =========================================================
# Part 3: Data Cleaning
# =========================================================
print("\n" + "=" * 60)
print("Part 3: Data Cleaning")
print("=" * 60)

df_clean = df.copy()

# ---- แก้ Age ที่มีตัวอักษรปนกับตัวเลข (เช่น "1000+" ของ Zunesha) ----
# pd.to_numeric(errors="coerce") จะพยายามแปลงเป็นตัวเลข ถ้าแปลงไม่ได้จะใส่ NaN ให้แทนที่จะ error ทั้งโปรแกรม
n_before = df_clean["Age"].isnull().sum()
df_clean["Age"] = pd.to_numeric(df_clean["Age"], errors="coerce")
n_after = df_clean["Age"].isnull().sum()
print(f"\nAge ที่แปลงเป็นตัวเลขไม่ได้ (เช่น '1000+'): {n_after - n_before} ค่า -> กลายเป็น NaN แล้วรอเติมค่าต่อ")

# ---- เปรียบเทียบ Mean vs Median ก่อนแก้ไข (คอลัมน์ Height_cm มีทั้ง missing และ outlier) ----
mean_before = df_clean["Height_cm"].mean()
median_before = df_clean["Height_cm"].median()
print(f"\nHeight_cm -> Mean ก่อนแก้: {mean_before:.2f}, Median ก่อนแก้: {median_before:.2f}")
print("สังเกตว่า Mean ถูกดึงสูงเพราะมีตัวละครที่สูงมากๆ อย่าง Big Mom/Kaidou (ค่าจริง ไม่ใช่ error)")
print("นี่คือตัวอย่างว่าทำไมเวลาข้อมูลเบ้ (skewed) เราถึงมักเติม missing values ด้วย Median แทน Mean")

# ---- Incorrect Data Correction ----
# หมายเหตุ: Big Mom, Kaidou, King ตัวสูงเป็นหลักร้อยถึงเกือบ 900 ซม. ไม่ใช่ข้อมูลพิมพ์ผิด
# แต่เป็นค่าจริงในเรื่อง (ยักษ์/มนุษย์ปลา) ตรงนี้คือจุดสำคัญ: ก่อนจะ "แก้" ค่าที่ดูสุดโต่ง
# ต้องเช็คก่อนว่ามันคือ error จริงๆ หรือเป็นค่าที่ถูกต้องแต่บังเอิญต่างจากกลุ่มมาก
# (ถ้าลบ/แก้มั่วจะทำให้ข้อมูลผิดยิ่งกว่าเดิม)

# แก้ Age ที่ติดลบ (Portgas D. Ace อายุ -20 เป็นไปไม่ได้) ด้วยการใช้ค่าสัมบูรณ์
invalid_age_count = (df_clean["Age"] < 0).sum()
print(f"\nพบ Age ที่ติดลบ (ผิดปกติ): {invalid_age_count} แถว -> แก้เป็นค่าสัมบูรณ์")
df_clean["Age"] = df_clean["Age"].abs()

# แก้ข้อมูล Crew ที่พิมพ์ไม่ตรงกัน (ตัวพิมพ์เล็ก/มีช่องว่างเกิน) ให้เป็นรูปแบบเดียวกัน
df_clean["Crew"] = df_clean["Crew"].str.strip().str.title()
df_clean["Devil_Fruit_Type"] = df_clean["Devil_Fruit_Type"].str.strip().str.title()

# ---- Missing Value Handling ----
# Bounty: ตัวเลขต่อเนื่อง มี outlier (Roger บันทีบันลือโลก) -> เติมด้วย Median
df_clean["Bounty"] = df_clean["Bounty"].fillna(df_clean["Bounty"].median())

# Height_cm, Age: เติมด้วย Median เช่นกัน (ทนต่อ outlier กว่า Mean)
df_clean["Height_cm"] = df_clean["Height_cm"].fillna(df_clean["Height_cm"].median())
df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())

# Origin_Sea: เป็น category เติมด้วยค่าที่พบบ่อยที่สุด (mode)
df_clean["Origin_Sea"] = df_clean["Origin_Sea"].fillna(df_clean["Origin_Sea"].mode()[0])

# Status: เป็น target column เติมด้วย mode เหมือนกัน (ในทางปฏิบัติจริงบางทีลบแถวทิ้งดีกว่า
# ถ้า target หายไปเยอะ แต่ที่นี่หายแค่ 4 จาก 39 แถว เติมด้วย mode ได้)
df_clean["Status"] = df_clean["Status"].fillna(df_clean["Status"].mode()[0])

# Crew: เผื่อไว้กรณีมีตัวละครที่ไม่ได้สังกัดกลุ่มโจรสลัดใดๆ -> missing แบบนี้มีความหมาย
# เติมด้วยข้อความเฉพาะแทนการเติมด้วย mode ที่จะให้ความหมายผิด (ไม่มีผลตอนนี้เพราะไม่มีค่าไหนหาย)
df_clean["Crew"] = df_clean["Crew"].fillna("Independent")

# Devil_Fruit / Devil_Fruit_Type: NaN ในที่นี้ "มีความหมาย" คือตัวละครไม่ได้กินผลไม้ปีศาจ
# ไม่ควรเติมด้วย mean/median/mode เพราะจะกลายเป็นข้อมูลเท็จ -> เติมด้วยคำว่า "None" แทน
df_clean["Devil_Fruit"] = df_clean["Devil_Fruit"].fillna("None")
df_clean["Devil_Fruit_Type"] = df_clean["Devil_Fruit_Type"].fillna("None")

print("\nMissing Values หลังจัดการ:")
print(df_clean.isnull().sum())

mean_after = df_clean["Height_cm"].mean()
median_after = df_clean["Height_cm"].median()
print(f"\nHeight_cm -> Mean หลังแก้: {mean_after:.2f}, Median หลังแก้: {median_after:.2f}")

# ---- Duplicate Removal ----
before_rows = df_clean.shape[0]
df_clean = df_clean.drop_duplicates()
after_rows = df_clean.shape[0]
print(f"\nลบแถวซ้ำ: {before_rows - after_rows} แถว (จาก {before_rows} เหลือ {after_rows})")

# ---- Data Type Conversion ----
# Devil_Fruit_Type และ Status ทางความหมายเป็นหมวดหมู่ ไม่ใช่ข้อความอิสระ -> แปลงเป็น category
df_clean["Devil_Fruit_Type"] = df_clean["Devil_Fruit_Type"].astype("category")
df_clean["Status"] = df_clean["Status"].astype("category")
# Bounty ควรเป็นจำนวนเต็ม (ไม่มีสตางค์ในเบลรี่)
df_clean["Bounty"] = df_clean["Bounty"].astype("int64")
print("\nData Types หลังแปลง:")
print(df_clean.dtypes)


# =========================================================
# Part 4: Feature Engineering
# =========================================================
print("\n" + "=" * 60)
print("Part 4: Feature Engineering")
print("=" * 60)

df_encoded = df_clean.copy()

# ---- Label Encoding ----
# ใช้กับตัวแปรที่มีแค่ 2 ค่า เช่น Status (Alive/Deceased) -> 0/1
le = LabelEncoder()
df_encoded["Status_encoded"] = le.fit_transform(df_encoded["Status"])
print("\nตัวอย่าง Label Encoding คอลัมน์ Status:")
print(df_encoded[["Status", "Status_encoded"]].drop_duplicates())

# ---- One-Hot Encoding ----
# ใช้กับตัวแปรที่มีหลายค่าและไม่มีลำดับ เช่น Devil_Fruit_Type (Paramecia/Zoan/Logia/None)
# ถ้าใช้ Label Encoding จะทำให้โมเดลเข้าใจผิดว่ามีลำดับความสำคัญ ทั้งที่จริงไม่มี
df_encoded = pd.get_dummies(df_encoded, columns=["Devil_Fruit_Type"], prefix="DFType")
print("\nคอลัมน์หลังทำ One-Hot Encoding:")
print([c for c in df_encoded.columns if "DFType" in c])

print("\nตัวอย่างข้อมูลหลัง Feature Engineering ครบทุกขั้นตอน:")
print(df_encoded.head())

# บันทึกไฟล์ผลลัพธ์สุดท้ายเผื่อเอาไปใช้ต่อ
df_encoded.to_csv("one_piece_processed.csv", index=False)
print("\nบันทึก one_piece_processed.csv (ข้อมูลที่ preprocess เสร็จแล้ว) เรียบร้อย")

print("\n" + "=" * 60)
print("จบการทำงานทั้งหมด")
print("=" * 60)
