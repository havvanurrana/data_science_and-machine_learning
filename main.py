# BU ÇALIŞMA GENETİK VERİLER ÜZERİNE VERİ BİLİMİ VE MAKİNE ÖĞRENMESİ ADINA YAPILDI.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier # YENİ: RF eklendi
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. VERİ YÜKLEME
ornek = pd.read_csv("clinvar_conflicting.csv", low_memory=False)

# 2. ÖZELLİK MÜHENDİSLİĞİ
def check_trans(row):
    purines, pyrimidines = ['A', 'G'], ['C', 'T']
    ref, alt = str(row['REF']), str(row['ALT'])
    if len(ref) == 1 and len(alt) == 1:
        if (ref in purines and alt in purines) or (ref in pyrimidines and alt in pyrimidines):
            return 1 # Transition
        else: return 0 # Transversion
    return -1 # Indel

def get_exon_ratio(exon_str):
    try:
        if pd.isna(exon_str) or '/' not in str(exon_str): return 0
        parts = str(exon_str).split('/')
        return int(parts[0]) / int(parts[1])
    except: return 0

# Özellik ekleme
ornek['mutation_length'] = ornek['ALT'].str.len() - ornek['REF'].str.len()
ornek['is_transition'] = ornek.apply(check_trans, axis=1)
ornek['exon_ratio'] = ornek['EXON'].apply(get_exon_ratio)
ornek['average_af'] = ornek[['AF_ESP', 'AF_EXAC', 'AF_TGP']].mean(axis=1)

print("Yeni özellikler başarıyla eklendi!")

# 3. TEMİZLİK VE SADELEŞTİRME
missing_pct = ornek.isnull().sum() / len(ornek) * 100
too_many_nans = missing_pct[missing_pct > 50].index.tolist()
cols_to_exclude = ['CLNHGVS', 'CLNDISDB', 'CLNDN'] + too_many_nans
df_cleaned = ornek.drop(columns=cols_to_exclude)

# 4. EKSİK VERİ VE ENCODING
df_cleaned = df_cleaned.fillna(df_cleaned.median(numeric_only=True))
le = LabelEncoder()
for col in df_cleaned.select_dtypes(include='object').columns:
    df_cleaned[col] = le.fit_transform(df_cleaned[col].astype(str))

# 5. VERİYİ BÖLME
y = df_cleaned['CLASS']
X = df_cleaned.drop(columns='CLASS')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================================
# 6. MODEL 1: RANDOM FOREST (BAGGING)
# ==========================================================
print("\n--- RANDOM FOREST EĞİTİLİYOR ---")
# Daha önce Grid Search ile bulduğun en iyi parametreleri kullanıyoruz
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=2,
    class_weight='balanced', # Dengesiz veriyi dengelemek için kritik!
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Raporu:")
print(classification_report(y_test, y_pred_rf))

# ==========================================================
# 7. MODEL 2: XGBOOST (BOOSTING)
# ==========================================================
print("\n--- XGBOOST EĞİTİLİYOR ---")
weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    scale_pos_weight=weight,
    random_state=42
)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

print("XGBoost Raporu:")
print(classification_report(y_test, y_pred_xgb))

# 8. KIYASLAMA VE GÖRSELLEŞTİRME

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# RF Özellik Önem Sırası
feat_rf = pd.Series(rf_model.feature_importances_, index=X.columns)
feat_rf.nlargest(10).plot(kind='barh', ax=ax1, color='skyblue')
ax1.set_title("Random Forest: En Önemli 10 Özellik")

# XGBoost Özellik Önem Sırası
feat_xgb = pd.Series(xgb_model.feature_importances_, index=X.columns)
feat_xgb.nlargest(10).plot(kind='barh', ax=ax2, color='teal')
ax2.set_title("XGBoost: En Önemli 10 Özellik")

plt.tight_layout()
plt.show()