# CLINVAR CONFLICTING VERİ SETİ 
## VERİ BİLİMİ ÇALIŞMASI VE MAKİNE ÖĞRENMESİ

### Bu çalışmada Bagging ve Bootstrap işlemleri kullanışmıştır: 

Bagging:
- Bu çalışmada Random Forest Tercih edilmiştir.

Boosting:
- XGBoost kullanılmıştır.
- Aşağıda parametreler vardır 
````
Yeni özellikler başarıyla eklendi!
 eğitim seti boyutu:(52150, 42)
test seti boytu: (13038, 42)
Fitting 3 folds for each of 12 candidates, totalling 36 fits

en iyi ayarlar: {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 100}
XGBoost Doğruluk Oranı: 0.7903819604233778

Detaylı Rapor:
               precision    recall  f1-score   support

           0       0.81      0.94      0.87      9768
           1       0.66      0.34      0.45      3270

    accuracy                           0.79     13038
   macro avg       0.73      0.64      0.66     13038
weighted avg       0.77      0.79      0.77     13038

Yeni XGBoost (Dengelenmiş) Raporu:
              precision    recall  f1-score   support

           0       0.89      0.71      0.79      9768
           1       0.46      0.74      0.57      3270

    accuracy                           0.72     13038
   macro avg       0.68      0.72      0.68     13038
weighted avg       0.78      0.72      0.74     13038

````