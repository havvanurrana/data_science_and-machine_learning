# CLINVAR CONFLICTING VERİ SETİ 
## VERİ BİLİMİ ÇALIŞMASI VE MAKİNE ÖĞRENMESİ

### Bu çalışmada Bagging ve Bootstrap işlemleri kullanışmıştır: 

Bagging:
- Bu çalışmada Random Forest Tercih edilmiştir.

Boosting:
- XGBoost kullanılmıştır.
- Aşağıda kodumun çıktısı bulunuyor:
````
Yeni özellikler başarıyla eklendi!

--- RANDOM FOREST EĞİTİLİYOR ---
Random Forest Raporu:
              precision    recall  f1-score   support

           0       0.84      0.84      0.84      9768
           1       0.52      0.50      0.51      3270

    accuracy                           0.76     13038
   macro avg       0.68      0.67      0.68     13038
weighted avg       0.76      0.76      0.76     13038


--- XGBOOST EĞİTİLİYOR ---
XGBoost Raporu:
              precision    recall  f1-score   support

           0       0.89      0.71      0.79      9768
           1       0.46      0.73      0.56      3270

    accuracy                           0.71     13038
   macro avg       0.67      0.72      0.67     13038
weighted avg       0.78      0.71      0.73     13038

````
