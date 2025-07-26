# Conclusion
- Dalam  dataset  yang  digunakan  sebanyak  1  juta  baris  data  memiliki  jumlah  data 
antar kelas positif dan negatif yang berbeda pada variabel target, kelas negatif lebih 
banyak  daripada  kelas  positif  perbedaan  ini  memiliki  rasio  90:10  untuk  itu 
dilakukan teknik slicing pada kelas negatif supaya jumlah data seimbang. Peneliti 
tidak melakukan teknik SMOTE dikarenakan rasio yang memiliki perbedaan yang 
cukup jauh. Karena konsep dari  teknik SMOTE ini melakukan replica pada kelas 
tertentu yang berati melakukan duplikasi kembali terhadap data dan ini akan rentan 
terhadap model menjadi overfitting.
- Feature extraction yang digunakan adalah teknik SelectKBest dengan 5 fitur yang 
terbaik  di antaranya  fitur  month,  velocity_4w,  velocity_24h,   housing_status  dan 
credit_risk_score. Dan terdapat outlier pada credit_risk_score dan housing status 
sehingga dilakukan penghapusan outlier pada fitur tersebut.
- Metode deep learning lebih unggul dari  pada machine learning klasik hal ini karena 
metode deep learning umunnya sering digunakan pada hal yang kompleks mulai 
dari data terstruktur dan data yang tidak terstruktur.
Untuk rekomendasi dari peneliti disarankan untuk melakukan penyeimbangan jumlah 
data yang seragam pada masing-masing kelas variabel target yakni fitur fraud_bool untuk 
mendapatkan  hasil  yang  optimal  dan  mengurangi  terjadinya  overfitting  model.  Karena 
terkait cakupan dan batasan dalam penilitian, disarankan untuk melakukan hyperparamete r 
tuning kembali pada metode deep learning supaya hasilnya lebih optimal, hyperparameter 
tuning tersebut bisa dilakukan pada jumlah layer, neuron, dan pengaturan fungsi aktifasi
seperti ReLu dan sebagainya.
