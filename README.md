# NLP_Final
# 🍽️ Metin Benzerliği Projesi – NLP Final

Bu projede yemek tarifi metinlerinden oluşan veri seti üzerinde TF-IDF ve Word2Vec modelleri ile benzerlik analizi yapılmıştır.  
Word2Vec modelleri CBOW ve SkipGram mimarileriyle eğitilmiş, lemmatized ve stemmed veriler ayrı ayrı değerlendirilmiştir.

---

## 📁 Kullanılan Dosyalar

- `lemmatized_data.csv` – Lemmatize edilmiş tarif verisi  
- `stemmed_data.csv` – Stemmed (kök alınmış) tarif verisi  
- `lemma_modelleri/` – 8 adet lemmatized Word2Vec modeli  
- `stem_modelleri/` – 8 adet stemmed Word2Vec modeli  

---

## 🚀 Çalıştırma Adımları

1. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install pandas gensim scikit-learn
2. Giriş metni alın
 ```
import pandas as pd

# Lemmatized versiyondan giriş metnini seçiyoruz
df_lemma = pd.read_csv(r"final_2.odev\\lemmatized_data.csv")

# Örneğin 0. satırı alalım
query_text = df_lemma.iloc[0]['text']

print("Giriş Metni:")
print(query_text)
```
3. Modellerin sırasını, skorunu ve tariflerini yazdır
```
import os
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Giriş verisi (lemmatized sabit kalacak çünkü metin oradan alınacak)
df = pd.read_csv(r"final_2.odev\lemmatized_data.csv")
query = df.iloc[0]['text'].split()

def get_average_vector(model, tokens):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)

# Model klasörleri
model_folders = ["lemma_modelleri", "stem_modelleri"]

for folder in model_folders:
    print(f"\n📂 Klasör: {folder}\n{'-'*40}")
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".model"):
            model_path = os.path.join(folder, filename)
            model = Word2Vec.load(model_path)

            query_vector = get_average_vector(model, query).reshape(1, -1)

            doc_vectors = []
            for text in df['text']:
                tokens = text.split()
                vec = get_average_vector(model, tokens)
                doc_vectors.append(vec)

            doc_vectors = np.array(doc_vectors)
            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            top_5_indices = similarities.argsort()[::-1][1:6]

            print(f"\n🧠 Model: {filename}")
            for i, idx in enumerate(top_5_indices):
                print(f"{i+1}. Skor: {similarities[idx]:.4f}")
                print(f"   Tarif: {df.iloc[idx]['text']}\n")

```
4. En iyi 5 skoru alan indexler ile
```
import pandas as pd
import os
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from itertools import combinations

# Giriş verisi
df = pd.read_csv(r"final_2.odev\\lemmatized_data.csv")
query = df.iloc[0]['text'].split()

def get_average_vector(model, tokens):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Model klasörleri
model_paths = []
model_names = []
top5_indices_dict = {}

# Tüm modelleri yükle ve top 5 indexlerini al
for folder in ["lemma_modelleri", "stem_modelleri"]:
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".model"):
            model_path = os.path.join(folder, filename)
            model = Word2Vec.load(model_path)

            query_vector = get_average_vector(model, query).reshape(1, -1)
            doc_vectors = [get_average_vector(model, text.split()) for text in df['text']]
            doc_vectors = np.array(doc_vectors)

            similarities = cosine_similarity(query_vector, doc_vectors)[0]
            top5 = set(similarities.argsort()[::-1][1:6])  # ilk 5 sonucu al

            model_names.append(filename)
            top5_indices_dict[filename] = top5

jaccard_matrix = pd.DataFrame(index=model_names, columns=model_names)

for m1, m2 in combinations(model_names, 2):
    set1 = top5_indices_dict[m1]
    set2 = top5_indices_dict[m2]
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    score = intersection / union
    jaccard_matrix.loc[m1, m2] = round(score, 2)
    jaccard_matrix.loc[m2, m1] = round(score, 2)

# Kendisiyle kıyaslamalar 1.00 olarak ayarlanır
for name in model_names:
    jaccard_matrix.loc[name, name] = 1.00

# Sayısal çeviri ve görüntü
jaccard_matrix = jaccard_matrix.astype(float)
display(jaccard_matrix.round(2))
```

5. TF-IDF benzerlik hesaplama yapıldı
```
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['text'])
query_vector = vectorizer.transform([query_text])
similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
top_5 = similarities.argsort()[::-1][1:6]
```
6. Word2Vec ortalama vector hesaplama yapıldı
```
def get_average_vector(model, tokens):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)
```
