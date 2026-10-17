import re
import nltk
import pymorphy3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

morph = pymorphy3.MorphAnalyzer()
stop_words = set(nltk.corpus.stopwords.words('russian'))

extra_stop_words = {'который', 'которая', 'которое', 'которые', 'этот', 'эта', 'это', 'эти', 'свой', 'своя', 'своё', 'свои', 'тот', 'та', 'то', 'те', 'так', 'как', 'что', 'чтобы', 'уже', 'еще', 'только', 'весь', 'вся', 'все', 'всё', 'быть', 'был', 'была', 'было', 'были', 'он', 'она', 'оно', 'они', 'я', 'ты', 'мы', 'вы', 'мой', 'твой', 'наш', 'ваш', 'его', 'ее', 'их', 'нее', 'него', 'ним', 'ней', 'мне', 'тебе', 'ему', 'ей', 'нами', 'вами', 'ими', 'обо', 'мной', 'тобой', 'собой', 'себя', 'себе', 'собой', 'собою'}
stop_words.update(extra_stop_words)

def normalize_text(text):
    """Нормализация текста: нижний регистр, удаление пунктуации, лемматизация, удаление стоп-слов."""
    
    text = text.lower()
   
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    
    words = text.split()
    # Лемматизация и фильтрация
    lemmatized = []
    for word in words:
        if word not in stop_words and len(word) > 2: # Игнорируем слова короче 3 букв
            normal_form = morph.parse(word)[0].normal_form
            lemmatized.append(normal_form)
            
    return " ".join(lemmatized)

def remove_frequent_patterns(sentences, top_n=5, min_len=10):
    """Удаляет предложения, которые встречаются чаще всего (шаблоны, заголовки) и слишком короткие."""
    # Считаем частоту точных совпадений предложений
    sentence_counts = Counter(sentences)
    
    # Находим наиболее частые паттерны (например, "ЧАСТЬ ПЕРВАЯ", "I", "II")
    common_patterns = [s for s, count in sentence_counts.most_common(top_n) if count > 1]
    
    filtered_sentences = []
    for s in sentences:
        # Убираем частые паттерны и слишком короткие предложения (мусор)
        if s not in common_patterns and len(s.split()) >= min_len:
            filtered_sentences.append(s)
            
    return filtered_sentences

def get_top_words_per_cluster(vectorizer, kmeans, n_words=10):
    """Получает самые важные слова для каждого кластера."""
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    cluster_keywords = {}
    for i in range(kmeans.n_clusters):
        top_words = [terms[ind] for ind in order_centroids[i, :n_words]]
        cluster_keywords[i] = top_words
    return cluster_keywords

# --- 2. Основной процесс ---
if __name__ == "__main__":
    INPUT_FILE = "Part1_Ch1_2.txt"  # Укажите имя вашего файла
    
    print("1. Чтение файла...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    print("2. Разбиение на предложения...")
    # Используем nltk для корректного разбиения русского текста на предложения
    sentences = nltk.sent_tokenize(raw_text, language='russian')
    print(f"   Найдено предложений: {len(sentences)}")

    print("3. Удаление частых паттернов и мусора...")
    clean_sentences = remove_frequent_patterns(sentences, top_n=10, min_len=8)
    print(f"   Осталось предложений после очистки: {len(clean_sentences)}")

    print("4. Нормализация текста (лемматизация может занять время)...")
    normalized_sentences = [normalize_text(s) for s in clean_sentences]
    
    # Фильтруем пустые строки, которые могли получиться после нормализации
    valid_indices = [i for i, s in enumerate(normalized_sentences) if len(s.split()) > 3]
    final_sentences = [clean_sentences[i] for i in valid_indices]
    final_normalized = [normalized_sentences[i] for i in valid_indices]
    print(f"   Годных предложений для кластеризации: {len(final_sentences)}")

    print("5. Векторизация (TF-IDF)...")
    # Ограничиваем количество признаков для скорости и качества кластеризации
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(final_normalized)

    print("6. Кластеризация (K-Means)...")
    num_clusters = 5  # Можно изменить количество кластеров
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X)

    print("7. Анализ результатов...")
    cluster_keywords = get_top_words_per_cluster(vectorizer, kmeans, n_words=8)
    
    for cluster_id, keywords in cluster_keywords.items():
        print(f"\n--- Кластер {cluster_id} (Ключевые слова: {', '.join(keywords)}) ---")
        # Выводим 3 случайных предложения из этого кластера для примера
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        sample_indices = np.random.choice(cluster_indices, size=min(3, len(cluster_indices)), replace=False)
        for idx in sample_indices:
            print(f"  - {final_sentences[idx]}")



print("8. Анализ и визуализация паттернов кластеров...")

# --- А. Находим самые репрезентативные предложения для каждого кластера ---
# (Те, что ближе всего к математическому центру кластера)
print("\n--- САМЫЕ ТИПИЧНЫЕ ПРЕДЛОЖЕНИЯ ДЛЯ КАЖДОГО КЛАСТЕРА ---")
for cluster_id in range(num_clusters):
    print(f"\n🔹 КЛАСТЕР {cluster_id} (Ключевые слова: {', '.join(cluster_keywords[cluster_id][:8])})")
    
    # Находим индекс центроида (центра) этого кластера
    centroid = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
    
    # Находим все предложения, принадлежащие этому кластеру
    cluster_indices = np.where(cluster_labels == cluster_id)[0]
    cluster_vectors = X[cluster_indices]
    
    # Считаем косинусное сходство каждого предложения с центром кластера
    similarities = cosine_similarity(centroid, cluster_vectors)[0]
    
    # Берем топ-3 предложения с наибольшим сходством (самые типичные)
    top_local_indices = similarities.argsort()[-3:][::-1]
    top_global_indices = cluster_indices[top_local_indices]
    
    for i, idx in enumerate(top_global_indices):
        print(f"  {i+1}. {final_sentences[idx]}")


# --- Б. Визуализация: Облака слов для каждого кластера ---
print("Построение точных графиков топ-слов для каждого кластера...")

# Получаем названия признаков (слов) из векторизатора
feature_names = vectorizer.get_feature_names_out()
num_clusters = kmeans.n_clusters
top_n_words = 15 # Сколько слов показывать на графике

# Создаем сетку графиков (например, 2 строки, 3 колонки для 5 кластеров)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i in range(num_clusters):
    # Находим все предложения, принадлежащие текущему кластеру
    cluster_indices = np.where(cluster_labels == i)[0]
    
    # Усредняем TF-IDF векторы всех предложений в этом кластере
    # Это даст нам "средний профиль" кластера
    cluster_vector = X[cluster_indices].mean(axis=0)
    
    # Преобразуем в массив и сортируем по убыванию
    cluster_array = np.asarray(cluster_vector).flatten()
    sorted_indices = cluster_array.argsort()[::-1]
    
    # Берем топ-N слов и их веса
    top_words = [feature_names[idx] for idx in sorted_indices[:top_n_words]]
    top_scores = [cluster_array[idx] for idx in sorted_indices[:top_n_words]]
    
    # Строим горизонтальный график
    ax = axes[i]
    ax.barh(top_words[::-1], top_scores[::-1], color=plt.cm.viridis(i / num_clusters))
    ax.set_title(f"Кластер {i} (Тема: {', '.join(cluster_keywords[i][:3])})", fontsize=12, fontweight='bold')
    ax.set_xlabel("TF-IDF Weight")
    
    # Убираем лишний фон
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Скрываем пустой график, если кластеров меньше, чем ячеек сетки
for j in range(num_clusters, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('cluster_top_words_barplot.png', dpi=300)
plt.show()