from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
import Levenshtein
from deep_translator import GoogleTranslator

app = Flask(__name__)

df = pd.read_csv('books.csv')

df['Description'] = df['Description'].fillna('') # Uzupełnij wartości NaN pustym ciągiem znaków
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Description'])

num_topics = 10  # Można optymalizować
lsa = TruncatedSVD(n_components=num_topics)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

selected_features = ['Book', 'Author', 'Genres']
feature_matrix = df[selected_features].fillna('').values
knn_model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
knn_model.fit(tfidf_matrix)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    search_term = request.form.get('search_term')

    df['Levenshtein_distance'] = df['Book'].apply(lambda x: Levenshtein.distance(x.lower(), search_term.lower()))

    search_term_set = set(search_term.lower().split())
    df['Jaccard_index'] = df['Book'].apply(lambda x: len(set(x.lower().split()) & search_term_set) / len(set(x.lower().split()) | search_term_set))

    search_term_set = set(search_term.lower())
    df['Dice_coefficient'] = df['Book'].apply(lambda x: 2 * len(search_term_set & set(x.lower())) / (len(search_term_set) + len(set(x.lower()))))

    search_tfidf = tfidf_vectorizer.transform([search_term])
    search_lsa = lsa.transform(search_tfidf)

    df['LSA_similarity'] = linear_kernel(search_lsa, lsa_matrix).flatten()

    exact_match_results = df[df['Book'].str.contains(search_term, case=False)]

    cosine_similarities = linear_kernel(search_tfidf, tfidf_matrix).flatten()

    top_indices = cosine_similarities.argsort()[:-6:-1]

    results = df.iloc[top_indices]

    related_results = df[~df['Book'].isin(results['Book'])].sample(5)

    return render_template('results.html', results=exact_match_results, related_results=related_results, searched_term=search_term)

@app.route('/books/<int:book_id>')
def book_detail(book_id):
    book_details = df[df['id'] == book_id].squeeze()

    if book_details is not None:

        book_desc_translated=GoogleTranslator(source='auto', target='pl').translate(book_details['Description'])

        query_features = book_details[selected_features].fillna('').values
        query_tfidf = tfidf_vectorizer.transform(query_features)
        _, indices = knn_model.kneighbors(query_tfidf, n_neighbors=5)

        indices = [i for i in indices[0] if i != book_id and df.iloc[i]['Book'] != book_details['Book']]

        similar_books = df.iloc[indices]

        return render_template('book_detail.html', book_details=book_details, book_trans=book_desc_translated, similar_books=similar_books)
    else:
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)