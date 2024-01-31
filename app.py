from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import linear_kernel
from sklearn.neighbors import NearestNeighbors
import Levenshtein  # Import the Levenshtein module

app = Flask(__name__)

# Load the CSV data into a Pandas DataFrame
df = pd.read_csv('books.csv')

# Preprocess the book descriptions for TF-IDF
df['Description'] = df['Description'].fillna('')  # Fill NaN values with an empty string
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Description'])

# Perform Latent Semantic Analysis (LSA)
num_topics = 10  # You can adjust the number of topics based on your preferences
lsa = TruncatedSVD(n_components=num_topics)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

# Fit k-NN model on selected features
selected_features = ['Book', 'Author', 'Genres']
feature_matrix = df[selected_features].fillna('').values
knn_model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
knn_model.fit(tfidf_matrix)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    # Get the search term from the form
    search_term = request.form.get('search_term')

    # Calculate Levenshtein distance for each book title
    df['Levenshtein_distance'] = df['Book'].apply(lambda x: Levenshtein.distance(x.lower(), search_term.lower()))

    # Calculate Jaccard index for each book title
    search_term_set = set(search_term.lower().split())
    df['Jaccard_index'] = df['Book'].apply(lambda x: len(set(x.lower().split()) & search_term_set) / len(set(x.lower().split()) | search_term_set))

    # Calculate Sørensen–Dice coefficient for each book title
    search_term_set = set(search_term.lower())
    df['Dice_coefficient'] = df['Book'].apply(lambda x: 2 * len(search_term_set & set(x.lower())) / (len(search_term_set) + len(set(x.lower()))))

    # Calculate LSA similarity for each book
    search_tfidf = tfidf_vectorizer.transform([search_term])
    search_lsa = lsa.transform(search_tfidf)

    df['LSA_similarity'] = linear_kernel(search_lsa, lsa_matrix).flatten()

    # Perform a case-insensitive search on the 'Book' column
    exact_match_results = df[df['Book'].str.contains(search_term, case=False)]

    # Calculate cosine similarity between search term and book descriptions
    cosine_similarities = linear_kernel(search_tfidf, tfidf_matrix).flatten()

    # Get the top 5 book indices with highest similarity scores
    top_indices = cosine_similarities.argsort()[:-6:-1]

    # Retrieve the corresponding book details
    results = df.iloc[top_indices]

    # For related works, exclude the ones that are already in the main results
    related_results = df[~df['Book'].isin(results['Book'])].sample(5)

    return render_template('results.html', results=exact_match_results, related_results=related_results, searched_term=search_term)


@app.route('/books/<int:book_id>')
def book_detail(book_id):
    book_details = df[df['id'] == book_id].squeeze()

    if book_details is not None:
        # Use k-NN to find similar books
        query_features = book_details[selected_features].fillna('').values
        query_tfidf = tfidf_vectorizer.transform(query_features)
        _, indices = knn_model.kneighbors(query_tfidf, n_neighbors=5)

        # Retrieve similar books based on k-NN results
        similar_books = df.iloc[indices[0]]

        return render_template('book_detail.html', book_details=book_details, similar_books=similar_books)
    else:
        # Redirect to home if book_id is not found
        return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)