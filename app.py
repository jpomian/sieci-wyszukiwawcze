from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

# Load the CSV data into a Pandas DataFrame
df = pd.read_csv('books.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    # Get the search term from the form
    search_term = request.form.get('search_term')

    # Perform a case-insensitive search on the 'Book' column
    results = df[df['Book'].str.contains(search_term, case=False)]

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
