from flask import Flask, render_template_string, request, send_file
import pandas as pd

app = Flask(__name__)

# Constants
BUCKET_NAME = "bharatpe_ml_image_bucket_2"
GCS_FOLDER = "kyc_store_merchant/kyc_sept24_data2"
BASE_URL = f"https://storage.cloud.google.com/{BUCKET_NAME}/{GCS_FOLDER}/"

# Load the initial data from CSV
file_path = "filtered_data.csv"  # Replace with your CSV file path
df = pd.read_csv(file_path)

# Dynamically construct the image URLs using the bucket and folder constants
df["query_image_url"] = df["query_file_name"].apply(lambda x: BASE_URL + x)
df["similar_image_url"] = df["similar_file_name"].apply(lambda x: BASE_URL + x)

# # Load the initial data from CSV
# file_path = "similar_data_all.csv"  # Replace with your CSV file path
# df = pd.read_csv(file_path)

# # Ensure the 'labels' column exists in the DataFrame
# if 'labels' not in df.columns:
#     df['labels'] = ''  # Add the column with default empty values or a default value like 0

# Dynamically construct the image URLs using the bucket and folder constants
df["query_image_url"] = df["query_file_name"].apply(lambda x: BASE_URL + x)
df["similar_image_url"] = df["similar_file_name"].apply(lambda x: BASE_URL + x)


# HTML Template for the editable form
html_template = """
<html>
<head>
    <title>Image Label Validation</title>
    <style>
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: center; border: 1px solid #ddd; }
        img { width: 150px; height: auto; }
    </style>
</head>
<body>
    <h2>Image Label Validation</h2>
    <form method="POST" action="/submit_labels">
        <table>
            <thead>
                <tr>
                    <th>Merchant ID (Query)</th>
                    <th>Merchant ID (Similar)</th>
                    <th>Query Image</th>
                    <th>Similar Image</th>
                    <th>labels</th>
                    <th>Score</th>
                </tr>
            </thead>
            <tbody>
"""

# Loop through DataFrame and create rows for each image with editable fields
for index, row in df.iterrows():
    query_image_url = row['query_image_url']
    similar_image_url = row['similar_image_url']
    label_value = row['labels']
    score_value = row['score']

    html_template += f"""
        <tr>
            <td>{row['merchant_id_x']}</td>
            <td>{row['merchant_id_y']}</td>
            <td><img src="{query_image_url}" alt="Query Image"></td>
            <td><img src="{similar_image_url}" alt="Similar Image"></td>
            <td><input type="text" name="label_{index}" value="{label_value}" size="5"></td>
            <td><input type="text" name="score_{index}" value="{score_value}" size="5"></td>
        </tr>
    """

html_template += """
            </tbody>
        </table>
        <br>
        <input type="submit" value="Submit Labels and Scores">
    </form>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(html_template)

@app.route('/submit_labels', methods=["POST"])
def submit_labels():
    updated_data = {}

    # Collect updated label and score values from the form
    for index, row in df.iterrows():
        label_key = f"label_{index}"
        score_key = f"score_{index}"
        updated_data[label_key] = request.form.get(label_key, row['labels'])  # Default to original label if not updated
        updated_data[score_key] = request.form.get(score_key, row['score'])  # Default to original score if not updated

    # Update the DataFrame with the new labels and scores
    for index, row in df.iterrows():
        label_key = f"label_{index}"
        score_key = f"score_{index}"
        new_label = updated_data.get(label_key, row['labels'])
        new_score = updated_data.get(score_key, row['score'])
        df.at[index, 'labels'] = new_label  # Update the label
        df.at[index, 'score'] = new_score  # Update the score

    # Generate the updated HTML with the new labels and scores
    updated_html_template = """
    <html>
    <head>
        <title>Updated Image Labels</title>
        <style>
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 10px; text-align: center; border: 1px solid #ddd; }
            img { width: 150px; height: auto; }
        </style>
    </head>
    <body>
        <h2>Updated Image Label Validation</h2>
        <table>
            <thead>
                <tr>
                    <th>Merchant ID (Query)</th>
                    <th>Merchant ID (Similar)</th>
                    <th>Query Image</th>
                    <th>Similar Image</th>
                    <th>Updated Label</th>
                    <th>Updated Score</th>
                </tr>
            </thead>
            <tbody>
    """

    # Loop through DataFrame again to create the updated HTML table
    for index, row in df.iterrows():
        updated_html_template += f"""
            <tr>
                <td>{row['merchant_id_x']}</td>
                <td>{row['merchant_id_y']}</td>
                <td><img src="{row['query_image_url']}" alt="Query Image"></td>
                <td><img src="{row['similar_image_url']}" alt="Similar Image"></td>
                <td>{row['labels']}</td>
                <td>{row['score']}</td>
            </tr>
        """

    updated_html_template += """
            </tbody>
        </table>
    </body>
    </html>
    """

    # Save the updated HTML to a file
    output_file = "2lakh_img.html"
    with open(output_file, 'w') as f:
        f.write(updated_html_template)

    return send_file(output_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True,port=8000)
