import os
import pandas as pd

# Specify the directory containing the CSV files
directory = 'test_results/fifth_run'  

# Load the CSV files
file_paths = [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.endswith('.csv')]

# Load each file into a dataframe and store in a dictionary
data = {f"{i+2}D": pd.read_csv(file) for i, file in enumerate(file_paths)}

# Define the problem genres
problem_genres = {
    'SimpleGaussian': [],
    'QuadCamel': [],
    'Camel': [], 
    'ExponentialProductProblem': [],
    'QuadraticProblem': [],
    'Ripple': [],
    'Oscillatory': [],
    'ProductPeak': [],
    'C0function': [],
    'CornerPeak': [],
    'Discontinuous': []
}

# Function to categorize the data by problem genre
def categorize_by_genre(data, problem_genres):
    for dim, df in data.items():
        for problem in problem_genres.keys():
            genre_data = df[df['problem'].str.contains(problem)]
            if not genre_data.empty:
                problem_genres[problem].append(genre_data)
    return problem_genres

# Categorize the data by problem genre
categorized_data = categorize_by_genre(data, problem_genres)

# Concatenate the dataframes for each problem genre and save to a new file
for genre, df_list in categorized_data.items():
    genre_df = pd.concat(df_list, ignore_index=True)
    genre_df.to_csv(os.path.join(directory, f'{genre}_results.csv'), index=False)

print("Files have been successfully rearranged and saved by problem genre.")
