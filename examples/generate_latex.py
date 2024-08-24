import pandas as pd

# Function to generate LaTeX table
def generate_latex_table(df, problem_name):
    table_str = (
        f"\\begin{{table}}[H]\n\\centering\n\\caption{{Comparison of fitting for {problem_name}}}\n"
        "\\renewcommand{\\arraystretch}{1.5} % Increases the row height\n"
        "\\begin{tabular}{lccc|c}\n\\hline\n"
        "\\textbf{Dimension} & \\multicolumn{2}{c|}{\\textbf{Relative Error (\\%)}} & \\textbf{Time (s)} & \\textbf{n\\_evals} \\\\ \\cline{2-3}\n"
        " & \\textbf{Iterative GP} & \\textbf{Even Sample GP} & & \\\\ \\hline\n"
    )
    
    # Sort dimensions based on numerical order extracted from the problem column
    df['dimension'] = df['problem'].str.extract(r'D=(\d+)').astype(int)
    df = df.sort_values(by='dimension')
    
    for dimension in df['dimension'].unique():
        sub_df = df[df['dimension'] == dimension]
        iterative_gp = sub_df[sub_df['integrator'] == 'Iterative GP'].iloc[0]
        even_sample_gp = sub_df[sub_df['integrator'] == 'Even Sample GP'].iloc[0]
        n_evals = iterative_gp['n_evals']  # Assuming n_evals is the same for both
        
        iterative_error = f"{float(iterative_gp['error'].replace('%', '')):.5g}\\%"
        even_sample_error = f"{float(even_sample_gp['error'].replace('%', '')):.5g}\\%"
        iterative_error_std = f"{float(iterative_gp['error_std'].replace('%', '')):.5g}\\%"
        even_sample_error_std = f"{float(even_sample_gp['error_std'].replace('%', '')):.5g}\\%"
        
        # Format time to 4 significant digits
        time_iterative = f"{float(iterative_gp['time_taken']):.4g}"
        time_even_sample = f"{float(even_sample_gp['time_taken']):.4g}"
        
        table_str += f"{dimension} & {iterative_error} ± {iterative_error_std} & {even_sample_error} ± {even_sample_error_std} & {time_iterative} / {time_even_sample} & {n_evals} \\\\\n"
    
    table_str += "\\hline\n\\end{tabular}\n\\end{table}\n"
    return table_str

# Load the CSV files
files = {
    'C0function': 'test_results/ablation_iterative_fitting/C0function_results.csv',
    'Oscillatory': 'test_results/ablation_iterative_fitting/Oscillatory_results.csv',
    'ProductPeak': 'test_results/ablation_iterative_fitting/ProductPeak_results.csv',
    'Ripple': 'test_results/ablation_iterative_fitting/Ripple_results.csv'
}

# Generate and print LaTeX tables for each problem
for problem, filepath in files.items():
    df = pd.read_csv(filepath)
    table = generate_latex_table(df, problem)
    print(table)
