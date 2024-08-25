import os
import pandas as pd

# Directory containing your CSV files
directory_path = 'test_results/fifth_run/genres'  # Replace with your directory path
# Automatically load all CSV files in the directory
csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

# Function to extract problem name from filename (assuming the filename reflects the problem name)
def extract_problem_name(file_name):
    return file_name.replace('_results.csv', '')

def significant_overlap(mean1, std1, mean2, std2, tolerance=0.3):
    return not (mean1 + std1 * tolerance < mean2 - std2 * tolerance or mean2 + std2 * tolerance < mean1 - std1 * tolerance)

def is_significantly_better(best_mean, best_std, others, std_tolerance=2):
    # Determine the smallest error bar among the others
    min_other_std = min(std for _, std in others)
    
    # If the best integrator's error bar is significantly larger than the smallest other error bar, do not highlight
    if best_std > std_tolerance * min_other_std:  # Adjust this factor as needed
        return False
    
    # Ensure that the best mean is closest to zero and doesn't significantly overlap with others
    for mean, std in others:
        if significant_overlap(best_mean, best_std, mean, std):
            if abs(best_mean) > abs(mean):
                return False
    return True


# Function to generate LaTeX table in landscape mode with highlighted best integrator
def generate_latex_table(df, problem_name, error_sig_digits=5, time_sig_digits=4):
    # Get unique integrators in the data
    integrators = df['integrator'].unique()
    
    # Start the LaTeX table in landscape mode
    table_str = (
        f"\\begin{{landscape}}\n"
        f"\\begin{{table}}[H]\n\\centering\n\\caption{{Comparison of integrators on {problem_name} benchmark problem}}\n"
        "\\renewcommand{\\arraystretch}{1.5} % Increases the row height\n"
        "\\begin{tabular}{l" + "c" * len(integrators) + "|c}\n\\hline\n"
        "\\textbf{Dimension} & " + " & ".join([f"\\textbf{{{integrator}}}" for integrator in integrators]) + " & \\textbf{Time (s)} \\\\ \\cline{2-" + str(len(integrators)+1) + "}\n"
    )
    
    # Sort dimensions based on numerical order extracted from the problem column
    df['dimension'] = df['problem'].str.extract(r'D=(\d+)').astype(int)
    df = df.sort_values(by='dimension')
    
    for dimension in df['dimension'].unique():
        sub_df = df[df['dimension'] == dimension]
        error_strs = []
        time_strs = []
        
        best_mean = None
        best_std = None
        best_integrator = None

        for integrator in integrators:
            if not sub_df[sub_df['integrator'] == integrator].empty:
                integrator_data = sub_df[sub_df['integrator'] == integrator].iloc[0]
                mean_error = float(integrator_data['error'].replace('%', ''))
                error_std = float(integrator_data['error_std'].replace('%', ''))
                time = f"{float(integrator_data['time_taken']):.{time_sig_digits}g}"

                # If this is the first integrator or it has a mean error closer to zero and a smaller or similar error bar, select it as the best
                if best_mean is None or (abs(mean_error) < abs(best_mean) and error_std <= best_std):
                    best_mean = mean_error
                    best_std = error_std
                    best_integrator = integrator

                error_strs.append((mean_error, error_std))
                time_strs.append(time)
        
        highlighted_error_strs = []
        for idx, (mean_error, error_std) in enumerate(error_strs):
            integrator = integrators[idx]
            if integrator == best_integrator and is_significantly_better(best_mean, best_std, [(m, s) for i, (m, s) in enumerate(error_strs) if i != idx]):
                highlighted_error_strs.append(f"\\textbf{{{mean_error:.{error_sig_digits}g}\\% ± {error_std:.{error_sig_digits}g}\\%}}")
            else:
                highlighted_error_strs.append(f"{mean_error:.{error_sig_digits}g}\\% ± {error_std:.{error_sig_digits}g}\\%")
        
        time_combined = " / ".join(time_strs)
        table_str += f"{dimension} & " + " & ".join(highlighted_error_strs) + f" & {time_combined} \\\\\n"
    
    table_str += "\\hline\n\\end{tabular}\n\\end{table}\n\\end{landscape}\n"
    return table_str

# Process each CSV file in the directory
for file_name in csv_files:
    file_path = os.path.join(directory_path, file_name)
    problem_name = extract_problem_name(file_name)
    df = pd.read_csv(file_path)
    table = generate_latex_table(df, problem_name, error_sig_digits=4, time_sig_digits=3)
    print(table)