import pandas as pd
import matplotlib.pyplot as plt
import glob

# Read all CSV files and concatenate them into a single DataFrame
csv_files = glob.glob("results/*.csv")  # Update the path to your CSV files
all_data = pd.concat([pd.read_csv(f) for f in csv_files])

# Extract the dimension information from the problem column and create a new column 'Dimension'
all_data['Dimension'] = all_data['problem'].str.extract(r'D=(\d+)').astype(int)

# Remove the '%' sign and convert 'error' and 'error_std' columns to numeric
all_data['error'] = all_data['error'].str.replace('%', '').astype(float)
all_data['error_std'] = all_data['error_std'].str.replace('%', '').astype(float)

# Define the problem genres
genres = ["SimpleGaussian", "Camel", "QuadCamel"]

# Plotting the error and error_std for each genre and each algorithm
def plot_errors(error_bar=False):
    for genre in genres:
        genre_data = all_data[all_data['problem'].str.contains(genre)]
        dimensions = genre_data['Dimension'].unique()
        dimensions.sort()

        plt.figure(figsize=(14, 7))
        
        for integrator in genre_data['integrator'].unique():
            if integrator == 'Vegas':
                continue
            genre_integrator_data = genre_data[genre_data['integrator'] == integrator]
            errors = []
            error_stds = []
            
            for dim in dimensions:
                data = genre_integrator_data[genre_integrator_data['Dimension'] == dim]
                errors.append(data['error'].values[0])
                error_stds.append(data['error_std'].values[0])
            
            if error_bar:
                plt.errorbar(dimensions, errors, yerr=error_stds, label=integrator, capsize=5)
            else:
                plt.plot(dimensions, errors, label=integrator, marker='o')
        
        plt.title(f'Error and Error Std for {genre}')
        plt.xlabel('Dimension')
        plt.ylabel('Error (%)')
        plt.ylim([-5, 105])
        plt.legend()
        plt.grid(True)
        plt.savefig(f'figures\{genre}_error_plot.png')  # Save the error plot to a file
        plt.close()
        print(f'figure saved to figures\{genre}_error_plot.png')

def plot_times():
    # Time taken plot
    plt.figure(figsize=(14, 7))

    for genre in genres:
        genre_data = all_data[all_data['problem'].str.contains(genre)]
        dimensions = genre_data['Dimension'].unique()
        dimensions.sort()

        plt.figure(figsize=(14, 7))
    
        for integrator in genre_data['integrator'].unique():
            genre_integrator_data = genre_data[genre_data['integrator'] == integrator]
            times = []
            
            for dim in dimensions:
                data = genre_integrator_data[genre_integrator_data['Dimension'] == dim]
                if not data.empty:
                    times.append(float(data['time_taken'].values[0]))  # Ensure time_taken is float
                    
            plt.plot(dimensions, times, label=integrator, marker='o')

            plt.title(f'Time Taken for {genre} - {integrator}')
            plt.xlabel('Dimension')
            plt.ylabel('Time Taken (seconds)')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'figures\{genre}_time_plot_{integrator}.png')  # Save the time plot to a file
            plt.close()
            print(f'figure saved to figures\{genre}_time_plot_{integrator}.png')

if __name__ == '__main__':
    plot_errors()