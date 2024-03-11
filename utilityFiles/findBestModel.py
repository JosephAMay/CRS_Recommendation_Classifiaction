#!/usr/bin/env python3
import os
import csv

# Get the current directory
current_directory = os.getcwd()

# Initialize variables to store the highest average accuracy and its corresponding directory
highest_average_acc = 0.0
directory_with_highest_average_acc = ''

# Iterate through each directory in the current directory
for directory in os.listdir(current_directory):
    if os.path.isdir(directory):
        # Initialize variables to calculate average accuracy for the current directory
        total_val_acc = 0.0
        num_files = 0

        # Check if metrics.csv exists in the current directory
        metrics_file_path = os.path.join(current_directory, directory, 'metrics.csv')
        if os.path.exists(metrics_file_path):
            print("Processing directory:", directory)
            # Open and read metrics.csv file
            with open(metrics_file_path, 'r') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)  # Skip the header row
                for row in csv_reader:
                    val = row[0]
                    if val != '':
                        val_acc = float(val)  # Assuming val_acc is in the first column
                        total_val_acc += val_acc
                    num_files += 1
            
            # Calculate average accuracy for the current directory
            if num_files > 0:
                average_acc = total_val_acc / num_files

                # Update highest average accuracy and corresponding directory if necessary
                if average_acc > highest_average_acc:
                    highest_average_acc = average_acc
                    directory_with_highest_average_acc = directory

# Output the directory with the highest average accuracy
print("Directory with highest average accuracy:", directory_with_highest_average_acc)
print("Highest average accuracy:", highest_average_acc)

