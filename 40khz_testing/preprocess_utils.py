import os
import csv
import re

def read_csv_file(file_path, num_points_to_read=None):
    """
    Reads a CSV file and returns its content as a list of floats.
    
    Handles CSV files where each line contains one number followed by a comma.

    Args:
        file_path (str): Path to the CSV file.
        num_points_to_read (int, optional): Maximum number of points to read.

    Returns:
        list of float: The time series data from the file.
    """
    time_series = []
    try:
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                for item in row:
                    item = item.strip()
                    if item:  # Ignore empty strings
                        try:
                            number = float(item)
                            time_series.append(number)
                            if num_points_to_read and len(time_series) >= num_points_to_read:
                                return time_series
                        except ValueError:
                            print(f"Non-numeric value '{item}' found in file '{os.path.basename(file_path)}'. Skipping this value.")
        return time_series
    except Exception as e:
        print(f"Error reading file '{os.path.basename(file_path)}': {e}")
        return None

def calculate_median(data):
    """
    Calculates the median of a list of numbers.

    Args:
        data (list of float): The data to calculate the median for.

    Returns:
        float: The median value.
    """
    sorted_data = sorted(data)
    n = len(sorted_data)
    if n == 0:
        return 0  # Default median for empty list
    mid = n // 2
    if n % 2 == 0:
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else:
        return sorted_data[mid]

def clean_data(data):
    """
    Cleans a time series by replacing outlier values with the previous valid value.
    
    Outliers are defined as values 50 times lower or higher than the median.
    The first value, if outlier, is replaced with the median.

    Args:
        data (list of float): Time series data.

    Returns:
        list of float: Cleaned time series data.
    """
    if not data:
        return None

    # Calculate median
    median = calculate_median(data)

    # Define thresholds
    if median == 0:
        # Avoid division by zero; use standard deviation-based thresholds
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        std = variance ** 0.5
        upper_threshold = mean + (std * 10)  # Adjust factor as needed
        lower_threshold = mean - (std * 10)
    else:
        upper_threshold = median * 50
        lower_threshold = median / 50

    cleaned_data = []
    previous_valid = None

    for idx, value in enumerate(data):
        if lower_threshold <= value <= upper_threshold:
            cleaned_data.append(value)
            previous_valid = value
        else:
            if previous_valid is not None:
                cleaned_data.append(previous_valid)
            else:
                # If it's the first value and outlier, replace with median
                cleaned_data.append(median)

    return cleaned_data

def parse_and_preprocess_csv(directory, num_points_to_read=2000):
    """
    Parses and preprocesses CSV files in a given directory.

    Groups files by the number at the end of the filename and by class_id.

    Args:
        directory (str): Path to the directory containing CSV files.
        num_points_to_read (int): Maximum number of points to read from each file.

    Returns:
        dict: Nested dictionary where the first key is the number at the end of the filename,
              the second key is the class_id (filename before the number),
              and the value is a list of cleaned time series lists.
    """
    # Regular expression to extract class_id and number from filename
    filename_pattern = re.compile(r"^(.*?)(\d+)\.csv$")

    grouped_data = {}

    for filename in os.listdir(directory):
        
        if filename.endswith('.csv'):
            
            match = filename_pattern.match(filename)

            if not match:
                print(f"Filename '{filename}' does not match the pattern 'class_id<number>.csv'. Skipping.")
                continue

            class_id, number = match.groups()
            class_id = class_id.replace(" ", "")  # Remove all spaces from the class name
            try:
                number = int(number)  # Convert number to integer for consistency
            except ValueError:
                print(f"Invalid number '{number}' in filename '{filename}'. Skipping.")
                continue

            file_path = os.path.join(directory, filename)
            time_series = read_csv_file(file_path, num_points_to_read)
            if time_series is not None:
                cleaned_series = clean_data(time_series)
                if cleaned_series and len(cleaned_series) > 0:
                    if number not in grouped_data:
                        grouped_data[number] = {}
                    if class_id not in grouped_data[number]:
                        grouped_data[number][class_id] = []
                    grouped_data[number][class_id].append(cleaned_series)
                else:
                    print(f"Cleaned data for file '{filename}' is empty after preprocessing. Skipping.")
            else:
                print(f"Failed to read or preprocess file '{filename}'. Skipping.")

    return grouped_data