import re
from collections import defaultdict

# Function to process each line and extract error messages
def process_line(line):
    error_match = re.search(r"Error: (.*?)(?: at position|: …|$)", line)
    return error_match.group(1).strip() if error_match else None

# Function to categorize errors
def categorize_errors(file_path):
    error_summary = defaultdict(int)
    with open(file_path, 'r') as file:
        for line in file:
            error_message = process_line(line)
            if error_message:
                # Generalize the error message by removing specific details
                generalized_error = re.sub(r' at position \d+', '', error_message)
                generalized_error = re.sub(r':\s*….*$', '', generalized_error)
                error_summary[generalized_error] += 1
    return error_summary

# File path (replace with your actual file path)
file_path = './katex-errors.txt'

# Get the summary of errors
error_summary = categorize_errors(file_path)

# Print the results
for error, count in sorted(error_summary.items(), key=lambda x: x[1], reverse=True):
    print(f"{count} occurrences of error: {error}")
