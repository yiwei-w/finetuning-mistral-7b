import os
import subprocess

files_in_directory = [f for f in os.listdir() if os.path.isfile(f)]
total_files = len(files_in_directory)

for index, file in enumerate(files_in_directory, start=1):
    print(f"Processing {file} ({index}/{total_files})...")
    
    command = ["nougat", file, "-o", "ocr_output"]
    
    # Start the process and connect to its output stream
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    file_prefix = os.path.splitext(file)[0]
    log_filename = file_prefix + ".log"
    
    with open(log_filename, 'w') as log_file:
        # Stream the output line-by-line
        for line in process.stdout:
            print(line, end='')  # Print to console
            log_file.write(line)  # Write to log file

    print("\n" + "-"*50 + "\n")  # Separate logs of different files
