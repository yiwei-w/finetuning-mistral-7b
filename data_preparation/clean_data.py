import os
import glob
import re

# Define the source and target folders
source_folder = './complete_files'
target_folder = './complete_files_md'

# Ensure the target folder exists
os.makedirs(target_folder, exist_ok=True)

# Get a list of all .mmd files in the source folder
mmd_files = glob.glob(os.path.join(source_folder, '*.mmd'))

# Regex pattern for Markdown images
image_pattern = re.compile(r'!\[.*?\]\(.*?\)')

# Process each file
for file_path in mmd_files:
    # Read the content of the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Perform the replacements
    content = content.replace("[MISSING_PAGE_POST]", "")
    # content = content.replace('\\[', '$$').replace('\\]', '$$')
    # content = content.replace('\\(', '$').replace('\\)', '$')
    # content = content.replace('\\mbox', '\\text')
    # content = content.replace("$$ $$", "$$\n$$")

    # Remove all Markdown images
    content = re.sub(image_pattern, '', content)

    # Define the new file path with .md extension
    base_name = os.path.basename(file_path)
    new_file_path = os.path.join(target_folder, os.path.splitext(base_name)[0] + '.md')

    # Write the modified content to the new file
    with open(new_file_path, 'w') as file:
        file.write(content)
