import os
import re
import json
import requests
import PyPDF2
import time
from tqdm import tqdm

with open("mathpix_credentials.json", "r") as f:
    credentials = json.load(f)

APP_ID = credentials["app_id"]
APP_KEY = credentials["app_key"]

def send_pdf_to_mathpix(temp_pdf_path, output_format='mmd'):
    url = 'https://api.mathpix.com/v3/pdf'
    headers = {'app_id': APP_ID, 'app_key': APP_KEY}
    with open(temp_pdf_path, 'rb') as file:
        files = {'file': file}
        options = {"conversion_formats": {"md": True}}
        print(f"Sending {os.path.getsize(temp_pdf_path) / 1000} kb to Mathpix")
        response = requests.post(url, headers=headers,
                                    files=files, data={"options_json": json.dumps(options)})
        response_data = response.json()
        print(response_data)

        if 'pdf_id' in response_data:
            pdf_id = response_data['pdf_id']
            print(f"PDF ID: {pdf_id}")
            return pdf_id
        else:
            print("Error: Unable to send PDF to Mathpix")
            return None

def wait_for_processing(pdf_id, output_format):
    url = f'https://api.mathpix.com/v3/pdf/{pdf_id}.{output_format}'
    headers = {'app_id': APP_ID, 'app_key': APP_KEY}
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.text

def process_mmd_files(folder_path, pdf_folder_path, output_folder_path):

    # mmd_files = [f for f in os.listdir(folder_path) if f.endswith(".mmd") and f.replace(".mmd", "_filled.mmd") not in os.listdir(output_folder_path)]
    mmd_files = [f for f in os.listdir(folder_path) if f.endswith(".mmd")]
    total_files = len(mmd_files)

    for file_name in tqdm(mmd_files, desc="Processing MMD files", total=total_files):
        mmd_path = os.path.join(folder_path, file_name)
        pdf_path = os.path.join(pdf_folder_path, file_name.replace(".mmd", ".pdf"))

        with open(mmd_path, 'r') as f:
            content = f.read()

        pdf_reader = PyPDF2.PdfReader(open(pdf_path, "rb"))

        for match in re.finditer(r'\[MISSING_PAGE_(EMPTY|FAIL):(\d+)\]', content):
            err_type = match.group(0)
            page_num = int(match.group(1))
            

            # pdf_writer = PyPDF2.PdfFileWriter()
            # pdf_writer.addPage(pdf_reader.getPage(page_num - 1))

            # with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            #     pdf_writer.write(temp_pdf)
            #     temp_pdf_path = temp_pdf.name

            pdf_writer = PyPDF2.PdfWriter()
            pdf_writer.add_page(pdf_reader.pages[page_num - 1])
            temp_pdf_path = f"./temp_pdf/temp_page_{page_num}.pdf"

            with open(temp_pdf_path, "wb") as temp_pdf:
                pdf_writer.write(temp_pdf)


            pdf_id = send_pdf_to_mathpix(temp_pdf_path)
            recognized_text = wait_for_processing(pdf_id, 'mmd')


            content = content.replace(f"[MISSING_PAGE_{err_type}:{page_num}]", recognized_text)
            # content = content.replace(f"[MISSING_PAGE_FAIL:{page_num}]", recognized_text)

        output_name = file_name.replace(".mmd", "_filled.mmd")
        output_path = os.path.join(output_folder_path, output_name)
        with open(output_path, 'w') as f:
            f.write(content)


if __name__ == "__main__":
    folder_path = "./test_folder"
    process_mmd_files(folder_path, "./qftbooks", "./test_folder")
