from datetime import datetime
from pathlib import Path

def filename_suffix(file_name):
    # build the filename for the OCR text file
    image_file = Path(file_name).name

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    output_file = f"output/{image_file}_{timestamp}.markdown"

    return output_file