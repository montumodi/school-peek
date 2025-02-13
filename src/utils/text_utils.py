import re

def remove_extra_lines_from_string(text):
    cleaned_text = '\n'.join([line.strip() for line in text.split('\n') if line.strip() != ''])
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text