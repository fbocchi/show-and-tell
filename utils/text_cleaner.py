import re


def clean(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def clean_all(captions):
    return [clean(c) for c in captions]

