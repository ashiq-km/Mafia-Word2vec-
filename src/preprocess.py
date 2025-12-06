import nltk
import pypdf
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

import src.config as config

# Download NLTK resources:

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def get_text_from_pdf(pdf_path):
    """
    Reads a PDF file and returns extracted raw text.
    Adds space between pages to prevent word merging.
    """

    text = ""

    try:

        reader = pypdf.PdfReader(str(pdf_path))

        for page in tqdm(reader.pages, desc=f"Reading {pdf_path.name}", leave=False):
            page_text = page.extract_text()

            if page_text:
                text += page_text + " "

    except Exception as e:
        print(f"[Error] failed to read {pdf_path}: {e}")

    return text


def clean_tokenize(raw_text):
    """
    cleans raw text, splits into sentences, tokenize each sentence,
    and removes punctuation + lowercases uisng simple_preprocess()
    """

    clean_text = raw_text.replace("\n", " ")

    sentences = sent_tokenize(clean_text)

    tokenized_sentences = []

    for sentence in tqdm(sentences, desc="Tokenizing sentences", leave=False):

        tokens = simple_preprocess(sentence)

        if tokens:
            tokenized_sentences.append(tokens)

    return tokenized_sentences


def main():

    print("👽 Starting preprocessing...")

    all_sentences = []

    for pdf_file in tqdm(config.RAW_DATA_FILES, desc="Processing PDFs", leave=False):
        raw_text = get_text_from_pdf(pdf_file)
        sentences = clean_tokenize(raw_text)
        all_sentences.extend(sentences)

    print(f"Total sentences extracted: {len(all_sentences)}")

    # Saving to file with Progress

    print(f"Saving to {config.PROCESSED_DATA_FILE}...")

    with open(config.PROCESSED_DATA_FILE, "w", encoding="utf-8") as f:
        for sentence in tqdm(all_sentences, desc="Writing sentences", leave=False):
            f.write(" ".join(sentence) + "\n")

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
