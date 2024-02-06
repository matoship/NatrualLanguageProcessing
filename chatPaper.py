import fitz
import numpy as np
import tensorflow_hub as hub
from sklearn.neighbors import NearestNeighbors
import openai
import re
import sys
import requests
from bs4 import BeautifulSoup
from lxml import etree
import tiktoken
import warnings
from bs4 import XMLParsedAsHTMLWarning
from urllib3.exceptions import InsecureRequestWarning
import logging

# logging.getLogger("urllib3").setLevel(logging.ERROR)
# warnings.simplefilter('ignore', InsecureRequestWarning)
# warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
file_path = sys.argv[1]
MaxToken = 800
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"

openai.api_key = '******'


class SemanticSearch:
    def __init__(self):
        self.use = hub.load(module_url)
        self.fitted = False

    def fit(self, data, batch=1000, n_neighbors=5):
        self.data = data
        self.embeddings = self.get_embeddings(data, batch=batch)
        self.nn = NearestNeighbors(n_neighbors=min(
            n_neighbors, len(self.embeddings)))
        self.nn.fit(self.embeddings)
        self.fitted = True

    def __call__(self, text):
        inp_emb = self.get_embeddings([text])
        distances, indices = self.nn.kneighbors(inp_emb)
        return [(self.data[i], dist) for dist, i in zip(distances[0], indices[0])]

    def get_embeddings(self, texts, batch=1000):
        embeddings = []
        for i in range(0, len(texts), batch):
            text_batch = texts[i: (i + batch)]
            emb_batch = self.use(text_batch)
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings

    def num_tokens_from_string(self, string: str) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(string))
        return num_tokens


class GrobidService:
    def __init__(self):
        self.url = "http://localhost:8070/api/processFulltextDocument"

    def extract_metadata(self, pdf_path):
        # Send PDF to GROBID service
        files = {"input": open(pdf_path, "rb")}
        response = requests.post(self.url, files=files)

        # Parse the TEI XML response
        soup = BeautifulSoup(response.text, 'lxml')
        references = soup.find('listbibl')

        # Extract the title
        title = soup.find('title').text if soup.find('title') else None

        # Create the root element for the XML tree
        root = etree.Element("Metadata")

        # Add the title to the XML tree
        if title:
            title_xml = etree.SubElement(root, "Title")
            title_xml.text = title

        # Add the references to the XML tree
        if references:
            # Remove unwanted characters and line breaks from the reference XML
            reference_xml_str = str(references).replace('\n', '')

            # Find all the reference tags
            ref_tags = BeautifulSoup(
                reference_xml_str, 'lxml').find_all('biblstruct')

            # Add each reference to the XML tree
            for ref_tag in ref_tags:
                ref_xml = etree.SubElement(root, "Reference")
                ref_xml.text = ref_tag.get_text()

        # Create an XML string from the XML tree
        xml_string = etree.tostring(
            root, encoding='utf-8', xml_declaration=True)

        return title, xml_string


def extract_text_from_pdf(file_path):
    with fitz.open(file_path) as doc:
        text = " ".join(page.get_text("text") for page in doc)
    return text


def split_into_paragraphs(text, max_len=200):
    paragraphs = re.split('\n{2,}', text)
    paragraphs = [para for para in paragraphs if len(para.split()) > 5]
    new_paragraphs = []
    for paragraph in paragraphs:
        words = paragraph.split()
        if len(words) > max_len:
            chunks = [' '.join(words[i:i + max_len])
                      for i in range(0, len(words), max_len)]
            new_paragraphs.extend(chunks)
        else:
            new_paragraphs.append(paragraph)
    return new_paragraphs


def is_reference_question(question):
    reference_keywords = ['reference', 'cited',
                          'bibliography', 'works cited', 'references']
    for keyword in reference_keywords:
        if keyword in question.lower():
            return True
    return False


def ask_gpt(context, question):
    prompt = context + "\nQuestion: " + question + "\nAnswer:"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=800,
    )
    return response.choices[0].text.strip()


def main():

    while True:
        question = input("Ask a question (or type 'quit' to stop): ")
        if question.lower() == 'quit':
            break

        # Initialize selected context and token count
        selected_context = []
        total_tokens = 0
        searcher = SemanticSearch()
        grobid = GrobidService()

        if (is_reference_question(question)):
            # Extract metadata (title and references)
            title, reference_xml = grobid.extract_metadata(file_path)
            question = "which following references are the top 3 key references for a paper titled {title}"
            # Convert the reference XML to a string
            reference_xml_str = reference_xml.decode('utf-8')

            # Remove the XML declaration from the string
            reference_xml_str = reference_xml_str.replace(
                "<?xml version='1.0' encoding='utf-8'?>", '').strip()

            # Parse the XML string using lxml
            root = etree.fromstring(reference_xml_str)
            # Extract the title
            title = root.find('Title').text
            # Extract all the references
            references = [ref.text for ref in root.findall('Reference')]

            # Compute the similarities in batch
            similarities = np.inner(searcher.get_embeddings(
                [title]), searcher.get_embeddings(references)).flatten()

            # Create a list of (similarity, reference) tuples
            results = list(zip(similarities, references))

            # Sort the results by similarity score in descending order
            results.sort(reverse=True)
            for similarity, reference_text in results:
                # Compute the number of tokens in the current result
                num_result_tokens = searcher.num_tokens_from_string(
                    reference_text)

                # If the current result fits within the maximum token limit, print it
                if total_tokens + num_result_tokens <= MaxToken:
                    selected_context.append(reference_text)
                    total_tokens += num_result_tokens
                # If the current result does not fit within the maximum token limit, stop printing results
                else:
                    print(f"Total tokens sent: {total_tokens}")
                    break

        else:
            text = extract_text_from_pdf(file_path)
            text_paragraphs = split_into_paragraphs(text)
            searcher.fit(text_paragraphs)

            search_result = searcher(question)
            for res in search_result:
                context, _ = res
                if total_tokens + searcher.num_tokens_from_string(context) <= MaxToken:
                    selected_context.append(context)
                    total_tokens += searcher.num_tokens_from_string(context)
                else:
                    print(f"Total tokens sent: {total_tokens}")
                    break

        print('Answer:', ask_gpt('\n'.join(selected_context), question))


if __name__ == "__main__":
    main()
