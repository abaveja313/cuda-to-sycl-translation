import os.path
import re
import shutil
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader

from prompts import CudaToSyclPromptTemplate
from sources import REFERENCE_PAGES, DOCS_BASE_URL

# Retrieval + Indexing Constant
PDF_PAGES = "./pdfs/*.pdf"

ERROR_CODES = (1000, 1111)

# Parsing Constants
ERROR_CONTEXT_WINDOW_PRE_LINES = 1
ERROR_CONTEXT_WINDOW_POST_LINES = 25
REFERENCE_SPLITTER_CHUNK_SIZE = 55
REFERENCE_SPLITTER_CHUNK_OVERLAP = 15


class CustomMatchingRetriever(BaseRetriever):
	error_code_db: dict
	vector_db: Chroma
	search_kwargs: dict

	class Config:
		arbitrary_types_allowed = True

	def _get_error_code_documents(self, query: str):
		error_codes = re.findall('DPCT(\\d{4})', query)
		located = set()
		documents = []
		for code in error_codes:
			if code not in located:
				documents.extend(self.error_code_db[code])
				located.add(code)
		return documents

	def _get_semantically_similar_documents(self, query: str):
		return self.vector_db.similarity_search(query, **self.search_kwargs)

	def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
		documents = []
		documents.extend(self._get_error_code_documents(query))
		documents.extend(self._get_semantically_similar_documents(query))
		return documents


def get_soup_for(*, url: str) -> BeautifulSoup:
	request = requests.get(url)
	request.raise_for_status()

	soup = BeautifulSoup(request.content, "html.parser")
	return soup


def retrieve_error_docs_page(error_code: int = None, url: str = None) -> str:
	target = DOCS_BASE_URL.format(code=error_code) if not url else url
	soup = get_soup_for(url=target)
	return soup.find(id=f"contentSection-1").prettify()


def setup_directory(path: str, reset=True):
	if os.path.exists(path):
		if not reset:
			return True
		shutil.rmtree(path)
	os.mkdir(path)
	return False


def retrieve_and_store_webpages(error_docs_dir: str = 'error_docs', reference_dir: str = 'reference_docs',
								reset: bool = False):
	if setup_directory(error_docs_dir, reset=reset) and setup_directory(reference_dir, reset=reset):
		print("No need to set up directories as it is already stored")
		return

	print("Retrieving Error Code Manual")
	for code in tqdm(range(*ERROR_CODES), total=ERROR_CODES[1] - ERROR_CODES[0]):
		document = retrieve_error_docs_page(code)
		with open(f'{error_docs_dir}/{code}_docs.html', 'w') as f:
			f.write(document)

	print("Retrieving Reference Documents")
	for reference_page in tqdm(REFERENCE_PAGES):
		soup = get_soup_for(url=reference_page)
		filename = reference_page.split('https://')[1].replace('/', '_').replace('.', '_').split('#')[0]
		with open(f'{reference_dir}/{filename}.html', 'w') as f:
			f.write(soup.prettify())


def loop_through_directory(*, dir: str):
	for root, dirs, files in os.walk(os.path.abspath(dir)):
		for file in files:
			yield os.path.join(root, file), file


def build_error_code_database(directory: str = 'error_docs'):
	if not os.path.exists(directory):
		raise FileNotFoundError("Unable to find the error directory. Please regenerate it!")

	documents = {}

	for full_path, filename in loop_through_directory(dir=directory):
		error_code = filename.split('_docs')[0]  # which error code is this?
		loader = UnstructuredHTMLLoader(full_path)
		data = loader.load_and_split()

		for doc in data:
			doc.metadata |= {"error_code": error_code}

		documents[error_code] = data

	return documents


def build_document_vector_database(reference_directory: str = 'reference_docs', pdf_directory: str = 'pdfs'):
	if not os.path.exists(reference_directory):
		raise FileNotFoundError("Unable to find the reference directory. Please regenerate it!")

	documents = []

	splitter = RecursiveCharacterTextSplitter(
		chunk_size=REFERENCE_SPLITTER_CHUNK_SIZE,
		chunk_overlap=REFERENCE_SPLITTER_CHUNK_OVERLAP,
		length_function=len,
	)

	for full_path, filename in loop_through_directory(dir=reference_directory):
		loader = UnstructuredHTMLLoader(full_path)
		data = loader.load_and_split(text_splitter=splitter)
		documents.extend(data)

	for pdf_fp, pdf_filename in loop_through_directory(dir=pdf_directory):
		loader = PyPDFLoader(pdf_fp)
		pages = loader.load_and_split(text_splitter=splitter)
		documents.extend(pages)

	db = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory='vectordb')
	db.persist()
	return db


def get_vector_database(persist_dir: str = 'vectordb'):
	db = Chroma(persist_directory=persist_dir, embedding_function=OpenAIEmbeddings())
	return db


def parse_code_into_prompts(template: CudaToSyclPromptTemplate, code: str):
	lines = code.split('\n')
	for idx, line in enumerate(lines):
		codes = re.findall('DPCT(\\d{4})', line)

		if len(codes) > 0:
			window_start = max(0, idx - ERROR_CONTEXT_WINDOW_PRE_LINES)
			window_end = min(idx + ERROR_CONTEXT_WINDOW_POST_LINES, len(lines) - 1)
			window = '\n'.join(lines[window_start:window_end])

			yield template.format(
				error_codes=','.join(codes),
				block=window,
				pre_lines=ERROR_CONTEXT_WINDOW_PRE_LINES,
				post_lines=ERROR_CONTEXT_WINDOW_POST_LINES
			)


def build_chain(*, error_db: dict, vector_db: Chroma, k: int):
	retriever = CustomMatchingRetriever(
		error_code_db=error_db,
		vector_db=vector_db,
		search_kwargs={'k': k}

	)

	llm = ChatOpenAI(temperature=0.0, model='gpt-4')

	qa_stuff = RetrievalQA.from_chain_type(
		llm=llm,
		retriever=retriever,
		chain_type="stuff",
		verbose=True
	)

	return qa_stuff


def main():
	retrieve_and_store_webpages()

	with open('code.cu', 'r') as code:
		output = code.read()

	error_db = build_error_code_database()
	docs_db = get_vector_database()
	qa_chain = build_chain(error_db=error_db, vector_db=docs_db, k=4)

	cuda_to_sycl_template = CudaToSyclPromptTemplate(
		input_variables=["error_codes", "block", "pre_lines", "post_lines"]
	)

	for prompt in parse_code_into_prompts(cuda_to_sycl_template, output):
		response = qa_chain(prompt)
		print("Prompt:\n", response['query'])
		print("=" * 30)
		print("Result:\n", response['result'])


if __name__ == "__main__":
	main()
