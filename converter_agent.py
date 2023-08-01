import os.path
import shutil

import requests
from bs4 import BeautifulSoup
from langchain import LLMChain, PromptTemplate
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import BaseRetriever, Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import Field
from tqdm import tqdm

from prompts import *
from sources import REFERENCE_PAGES, DOCS_BASE_URL

# Retrieval + Indexing Constant
PDF_PAGES = "./pdfs/*.pdf"

ERROR_CODES = (1000, 1111)

# Parsing Constants
ERROR_CONTEXT_WINDOW_PRE_LINES = 1
ERROR_CONTEXT_WINDOW_POST_LINES = 25
REFERENCE_SPLITTER_CHUNK_SIZE = 55
REFERENCE_SPLITTER_CHUNK_OVERLAP = 15


class ErrorCodeInputSchema(BaseModel):
	error_code: int = Field(...)


class ErrorCodeMatchingRetriever(BaseRetriever):
	error_code_db: dict

	class Config:
		arbitrary_types_allowed = True

	def _get_relevant_documents(self, query: str, *,
								run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
		return self.error_code_db[query]


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
	output_parser = CustomOutputParser()

	error_code_retriever = ErrorCodeMatchingRetriever(
		error_code_db=error_db
	)

	vector_retriever = VectorStoreRetriever(vectorstore=vector_db, search_kwargs={'k': k})
	llm = ChatOpenAI(temperature=0.0, model='gpt-4')

	prompt = PromptTemplate(
		input_variables=["modifications"],
		template="""Rewrite the following code with the given modifications below it:
		{modifications}
		"""
	)

	rewrite_code_tool = LLMChain(llm=llm, prompt=prompt)

	error_code_retriever_tool = RetrievalQA.from_chain_type(
		llm=llm, chain_type="stuff", retriever=error_code_retriever, verbose=True
	)

	reference_retriever_tool = RetrievalQA.from_chain_type(
		llm=llm, chain_type="stuff", retriever=vector_retriever, verbose=True,
	)

	tools = [
		Tool(
			name="error_code_retriever_tool",
			func=error_code_retriever_tool.run,
			description=f"""Retrieve DPCT error code messages, descriptions, and guides on how to fix them. Your input
			should be an integer error code of the form XXXX where XXXX is between {ERROR_CODES}.""",
			args_schema=ErrorCodeInputSchea,
			verbose=True
		),
		Tool(
			name="rewrite_code_tool",
			func=rewrite_code_tool.run,
			description="""Make the specified modifications to the given code. Your input should be a string
			containing the entire block of code you would like to modify, followed by a newline, and then a detailed
			description of the modification you would like to make. If you need to make multiple modifications, 
			place them in a list below the code. Detail any modifications you make with inline comments, citing
			sources as to why the changes you made are valid or necessary.""",
			verbose=True
		),
		Tool(
			name="reference_retriever_tool",
			func=reference_retriever_tool.run,
			description="""Retrieve documentation on CUDA and SYCL APIs. Pass in a function or a small block of code to 
			obtain additional information about its arguments, return values, performance considerations, and 
			other information.""",
			verbose=True
		)
	]

	agent = initialize_agent(
		llm=llm,
		verbose=True,
		agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
		tools=tools,
		output_parser=output_parser,
		handle_parsing_errors="Check your output and make sure it conforms!",
		agent_kwargs={
			'prefix': AGENT_IT_TEMPLATE_PREFIX,
			'suffix': AGENT_IT_TEMPLATE_SUFFIX,
			'format_instructions': AGENT_IT_INSTRUCTIONS
		}
	)
	return agent


def main():
	retrieve_and_store_webpages()

	with open('code.cu', 'r') as code:
		output = code.read()

	error_db = build_error_code_database()
	docs_db = get_vector_database()
	agent = build_chain(error_db=error_db, vector_db=docs_db, k=4)
	cuda_to_sycl_template = CudaToSyclPromptTemplate(
		input_variables=["error_codes", "block", "pre_lines", "post_lines"]
	)

	for prompt in parse_code_into_prompts(cuda_to_sycl_template, output):
		agent.run(prompt)


if __name__ == "__main__":
	main()
