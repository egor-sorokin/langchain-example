## Introduction
This is a QA Retriever for the WCAG 2.1 standard.
It is a tool that allows you to search for specific WCAG 2.1 success criteria 
and retrieve the relevant information from the standard.

**Note**: This is just a proof of concept and a starting point for further development.
Since model itself is open-source, it's not guaranteed that it will retrieve the best results.

**Note**: even though the WCAG 2.1 standard is used, the QA Retriever can be used 
for any other website to retrieve data from. Make sure you are allowed to do so.
For that change `urls` in the `src/main.py` in `get_documents` function.

This QA Retriever is built with usage of open-source tools only:
- Langchain
- HuggingFace (embeddings and model)
- Chroma (vector store)

**Note**: Since Langchain is fast evolving, the QA Retriever might not work with the latest version.
If you upgrade make sure to check the changes in the Langchain API and integration docs. 


## Requirements
You need an account on HuggingFace to use the QA Retriever.
You can create one here: https://huggingface.co/join

Create an API token on HuggingFace and put in the `src/.env` file:
```
HUGGINGFACEHUB_API_TOKEN=<your_token>
```

## System requirements

- Python 3.11 or higher
- pip3

Better to use a virtual environment for the installation:
```
python3 -m venv venv
source venv/bin/activate
```

## Usage
Install the dependencies:
```
pip3 install -r requirements.txt
```

Navigate to the `src` directory and change `questions_to_ask` list, save and then run the `main.py` script:
```
python3 main.py
```
