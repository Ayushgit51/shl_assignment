import requests
import json

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

CATALOG_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"

# Download catalog
response = requests.get(CATALOG_URL)

# Parse JSON safely
import json

data = json.loads(response.text, strict=False)

# Save locally
with open("catalog.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

documents = []

# Create documents
for item in data:

    name = item.get("name", "")
    description = item.get("description", "")
    url = item.get("url", "")
    category = item.get("category", "")
    test_type = item.get("test_type", "")

    content = f"""
    Assessment Name: {name}
    Description: {description}
    Category: {category}
    Test Type: {test_type}
    URL: {url}
    """

    doc = Document(
        page_content=content,
        metadata={
            "name": name,
            "url": url,
            "test_type": test_type
        }
    )

    documents.append(doc)

print(f"Total documents: {len(documents)}")

# Embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector db
vectorstore = FAISS.from_documents(
    documents,
    embedding_model
)

# Save locally
vectorstore.save_local("faiss_index")

print("FAISS vector DB created successfully")