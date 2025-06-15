import base64
import io
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import numpy as np
import pandas as pd
from langchain.chains import LLMChain
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import CohereEmbeddings, HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain.retrievers import MultiVectorRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Milvus
from langchain_core.vectorstores import VectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from llmx import TextGenerator, TextGenerationConfig
from ntviz.utils import clean_code_snippet


from ntviz.datamodel import ChartExecutorResponse, Summary
from ntviz.utils import get_project_root

logger = logging.getLogger("ntviz")


class EmbeddingModel:
    """Factory class for embedding models"""

    @staticmethod
    def create(provider: str = "gemini", **kwargs) -> Any:
        """Create an embedding model based on the provider

        Args:
            provider (str, optional): Provider name. Defaults to "openai".
            **kwargs: Additional keyword arguments for the provider.

        Returns:
            Any: Embedding model instance
        """
        if provider == "openai":
            return OpenAIEmbeddings(**kwargs)
        elif provider == "cohere":
            return CohereEmbeddings(**kwargs)
        elif provider == "huggingface":
            return HuggingFaceEmbeddings(**kwargs)
        elif provider == "gemini":
            return GoogleGenerativeAIEmbeddings(**kwargs)
        else:
            raise ValueError(f"Unsupported embedding model provider: {provider}")


class VectorDatabase:
    """Factory class for vector databases"""

    @staticmethod
    def create(
        embedding_model, provider: str = "faiss", persist_dir: str = None, **kwargs
    ) -> VectorStore:
        """Create a vector database based on the provider

        Args:
            embedding_model: Embedding model to use
            provider (str, optional): Provider name. Defaults to "faiss".
            persist_dir (str, optional): Directory to persist the database. Defaults to None.
            **kwargs: Additional keyword arguments for the provider.

        Returns:
            VectorStore: Vector store instance
        """
        # if provider == "faiss":
        #     if persist_dir and os.path.exists(os.path.join(persist_dir, "index.faiss")):
        #         return FAISS.load_local(persist_dir, embedding_model)
        #     return FAISS.from_documents([], embedding_model)
        # elif provider == "milvus":
        #     return Milvus(embedding_function=embedding_model, **kwargs)
        # else:
        #     raise ValueError(f"Unsupported vector database provider: {provider}")
        
        if provider == "faiss":
            if persist_dir and os.path.exists(os.path.join(persist_dir, "index.faiss")):
                return FAISS.load_local(persist_dir, embedding_model)
            else:
                dummy_doc = Document(page_content="This is a placeholder document for initializing FAISS.")
                return FAISS.from_documents([dummy_doc], embedding_model)
        
        elif provider == "milvus":
            return Milvus(embedding_function=embedding_model, **kwargs)
        
        else:
            raise ValueError(f"Unsupported vector DB provider: {provider}")


class MultimodalRAGPipeline:
    """Multimodal RAG Pipeline for NTViz"""

    def __init__(
        self,
        embedding_provider: str = "gemini",
        embedding_kwargs: Dict = None,
        vector_db_provider: str = "faiss",
        vector_db_kwargs: Dict = None,
        provider: str = "gemini",
        llm_kwargs: Dict = None,
    ):
        """Initialize the multimodal RAG pipeline

        Args:
            embedding_provider (str, optional): Embedding model provider. Defaults to "openai".
            embedding_kwargs (Dict, optional): Embedding model kwargs. Defaults to None.
            vector_db_provider (str, optional): Vector database provider. Defaults to "faiss".
            vector_db_kwargs (Dict, optional): Vector database kwargs. Defaults to None.
            llm_provider (str, optional): LLM provider. Defaults to "openai".
            llm_kwargs (Dict, optional): LLM kwargs. Defaults to None.
        """
        self.embedding_provider = embedding_provider
        self.embedding_kwargs = embedding_kwargs or {}
        self.vector_db_provider = vector_db_provider
        self.vector_db_kwargs = vector_db_kwargs or {}
        self.provider = provider
        self.llm_kwargs = llm_kwargs or {}

        # Initialize embedding model
        self.embedding_model = EmbeddingModel.create(
            embedding_providerprovider=embedding_provider, **self.embedding_kwargs
        )

        # Set up storage for raw content (images, text chunks, tables)
        self.byte_store = InMemoryStore()  # Can be replaced with Redis, Filesystem, etc.

        # Initialize vector database for storing embeddings
        persist_dir = os.path.join(get_project_root(), "vectorstore")
        os.makedirs(persist_dir, exist_ok=True)
        
        self.vector_db_kwargs["persist_dir"] = persist_dir
        self.vector_db = VectorDatabase.create(
            self.embedding_model,
            provider=vector_db_provider,
            **self.vector_db_kwargs
        )

        # Initialize MultiVector retriever
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vector_db,
            docstore=self.byte_store,
            id_key="doc_id",
        )

        # Initialize LLM
        if provider == "openai":
            self.llm = ChatOpenAI(**self.llm_kwargs)
        elif provider == "gemini":
            self.llm = ChatGoogleGenerativeAI(**self.llm_kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        # Initialize text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )

    def _image_to_base64(self, image_data: bytes) -> str:
        """Convert image bytes to base64 string

        Args:
            image_data (bytes): Image bytes

        Returns:
            str: Base64 encoded image
        """
        return base64.b64encode(image_data).decode("ascii")

    def _extract_chart_image(self, chart: ChartExecutorResponse) -> Optional[bytes]:
        """Extract chart image bytes from chart object

        Args:
            chart (ChartExecutorResponse): Chart response object

        Returns:
            Optional[bytes]: Image bytes or None
        """
        if chart.raster:
            # If raster is already available as base64, decode it
            return base64.b64decode(chart.raster)
        
        # For non-raster charts (e.g., vega-lite), we need to render them
        # This would require a secure execution environment
        # For now, we return None
        return None

    # def analyze_chart(
    #     self, 
    #     chart: ChartExecutorResponse, 
    #     query: str = "Analyze this chart and provide insights", 
    #     k: int = 3
    # ) -> str:
    #     """Analyze a chart using multimodal RAG

    #     Args:
    #         chart (ChartExecutorResponse): Chart to analyze
    #         query (str, optional): User query. Defaults to "Analyze this chart and provide insights".
    #         k (int, optional): Number of documents to retrieve. Defaults to 3.

    #     Returns:
    #         str: Analysis results
    #     """
    #     # Extract chart image
    #     image_bytes = self._extract_chart_image(chart)
    #     if not image_bytes:
    #         return "Unable to extract chart image for analysis"

    #     # Convert image to base64 for the multimodal LLM
    #     image_b64 = self._image_to_base64(image_bytes)
        
    #     # Define prompt for chart analysis
    #     chart_analysis_prompt = PromptTemplate(
    #         template="""
    #         You are a SENIOR DATA ANALYST with deep expertise in visual analytics and storytelling with data and chart. Your task is to interpret the following chart accurately and provide a concise yet comprehensive analysis report.

    #         You are given:
    #         1. Chart Code:
    #         {chart_code}

    #         2. Visualization (base64-encoded):
    #         <image_base64>{image_b64}</image_base64>

    #         3. User Query:
    #         {query}

    #         4. Retrieved Context (optional background information):
    #         {context}

    #         Based on the information above with the real data and chart, provide a structured analysis that includes:
    #         1. Chart Description: 
    #             - What does the chart show? Mention variables, labels, what the axes represent, and overall structure.
                
    #         2. Key Trends Identification:
    #             - Are there any increasing or decreasing trends?
    #             - Where are the highest and lowest points?
    #             - Any anomalies or unexpected patterns?

    #         3. Time-based Breakdown (if applicable):
    #             - Which time periods show significant changes?
    #             - Are there notable turning points? Link them to possible real-world events?
                
    #         4. Contextual Insights: (mention the specific values in the chart)
    #             - Use actual data points shown in the chart.
    #             - Explain what these values indicate and why they are important.
    #             - What is the crucial information from the specific values in the chart? (taking the real values from the provided dataset to analyze the chart specificially)
    #             - What meaningful insights can be derived by combining the chart with the provided context?
    #             - What are the most surprising or critical insights the chart reveals?
    #             - Is there any hidden insight from the chart? Does it reveal any important information?
                
                
    #         5. External Influences: 
    #             - Are there global or external factors that may have influenced the data or its trends?
    #             - Explaining the potetial situations might affect the results.
                
    #         6. Recommendations: 
    #             - Suggest practical and data-driven actions or decisions based on your analysis.
    #             - Recommendations should be clear and useful even for a non-technical stakeholder.
                
    #         Conclusion:
    #             - Summarize the report analysis for the user to read in bullet points.

    #         **Note:**  Do not mention anything about inability to see or render the image.

    #         Present your analysis for the non-specialists to understand in the clear format (e.g. write the analysis in bullet points,...)
            
    #         Analysis:
    #         """,
    #         input_variables=["chart_code", "image_b64", "query", "context"],
    #     )
        
        
    #     # Get relevant documents first (outside the chain)
    #     relevant_docs = self.retriever.invoke(query)
        
    #     # Format the context as a string
    #     context_str = "\n\n".join([doc.page_content for doc in relevant_docs])
        
    #     # Set up the RAG chain with pre-fetched context
    #     rag_chain = (
    #         {
    #             "context": lambda _: context_str,
    #             "chart_code": lambda _: chart.code,
    #             "image_b64": lambda _: image_b64,
    #             "query": lambda _: query,
    #         }
    #         | chart_analysis_prompt
    #         | self.llm
    #         | StrOutputParser()
    #     )
    #     # Execute the chain
    #     return rag_chain.invoke({})
    

    def analyze_chart(
        self, 
        chart: ChartExecutorResponse, 
        query: str = "Analyze this chart and provide insights", 
        k: int = 3
    ) -> str:
        """Analyze a chart using multimodal RAG for Gemini"""

        # Extract chart image
        image_bytes = self._extract_chart_image(chart)
        if not image_bytes:
            return "Unable to extract chart image for analysis"

        # Convert image to base64 MIME for Gemini
        image_b64 = self._image_to_base64(image_bytes)
        image_mime = f"data:image/png;base64,{image_b64}"

        # Get relevant documents for context
        relevant_docs = self.retriever.invoke(query)
        context_str = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Build multimodal message for Gemini
        human_message = HumanMessage(content=[
            {
                "type": "image_url",
                "image_url": {
                    "url": image_mime
                }
            },
            {
                "type": "text",
                "text": f"""
                You are a SENIOR DATA ANALYST with deep expertise in visual analytics and storytelling with data and chart. Your task is to interpret the following chart accurately and provide a concise yet comprehensive analysis report.

                You are given:
                1. Chart Code:
                {chart.code}

                2. Visualization (base64-encoded):
                <image_base64>{image_b64}</image_base64>

                3. User Query:
                {query}

                4. Retrieved Context (optional background information):
                {context_str}

                Based on the information above with the real data and chart, provide a structured analysis that includes:
                1. Chart Description: 
                    - What does the chart show? Mention variables, labels, what the axes represent, and overall structure.
                    
                2. Key Trends Identification:
                    - Are there any increasing or decreasing trends?
                    - Where are the highest and lowest points?
                    - Any anomalies or unexpected patterns?

                3. Time-based Breakdown (if applicable):
                    - Which time periods show significant changes?
                    - Are there notable turning points? Link them to possible real-world events?
                    
                4. Contextual Insights: (mention the specific values in the chart)
                    - Use actual data points shown in the chart.
                    - Explain what these values indicate and why they are important.
                    - What is the crucial information from the specific values in the chart? (taking the real values from the provided dataset to analyze the chart specificially)
                    - What meaningful insights can be derived by combining the chart with the provided context?
                    - What are the most surprising or critical insights the chart reveals?
                    - Is there any hidden insight from the chart? Does it reveal any important information?
                    
                    
                5. External Influences: 
                    - Are there global or external factors that may have influenced the data or its trends?
                    - Explaining the potetial situations might affect the results.
                    
                6. Recommendations: 
                    - Suggest practical and data-driven actions or decisions based on your analysis.
                    - Recommendations should be clear and useful even for a non-technical stakeholder.
                    
                Conclusion:
                    - Summarize the report analysis for the user to read in bullet points.

                **Note:**  Do not mention anything about inability to see or render the image.

                Present your analysis for the non-specialists to understand in the clear format (e.g. write the analysis in bullet points,...)
                """
            }
        ])

        
        return self.llm.invoke([human_message]).content

    
    

    def ingest_pdf(self, file_path: str) -> None:
        """Ingest a PDF document into the RAG system

        Args:
            file_path (str): Path to the PDF file
        """
        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Process each chunk and add to retriever
        for i, chunk in enumerate(chunks):
            # Create a unique ID for this chunk
            doc_id = f"pdf_{os.path.basename(file_path)}_{i}"
            
            # Add to byte store and vector store
            self.byte_store.mset([(doc_id, chunk)])
            
            # Create a summary document to be embedded
            summary_doc = Document(
                page_content=chunk.page_content,
                metadata={"doc_id": doc_id, "source": file_path, **chunk.metadata}
            )
            
            # Add to vector store
            self.vector_db.add_documents([summary_doc])
    
    def ingest_df(self, df, name: str = "uploaded_dataset"):
        if isinstance(df, str):
            df = pd.read_csv(df)
        elif isinstance(df, pd.DataFrame):
            df = df.copy()
        else:
            raise TypeError("`df` must be either a file path (str) or a pandas DataFrame.")
        
        df = df.dropna().drop_duplicates()

        # Describe
        description = df.describe(include='all').fillna("N/A").to_string()
        info = f"Dataset Name: {name}\nNumber of Rows: {len(df)}\nNumber of Columns: {len(df.columns)}\n\nSummary Statistics:\n{description}"

        doc_id = f"dataframe_{name}"
        doc = Document(
            page_content=info,
            metadata={"doc_id": doc_id, "source": "ntviz_dataframe", "name": name}
        )

        self.byte_store.mset([(doc_id, doc)])
        self.vector_db.add_documents([doc])
        
    def ingest_summary(self, summary: Summary) -> None:
        """Ingest a LIDA summary into the RAG system

        Args:
            summary (Summary): NTViz summary object
        """
        
        # Summary is dict so we can not use the object type
        summary_text = f"Dataset: {summary.get('name', '')}\n"
        summary_text += f"Description: {summary.get('dataset_description', '')}\n\n"
        summary_text += "Fields:\n"

        for field in summary.get("fields", []):
            summary_text += f"- {field.get('column', '')}: {field['properties'].get('dtype', '')}\n"
            if "description" in field['properties'] and field['properties']['description']:
                summary_text += f"  Description: {field['properties']['description']}\n"
            if "semantic_type" in field['properties'] and field['properties']['semantic_type']:
                summary_text += f"  Semantic Type: {field['properties']['semantic_type']}\n"
            if "samples" in field['properties'] and field['properties']['samples']:
                samples = ", ".join([str(s) for s in field['properties']['samples'][:3]])
                summary_text += f"  Samples: {samples}\n"
        
        doc_id = f"summary_{summary.get('name', '')}"
        doc = Document(
            page_content=summary_text,
            metadata={"doc_id": doc_id, "source": "ntviz_summary", "name": summary.get('name', '')}
        )
        
        # Add to byte store and vector store
        self.byte_store.mset([(doc_id, doc)])
        self.vector_db.add_documents([doc])

    def ingest_web_content(self, url: str, content: str) -> None:
        """Ingest web content into the RAG system

        Args:
            url (str): URL of the web content
            content (str): Content to ingest
        """
        # Split content into chunks
        texts = self.text_splitter.split_text(content)
        
        # Process each chunk
        for i, text in enumerate(texts):
            # Create a unique ID for this chunk
            doc_id = f"web_{url.replace('/', '_')}_{i}"
            
            # Create a document to be embedded
            doc = Document(
                page_content=text,
                metadata={"doc_id": doc_id, "source": url}
            )
            
            # Add to byte store and vector store
            self.byte_store.mset([(doc_id, doc)])
            self.vector_db.add_documents([doc])

    def update_knowledge_base(self, source_path: str = None) -> None:
        """Update the knowledge base by processing new or changed sources

        Args:
            source_path (str, optional): Path to source directory. Defaults to None.
        """
        if source_path is None:
            source_path = os.path.join(get_project_root(), "data")
        
        # Check if the source directory exists
        if not os.path.exists(source_path):
            logger.warning(f"Source directory {source_path} does not exist")
            return
        
        # Get list of PDF files in the source directory
        pdf_files = [
            os.path.join(source_path, f) 
            for f in os.listdir(source_path) 
            if f.lower().endswith(".pdf")
        ]
        
        # Process each PDF file
        for pdf_file in pdf_files:
            # Check if the file has been modified since last ingestion
            # For now, we'll just re-ingest all files
            # In a production system, you would check file modification times
            self.ingest_pdf(pdf_file)
        
        logger.info(f"Updated knowledge base with {len(pdf_files)} PDF files")


class Analyzer:
    """NTViz Analysis Component using Multimodal RAG"""

    def __init__(self, text_gen: TextGenerator = None):
        """Initialize the Analyzer component

        Args:
            text_gen (TextGenerator, optional): Text generator object. Defaults to None.
        """
        self.text_gen = text_gen
        
        
        # Determine provider
        provider = "gemini"
        
        # Default configuration based on provider
        if provider == "openai":
            default_config = {
                "embedding_provider": "openai",
                "embedding_kwargs": {"model": "text-embedding-3-small"},
                "vector_db_provider": "faiss",
                "vector_db_kwargs": {},
                "provider": "openai",
                "llm_kwargs": {"model_name": "gpt-4o", "temperature": 0.2},
            }
        elif provider == "gemini":
            default_config = {
                "embedding_provider": "gemini",
                "embedding_kwargs": {
                    "model": "models/embedding-001"
                },
                "vector_db_provider": "faiss",
                "vector_db_kwargs": {},
                "provider": "gemini",
                "llm_kwargs": {"model": "gemini-1.5-flash", "temperature": 0.2},
            }
    
    
        # Initialize the RAG pipeline
        self.rag_pipeline = MultimodalRAGPipeline(**default_config)

    def analyze(
        self,
        chart: ChartExecutorResponse,
        df: None,
        summary: Summary = None,
        query: str = "Analyze this chart and provide insights",
        textgen_config: TextGenerationConfig = TextGenerationConfig(),
        context_k: int = 3,
    ) -> str:
        """Analyze a chart using multimodal RAG

        Args:
            chart (ChartExecutorResponse): Chart to analyze
            summary (Summary, optional): Data summary. Defaults to None.
            query (str, optional): User query. Defaults to "Analyze this chart and provide insights".
            textgen_config (TextGenerationConfig, optional): Text generation config. Defaults to TextGenerationConfig().
            context_k (int, optional): Number of context documents to retrieve. Defaults to 3.

        Returns:
            str: Analysis results
        """
        # If summary is provided, ingest it
        if summary:
            self.rag_pipeline.ingest_summary(summary)
            
        # Ingest the original data
        if df is not None and not df.empty:
            self.rag_pipeline.ingest_df(df)
        
        # Analyze the chart using the RAG pipeline
        result = self.rag_pipeline.analyze_chart(chart, query, k=context_k)
        
        return result

    def ingest_document(self, file_path: str) -> None:
        """Ingest a document into the RAG system

        Args:
            file_path (str): Path to the document
        """
        if file_path.lower().endswith(".pdf"):
            self.rag_pipeline.ingest_pdf(file_path)
        else:
            logger.warning(f"Unsupported document type: {file_path}")

    def update_knowledge_base(self, source_path: str = None) -> None:
        """Update the knowledge base

        Args:
            source_path (str, optional): Path to source directory. Defaults to None.
        """
        self.rag_pipeline.update_knowledge_base(source_path)

    def ingest_web_content(self, url: str, content: str) -> None:
        """Ingest web content into the RAG system

        Args:
            url (str): URL of the web content
            content (str): Content to ingest
        """
        self.rag_pipeline.ingest_web_content(url, content) 
        
        