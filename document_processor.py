import chromadb
import PyPDF2
import docx
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import re
import openai
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional

# Load environment variables
load_dotenv()

class DocumentProcessor:
    def __init__(self):
        # Initialize ChromaDB client with new configuration
        self.chroma_client = chromadb.PersistentClient(path="chroma_db")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create or get collection for each user
        self.collections = {}
        
        # OpenAI configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
        if not self.openai_api_key:
            print("Warning: OpenAI API key is missing")
        else:
            openai.api_key = self.openai_api_key
            
        # Fine-tuned model configuration
        self.ft_model_id = os.getenv('OPENAI_FT_MODEL_ID', 'gpt-3.5-turbo')
        print(f"Using model: {self.ft_model_id}")
        
        # NDA detection keywords
        self.nda_keywords = [
            "confidential information", "non-disclosure", "nda", "confidentiality agreement",
            "proprietary information", "trade secret", "confidentiality obligation",
            "disclosing party", "receiving party", "disclosure of information",
            "confidential material", "confidentiality provision", "proprietary data",
            "confidential data", "non-disclosure agreement"
        ]
    
    def get_user_collection(self, user_id):
        if user_id not in self.collections:
            collection_name = f"user_{user_id}_docs"
            self.collections[user_id] = self.chroma_client.get_or_create_collection(
                name=collection_name
            )
        return self.collections[user_id]
    
    def read_pdf(self, file_path):
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def read_docx(self, file_path):
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    
    def read_txt(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def read_csv(self, file_path):
        """Read and process CSV file"""
        df = pd.read_csv(file_path)
        # Convert DataFrame to a formatted string representation
        text = "CSV Data:\n\n"
        # Add column names as header
        text += "Columns: " + ", ".join(df.columns) + "\n\n"
        # Add each row as a numbered entry
        for idx, row in df.iterrows():
            text += f"{idx + 1}. "
            for col in df.columns:
                text += f"{col}: {row[col]}, "
            text = text.rstrip(", ") + "\n"
        return text

    def read_excel(self, file_path):
        """Read and process Excel file"""
        df = pd.read_excel(file_path)
        # Convert DataFrame to a formatted string representation
        text = "Excel Data:\n\n"
        # Add column names as header
        text += "Columns: " + ", ".join(df.columns) + "\n\n"
        # Add each row as a numbered entry
        for idx, row in df.iterrows():
            text += f"{idx + 1}. "
            for col in df.columns:
                text += f"{col}: {row[col]}, "
            text = text.rstrip(", ") + "\n"
        return text

    def detect_format(self, text: str) -> Tuple[str, Dict]:
        """Detect the format of the text and return format type and metadata"""
        # Check for numbered format with descriptions (e.g. "1. Title: Description")
        numbered_desc_pattern = r'^\d+\.\s+[^:]+:.+'
        if re.search(numbered_desc_pattern, text, re.MULTILINE):
            return "numbered_with_desc", {"has_titles": True}
            
        # Check for simple numbered list
        numbered_pattern = r'^\d+\.\s+'
        if re.search(numbered_pattern, text, re.MULTILINE):
            return "numbered", {"has_titles": False}
            
        # Check for bullet points
        bullet_pattern = r'^\s*[-•*]\s+'
        if re.search(bullet_pattern, text, re.MULTILINE):
            return "bulleted", {"has_titles": False}
            
        return "paragraph", {"has_titles": False}

    def create_chunks(self, text: str, max_size: int = 500) -> List[Tuple[str, Dict]]:
        """Create chunks while preserving format"""
        format_type, metadata = self.detect_format(text)
        chunks = []
        
        if format_type in ["numbered_with_desc", "numbered"]:
            # Split by numbered items
            pattern = r'^\d+\.' if format_type == "numbered" else r'^\d+\.\s+[^:]+:'
            # Use re.MULTILINE flag instead of inline flag
            items = re.split(f'(?=^\\d+\\.)', text.strip(), flags=re.MULTILINE)
            items = [item.strip() for item in items if item.strip()]
            
            current_chunk = []
            current_size = 0
            
            for item in items:
                if current_size + len(item) > max_size and current_chunk:
                    chunks.append(('\n'.join(current_chunk), 
                                 {"format": format_type, **metadata}))
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(item)
                current_size += len(item)
            
            if current_chunk:
                chunks.append(('\n'.join(current_chunk), 
                             {"format": format_type, **metadata}))
                
        elif format_type == "bulleted":
            items = re.split(r'(?m)^(?=\s*[-•*]\s+)', text.strip())
            items = [item.strip() for item in items if item.strip()]
            
            current_chunk = []
            current_size = 0
            
            for item in items:
                if current_size + len(item) > max_size and current_chunk:
                    chunks.append(('\n'.join(current_chunk),
                                 {"format": format_type, **metadata}))
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(item)
                current_size += len(item)
            
            if current_chunk:
                chunks.append(('\n'.join(current_chunk),
                             {"format": format_type, **metadata}))
        else:
            # Handle paragraphs
            paragraphs = text.split('\n\n')
            current_chunk = []
            current_size = 0
            
            for para in paragraphs:
                if current_size + len(para) > max_size and current_chunk:
                    chunks.append(('\n\n'.join(current_chunk),
                                 {"format": format_type, **metadata}))
                    current_chunk = []
                    current_size = 0
                
                current_chunk.append(para)
                current_size += len(para)
            
            if current_chunk:
                chunks.append(('\n\n'.join(current_chunk),
                             {"format": format_type, **metadata}))
        
        return chunks
    
    def process_document(self, file_path, user_id):
        # Read the document based on its extension
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            text = self.read_pdf(file_path)
        elif file_extension == '.docx':
            text = self.read_docx(file_path)
        elif file_extension == '.txt':
            text = self.read_txt(file_path)
        elif file_extension == '.csv':
            text = self.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            text = self.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Create meaningful chunks
        chunks = self.create_chunks(text)
        
        if not chunks:
            raise ValueError("No valid content found in the document")
        
        # Get embeddings for the text content
        texts = [chunk[0] for chunk in chunks]
        embeddings = self.model.encode(texts)
        
        # Store in ChromaDB
        collection = self.get_user_collection(user_id)
        
        # Clear existing documents for this user
        try:
            collection.delete(ids=[f"doc_{i}" for i in range(len(chunks))])
        except:
            pass
        
        # Add documents to the collection with metadata
        collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[chunk[1] for chunk in chunks],
            ids=[f"doc_{i}" for i in range(len(chunks))]
        )
    
    def generate_response(self, context: str, question: str, format_type: str) -> str:
        """Generate response using OpenAI"""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is not configured")

        # For NDA analysis, we want to preserve the original format
        if self._is_nda_analysis_request(question):
            format_instruction = ""
        else:
            # For general questions, always use paragraph format for better readability
            format_instruction = "Respond in paragraph form with clear and concise language."
            format_type = "paragraph"

        system_prompt = "You are a helpful assistant that answers questions based on provided context. Always maintain the exact formatting as specified in the instructions."
        
        user_prompt = f"""Based on the following context, answer the question.
Context: {context}

Question: {question}

Important: {format_instruction}
Answer only using information from the context."""

        try:
            response = openai.chat.completions.create(
                model=self.ft_model_id,  # Use fine-tuned model if available
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating the response."

    def get_answer(self, question, user_id):
        """Generate answer using RAG with OpenAI"""
        if not self.openai_api_key:
            return "OpenAI API key is not configured."
        
        # Handle specialized NDA analysis if detected
        if self._is_nda_analysis_request(question):
            return self._perform_nda_analysis(question, user_id)

        # Get user collection
        collection = self.get_user_collection(user_id)
        
        # Query ChromaDB for relevant documents
        results = collection.query(
            query_texts=[question],
            n_results=3
        )
        
        # Extract results
        if not results or not results.get('documents') or not results['documents'][0]:
            return "I don't have enough information to answer that question."
        
        contexts = results['documents'][0]
        metadatas = results.get('metadatas', [[{}]])[0]
        
        # Get document format for preserving in response
        format_types = [meta.get('format', 'paragraph') for meta in metadatas]
        predominant_format = max(set(format_types), key=format_types.count) if format_types else 'paragraph'
        
        # Format context into single text
        context_text = "\n\n".join(contexts)
        
        # Generate response using OpenAI
        return self.generate_response(context_text, question, predominant_format)

    def _is_nda_analysis_request(self, question):
        """Detect if the question is asking for NDA analysis"""
        question = question.lower()
        nda_keywords = [
            "summarize this nda", "summarize the nda", "nda summary", 
            "simplify the clauses", "simplify key clauses", "simplify nda clauses",
            "identify risks", "risk assessment", "risks in this nda",
            "suggest improvements", "improve the nda", "better terms"
        ]
        
        return any(keyword in question for keyword in nda_keywords)
    
    def _is_document_nda(self, document_text: str) -> bool:
        """
        Determine if a document is likely an NDA based on keywords and content analysis
        
        Returns:
            bool: True if the document appears to be an NDA, False otherwise
        """
        # Convert to lowercase for case-insensitive matching
        document_lower = document_text.lower()
        
        # Check for common NDA terms
        keyword_matches = sum(1 for keyword in self.nda_keywords if keyword in document_lower)
        
        # If we find at least 3 NDA-related keywords, it's likely an NDA
        if keyword_matches >= 3:
            return True
            
        # Look for specific NDA sections or headings
        nda_section_patterns = [
            r'confidentiality\s+agreement',
            r'non.?disclosure\s+agreement',
            r'confidential\s+information',
            r'disclosure\s+of\s+information',
            r'confidentiality\s+obligations'
        ]
        
        for pattern in nda_section_patterns:
            if re.search(pattern, document_lower, re.IGNORECASE):
                return True
                
        return False
    
    def _perform_nda_analysis(self, question, user_id):
        """Handle different types of NDA analysis"""
        question_lower = question.lower()
        
        # Retrieve the entire document
        collection = self.get_user_collection(user_id)
        results = collection.query(
            query_texts=["document"],  # More generic query to retrieve any document
            n_results=10
        )
        
        if not results or not results.get('documents') or not results['documents'][0]:
            return "I couldn't find a document to analyze. Please upload one first."
        
        # Combine all chunks into a single document
        document_chunks = results['documents'][0]
        full_document = "\n\n".join(document_chunks)
        
        # Check if the document is actually an NDA
        if not self._is_document_nda(full_document):
            return "The uploaded document doesn't appear to be an NDA. Please upload an NDA document for NDA-specific analysis."
        
        # Determine which analysis to perform
        if "summarize" in question_lower:
            return self._summarize_nda(full_document)
        elif "simplify" in question_lower or "key clauses" in question_lower:
            return self._simplify_key_clauses(full_document)
        elif "risk" in question_lower or "identify" in question_lower:
            return self._identify_risks(full_document)
        elif "improve" in question_lower or "suggest" in question_lower:
            return self._suggest_improvements(full_document)
        else:
            # Default to summary if not specific
            return self._summarize_nda(full_document)
    
    def _summarize_nda(self, document_text):
        """Generate a comprehensive summary of the NDA using fine-tuned model if available"""
        prompt = """
        You are a legal expert specializing in NDAs (Non-Disclosure Agreements). 
        
        Provide a clear, concise summary of the following NDA document in paragraph form. 
        Your summary should be comprehensive yet easy to understand, covering:
        - The parties involved and their primary obligations
        - The scope of confidential information
        - Key restrictions and permitted uses
        - Duration of confidentiality obligations
        - Any notable special provisions
        
        NDA Document:
        {document}
        
        Provide a professional, easy-to-understand summary in several cohesive paragraphs.
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.ft_model_id,  # Use fine-tuned model if available
                messages=[
                    {"role": "system", "content": "You are a legal expert specializing in NDA analysis."},
                    {"role": "user", "content": prompt.format(document=document_text)}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating NDA summary: {str(e)}"
    
    def _simplify_key_clauses(self, document_text):
        """Simplify key clauses from the NDA using fine-tuned model if available"""
        prompt = """
        You are a legal expert who excels at making complex legal language accessible to non-lawyers.
        
        Analyze the following NDA document and identify the 5-7 most important clauses (such as confidentiality 
        obligations, non-compete provisions, term and termination, etc.).
        
        For each identified key clause:
        1. Quote the original clause text (briefly)
        2. Provide a plain-language explanation in 1-2 paragraphs
        3. Explain why this clause matters in practical terms
        
        Present your analysis in paragraph form, making it easy for a non-legal professional to understand.
        
        NDA Document:
        {document}
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.ft_model_id,  # Use fine-tuned model if available
                messages=[
                    {"role": "system", "content": "You are a legal expert who simplifies complex legal language."},
                    {"role": "user", "content": prompt.format(document=document_text)}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error simplifying NDA clauses: {str(e)}"
    
    def _identify_risks(self, document_text):
        """Identify and rate risks in the NDA using fine-tuned model if available"""
        prompt = """
        You are a legal expert specializing in contract risk assessment.
        
        Analyze the following NDA document and identify potentially problematic or risky clauses.
        For each identified risk:
        
        1. Quote the specific clause (briefly)
        2. Explain the potential risk or issue in 1-2 paragraphs
        3. Assign a severity rating (Low, Medium, High) and justify your rating
        4. Explain who bears this risk (primarily the disclosing or receiving party)
        
        Focus on issues like:
        - Overly broad definitions
        - Unreasonable time periods
        - Unbalanced obligations
        - Vague or ambiguous language
        - Missing standard protections
        
        Present your analysis in paragraph form, organized by risk severity (highest to lowest).
        
        NDA Document:
        {document}
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.ft_model_id,  # Use fine-tuned model if available
                messages=[
                    {"role": "system", "content": "You are a legal expert specializing in contract risk assessment."},
                    {"role": "user", "content": prompt.format(document=document_text)}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error identifying NDA risks: {str(e)}"
    
    def _suggest_improvements(self, document_text):
        """Suggest improvements to the NDA using fine-tuned model if available"""
        prompt = """
        You are a legal expert who specializes in drafting fair and balanced NDAs.
        
        Analyze the following NDA document and suggest specific improvements to make it clearer and more balanced.
        For each suggested improvement:
        
        1. Quote the original problematic clause (briefly)
        2. Explain why it should be improved in 1-2 paragraphs
        3. Provide specific alternative language or a draft of better wording
        4. Explain how your suggestion balances the interests of both parties
        
        Focus on practical improvements that would make the agreement:
        - More clear and specific
        - More balanced between the parties
        - More in line with standard industry practices
        - More enforceable
        
        Present your analysis and suggestions in paragraph form, focusing on the most important improvements first.
        
        NDA Document:
        {document}
        """
        
        try:
            response = openai.chat.completions.create(
                model=self.ft_model_id,  # Use fine-tuned model if available
                messages=[
                    {"role": "system", "content": "You are a legal expert who specializes in drafting fair and balanced contracts."},
                    {"role": "user", "content": prompt.format(document=document_text)}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error suggesting NDA improvements: {str(e)}" 