import os
import streamlit as st
import asyncio
import asyncpg
from dotenv import load_dotenv
from nanonets import NANONETSOCR
from groq import Groq
import json
from typing import Dict, Optional
from fuzzywuzzy import fuzz

# Load environment variables
load_dotenv()

# Configuration
NEON_DB_USER = os.getenv("NEON_DB_USER")
NEON_DB_PASSWORD = os.getenv("NEON_DB_PASSWORD")
NEON_DB_HOST = os.getenv("NEON_DB_HOST")
NEON_DB_PORT = os.getenv("NEON_DB_PORT")
NEON_DB_NAME = os.getenv("NEON_DB_NAME")
NANONETS_API_KEY = os.getenv("NANONETS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize OCR and LLM
model = NANONETSOCR()
model.set_token(NANONETS_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

# Database functions
async def connect_to_neon():
    return await asyncpg.connect(
        user=NEON_DB_USER,
        password=NEON_DB_PASSWORD,
        database=NEON_DB_NAME,
        host=NEON_DB_HOST,
        port=NEON_DB_PORT
    )

async def user_exists(username: str) -> bool:
    conn = await connect_to_neon()
    try:
        result = await conn.fetch('SELECT COUNT(*) FROM accounts WHERE username = $1', username)
        return result[0]['count'] > 0
    finally:
        await conn.close()

async def create_new_user(username: str):
    conn = await connect_to_neon()
    try:
        await conn.execute(
            'INSERT INTO accounts (username, doc_verification) VALUES ($1, $2)',
            username, None
        )
    finally:
        await conn.close()

async def update_verification_status(username: str, status: str):
    conn = await connect_to_neon()
    try:
        await conn.execute(
            'UPDATE accounts SET doc_verification = $1 WHERE username = $2',
            status, username
        )
    finally:
        await conn.close()

async def get_user_record(username: str):
    """Get user record from users table"""
    conn = await connect_to_neon()
    try:
        result = await conn.fetchrow(
            'SELECT * FROM users WHERE username = $1',
            username
        )
        return result
    finally:
        await conn.close()

async def create_or_update_user_record(username: str, name: str, phone: str, address: str):
    """Create or update user record in users table"""
    conn = await connect_to_neon()
    try:
        # Check if user exists
        existing_user = await get_user_record(username)
        
        if existing_user:
            # Update existing record
            await conn.execute(
                '''
                UPDATE users 
                SET name = $1, phone = $2, address = $3 
                WHERE username = $4
                ''',
                name, phone, address, username
            )
        else:
            # Create new record
            await conn.execute(
                '''
                INSERT INTO users (username, name, phone, address)
                VALUES ($1, $2, $3, $4)
                ''',
                username, name, phone, address
            )
    finally:
        await conn.close()

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temp directory and return the path"""
    file_path = os.path.join("temp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def get_extraction_prompt(doc_type: str, text: str) -> str:
    """Generate appropriate prompt based on document type"""
    base_prompt = """
    Extract the following information from the text below and return it as a JSON object:
    - name: The full name of the person
    - phone: The phone number (if present)
    - address: The complete address

    Text: {text}

    Important: Look carefully through the entire text for these details. They might appear anywhere in the document.
    The name might be preceded by terms like "Name:", "Customer Name:", etc.
    The address might be preceded by "Address:", "Residence:", "Location:", etc.
    Phone numbers might be in various formats including +91 prefix or 10 digits.

    Return only the JSON object in this format:
    {{
        "name": "extracted name or null",
        "phone": "extracted phone or null",
        "address": "extracted address or null"
    }}
    """
    
    if doc_type == "id":
        base_prompt += "\nNote: This is an ID document. Look for officially stated name and address."
    elif doc_type == "bank":
        base_prompt += "\nNote: This is a bank statement. Look for account holder details and registered address."
    
    return base_prompt.format(text=text)

def extract_entities_using_groq(text: str, doc_type: str) -> Dict[str, Optional[str]]:
    """Extract entities from text using Groq's language model"""
    prompt = get_extraction_prompt(doc_type, text)

    try:
        response = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="mixtral-8x7b-32768",
            temperature=0.1,
            max_tokens=500,
        )
        
        # Log the raw text and extracted information for debugging
        print(f"\nProcessing {doc_type.upper()} document")
        print("Raw OCR text:", text[:500] + "..." if len(text) > 500 else text)
        print("Groq response:", response.choices[0].message.content)
        
        # Parse the response into a dictionary
        extracted_info = json.loads(response.choices[0].message.content)
        return {
            'name': extracted_info.get('name'),
            'phone': extracted_info.get('phone'),
            'address': extracted_info.get('address')
        }
    except Exception as e:
        print(f"Error processing {doc_type} document:", str(e))
        return {'name': None, 'phone': None, 'address': None}

def process_document(file_path: str, doc_type: str) -> Dict[str, Optional[str]]:
    """Process document using OCR and Groq for entity extraction"""
    try:
        # Get OCR text
        text = model.convert_to_string(file_path, formatting='lines')
        
        # Extract entities using Groq
        return extract_entities_using_groq(text, doc_type)
    except Exception as e:
        print(f"Error in document processing: {str(e)}")
        return {'name': None, 'phone': None, 'address': None}

def compare_with_fuzzy_match(str1: str, str2: str, threshold: int = 80) -> bool:
    """Compare two strings using fuzzy matching"""
    if not str1 or not str2:
        return False
    
    # Clean the strings
    str1 = str1.lower().strip()
    str2 = str2.lower().strip()
    
    # Get the ratio
    ratio = fuzz.ratio(str1, str2)
    token_sort_ratio = fuzz.token_sort_ratio(str1, str2)
    
    # Use the higher of the two scores
    similarity = max(ratio, token_sort_ratio)
    
    return similarity >= threshold

def compare_phone_numbers(phone1: str, phone2: str) -> bool:
    """Compare phone numbers by removing all non-digits"""
    if not phone1 or not phone2:
        return False
    
    # Remove all non-digits
    phone1 = ''.join(filter(str.isdigit, phone1))
    phone2 = ''.join(filter(str.isdigit, phone2))
    
    # If both are empty after cleaning, return False
    if not phone1 or not phone2:
        return False
    
    # Compare the last 10 digits if longer
    phone1 = phone1[-10:] if len(phone1) >= 10 else phone1
    phone2 = phone2[-10:] if len(phone2) >= 10 else phone2
    
    return phone1 == phone2

def compare_extracted_info(id_info: Dict, bank_info: Dict) -> tuple:
    """Compare extracted information from both documents using fuzzy matching"""
    matches = []
    mismatches = []
    match_details = {}
    
    # Compare each field
    if id_info['name'] and bank_info['name']:
        name_match = compare_with_fuzzy_match(id_info['name'], bank_info['name'])
        if name_match:
            matches.append('name')
            match_details['name'] = "Matched with high confidence"
        else:
            mismatches.append('name')
            match_details['name'] = "Names differ significantly"
    
    if id_info['phone'] and bank_info['phone']:
        phone_match = compare_phone_numbers(id_info['phone'], bank_info['phone'])
        if phone_match:
            matches.append('phone')
            match_details['phone'] = "Phone numbers match"
        else:
            mismatches.append('phone')
            match_details['phone'] = "Phone numbers differ"
    
    if id_info['address'] and bank_info['address']:
        address_match = compare_with_fuzzy_match(id_info['address'], bank_info['address'], threshold=80)
        if address_match:
            matches.append('address')
            match_details['address'] = "Addresses match with high similarity"
        else:
            mismatches.append('address')
            match_details['address'] = "Addresses differ significantly"
    
    # Consider verified if at least 2 fields match
    is_verified = len(matches) >= 2
    return is_verified, matches, mismatches, match_details

def main():
    username = st.text_input("Enter your username:")
    
    if username:
        user_exists_result = asyncio.run(user_exists(username))
        if not user_exists_result:
            asyncio.run(create_new_user(username))
            st.success(f"New user '{username}' created.")
        
        st.write("Please upload your documents for verification:")
        
        id_file = st.file_uploader("Upload ID Document (PDF)", type="pdf")
        bank_file = st.file_uploader("Upload Bank Statement (PDF)", type="pdf")
        
        if id_file and bank_file:
            with st.spinner("Processing documents..."):
                try:
                    # Save and process files
                    id_path = save_uploaded_file(id_file)
                    bank_path = save_uploaded_file(bank_file)
                    
                    id_info = process_document(id_path, "id")
                    
                    # Log ID document information to users table
                    asyncio.run(create_or_update_user_record(
                        username,
                        id_info['name'],
                        id_info['phone'],
                        id_info['address']
                    ))
                    
                    bank_info = process_document(bank_path, "bank")
                    
                    # Display extracted information
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("ID Document Info")
                        st.write(f"Name: {id_info['name']}")
                        st.write(f"Phone: {id_info['phone']}")
                        st.write(f"Address: {id_info['address']}")
                    
                    with col2:
                        st.subheader("Bank Statement Info")
                        st.write(f"Name: {bank_info['name']}")
                        st.write(f"Phone: {bank_info['phone']}")
                        st.write(f"Address: {bank_info['address']}")
                    
                    # Compare and verify with detailed matching
                    is_verified, matches, mismatches, match_details = compare_extracted_info(id_info, bank_info)
                    
                    st.subheader("Verification Results")
                    if is_verified:
                        st.success("Documents Verified Successfully!")
                        status = "verified"
                    else:
                        st.error("Document Verification Failed")
                        status = "not verified"
                    
                    # Update database
                    asyncio.run(update_verification_status(username, status))
                    
                    # Show detailed matching results
                    st.subheader("Matching Details")
                    for field in ['name', 'phone', 'address']:
                        if field in match_details:
                            icon = "✅" if field in matches else "❌"
                            st.write(f"{icon} {field.title()}: {match_details[field]}")
                    
                finally:
                    # Cleanup
                    if 'id_path' in locals():
                        os.remove(id_path)
                    if 'bank_path' in locals():
                        os.remove(bank_path)

if __name__ == "__main__":
    main()
