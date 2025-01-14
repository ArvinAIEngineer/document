import streamlit as st
from pyzerox import zerox
import os
import asyncio
import asyncpg
import logging
from dotenv import load_dotenv
import json
from typing import Dict, Tuple

# Load environment variables
load_dotenv()

# Neon DB connection parameters
NEON_DB_USER = os.getenv("NEON_DB_USER")
NEON_DB_PASSWORD = os.getenv("NEON_DB_PASSWORD")
NEON_DB_HOST = os.getenv("NEON_DB_HOST")
NEON_DB_PORT = os.getenv("NEON_DB_PORT")
NEON_DB_NAME = os.getenv("NEON_DB_NAME")

# GPT-4 Vision setup
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4-vision-preview"

async def connect_to_neon():
    """Create database connection"""
    conn = await asyncpg.connect(
        user=NEON_DB_USER,
        password=NEON_DB_PASSWORD,
        database=NEON_DB_NAME,
        host=NEON_DB_HOST,
        port=NEON_DB_PORT
    )
    return conn

async def setup_database():
    """Create necessary database tables"""
    conn = await connect_to_neon()
    try:
        await conn.execute('''
            CREATE TABLE IF NOT EXISTS accounts (
                username TEXT PRIMARY KEY,
                photo_verification TEXT,
                doc_verification TEXT,
                extracted_data JSONB
            )
        ''')
    finally:
        await conn.close()

async def user_exists(username: str) -> bool:
    """Check if user exists in database"""
    conn = await connect_to_neon()
    try:
        result = await conn.fetch('SELECT COUNT(*) FROM accounts WHERE username = $1', username)
        return result[0]['count'] > 0
    finally:
        await conn.close()

async def create_new_user(username: str):
    """Create new user in database"""
    conn = await connect_to_neon()
    try:
        await conn.execute(
            'INSERT INTO accounts (username, photo_verification, doc_verification, extracted_data) VALUES ($1, $2, $3, $4)',
            username, None, None, '{}'
        )
    except Exception as e:
        logging.error(f"Error creating new user {username}: {e}")
    finally:
        await conn.close()

async def update_verification_result(username: str, verification_status: str, extracted_data: Dict):
    """Update verification results in database"""
    conn = await connect_to_neon()
    try:
        await conn.execute(
            'UPDATE accounts SET doc_verification = $1, extracted_data = $2 WHERE username = $3',
            verification_status, json.dumps(extracted_data), username
        )
    except Exception as e:
        logging.error(f"Error updating verification result for {username}: {e}")
    finally:
        await conn.close()

async def extract_document_info(file_path: str) -> Dict:
    """Extract information from document using py-zerox"""
    custom_system_prompt = """
    Extract the following information from the document:
    - Name
    - Address
    - Phone Number
    
    Return the information in a JSON format with these exact keys: name, address, phone_number
    If any information is not found, set the value as null.
    """
    
    try:
        result = await zerox(
            file_path=file_path,
            model=MODEL,
            custom_system_prompt=custom_system_prompt
        )
        
        # Parse the markdown result into JSON
        # Note: You might need to adjust this parsing based on actual output format
        extracted_info = json.loads(result)
        return extracted_info
    except Exception as e:
        logging.error(f"Error extracting document info: {e}")
        return {"name": None, "address": None, "phone_number": None}

def compare_extracted_info(id_info: Dict, bank_info: Dict) -> Tuple[bool, str]:
    """Compare extracted information from both documents"""
    matches = []
    mismatches = []
    
    for field in ['name', 'address', 'phone_number']:
        if id_info.get(field) and bank_info.get(field):
            if id_info[field].lower() == bank_info[field].lower():
                matches.append(field)
            else:
                mismatches.append(field)
    
    # Verification passes if at least 2 fields match and there are no mismatches
    is_verified = len(matches) >= 2 and len(mismatches) == 0
    
    status_message = f"Matched fields: {', '.join(matches)}\n"
    if mismatches:
        status_message += f"Mismatched fields: {', '.join(mismatches)}"
    
    return is_verified, status_message

def main():
    st.title("Document Verification System")
    
    # Initialize database
    asyncio.run(setup_database())
    
    # Get username
    username = st.text_input("Enter your username:")
    if username:
        # Check if user exists
        user_exists_in_db = asyncio.run(user_exists(username))
        if not user_exists_in_db:
            st.info("Username not found. Creating a new user...")
            asyncio.run(create_new_user(username))
            st.success(f"New user '{username}' created.")
        
        st.success(f"Welcome, {username}! Please upload your documents.")
        
        # File uploaders
        id_file = st.file_uploader("Upload ID Proof", type=['pdf'])
        bank_file = st.file_uploader("Upload Bank Statement", type=['pdf'])
        
        if id_file and bank_file:
            # Save uploaded files temporarily
            with open("temp_id.pdf", "wb") as f:
                f.write(id_file.getvalue())
            with open("temp_bank.pdf", "wb") as f:
                f.write(bank_file.getvalue())
            
            if st.button("Verify Documents"):
                with st.spinner("Processing documents..."):
                    # Extract information from both documents
                    id_info = asyncio.run(extract_document_info("temp_id.pdf"))
                    bank_info = asyncio.run(extract_document_info("temp_bank.pdf"))
                    
                    # Compare extracted information
                    is_verified, status_message = compare_extracted_info(id_info, bank_info)
                    
                    # Display results
                    st.write("Extracted Information:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("ID Proof:")
                        st.json(id_info)
                    with col2:
                        st.write("Bank Statement:")
                        st.json(bank_info)
                    
                    if is_verified:
                        st.success("Documents Verified Successfully!")
                    else:
                        st.error("Verification Failed")
                    
                    st.write(status_message)
                    
                    # Update database
                    verification_status = "verified" if is_verified else "not_verified"
                    extracted_data = {
                        "id_proof": id_info,
                        "bank_statement": bank_info
                    }
                    asyncio.run(update_verification_result(username, verification_status, extracted_data))
            
            # Clean up temporary files
            os.remove("temp_id.pdf")
            os.remove("temp_bank.pdf")

if __name__ == "__main__":
    main()
