import os
import re
import time  # ENHANCEMENT 3: Import timer
import pypdf

def parse_and_chunk_nptel_pdfs(pdf_folder: str, chunk_size: int = 150, overlap: int = 30) -> list[dict]:
    start_time = time.time()  # Start the clock
    
    if not os.path.exists(pdf_folder):
        print(f"[ERROR] Create a folder named '{pdf_folder}' and put your NPTEL PDFs inside it.")
        return []

    all_final_chunks = []
    
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"[PARSING] Processing file: {filename}")
            
            with open(pdf_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                
                for page_num, page in enumerate(reader.pages):
                    raw_text = page.extract_text()
                    
                    if not raw_text or not raw_text.strip():
                        continue
                    
                    # ENHANCEMENT 1: Clean out repetitive NPTEL header noise
                    raw_text = raw_text.replace("Introduction to Industry 4.0 and Industrial Internet of Things", "")
                    
                    cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()
                    words = cleaned_text.split(" ")
                    
                    start = 0
                    while start < len(words):
                        # ENHANCEMENT 2: Dynamic padding to protect acronyms
                        current_chunk_size = chunk_size
                        end = start + current_chunk_size
                        
                        # Look ahead: if the next word is a critical protocol, extend the window slightly
                        if end < len(words) and words[end-1] in ["MQTT", "CoAP", "IEEE", "RFID", "OPC-UA"]:
                            end += 5  
                        
                        chunk_words = words[start:end]
                        chunk_string = " ".join(chunk_words)
                        
                        all_final_chunks.append({
                            "text": chunk_string,
                            "metadata": {
                                "source_file": filename,
                                "page_number": page_num + 1
                            }
                        })
                        
                        start += (current_chunk_size - overlap)
                        
    execution_time = time.time() - start_time  # Stop the clock
    print(f"\n[SUCCESS] Pipeline Step 1 Complete in {execution_time:.2f} seconds!")
    print(f"[SUCCESS] Generated {len(all_final_chunks)} total chunks.")
    return all_final_chunks
