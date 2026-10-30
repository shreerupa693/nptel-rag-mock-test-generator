import os
from step1_pipeline import parse_and_chunk_nptel_pdfs
from step2_database import NPTELVectorDB

from step3_cloud_api import ask_cloud_llm
from dotenv import load_dotenv
load_dotenv()

# ANSI color codes to make the CLI look visually professional and easy to read
CLR_HEADER = "\033[95m"
CLR_BLUE = "\033[94m"
CLR_GREEN = "\033[92m"
CLR_YELLOW = "\033[93m"
CLR_RESET = "\033[0m"

def run_cli_app():
    print(f"{CLR_HEADER}==================================================")
    print("      NPTEL IIOT EXAM ACCELERATOR (RAG CLI)       ")
    print(f"=================================================={CLR_RESET}\n")

    # 1. Initialize local DB file
    db_file = "nptel_study.db"
    db = NPTELVectorDB(db_file)
    
    # Quick check to see if database table exists and has data
    cursor = db.conn.cursor()
    cursor.execute("SELECT count(*) FROM knowledge_base")
    row_count = cursor.fetchone()[0]

    if row_count == 0:
        print(f"{CLR_YELLOW}[SYSTEM] Local database empty. Starting Phase 1 Ingestion...{CLR_RESET}")
        raw_chunks = parse_and_chunk_nptel_pdfs("pdfs")
        
        if not raw_chunks:
            print(f"{CLR_YELLOW}[WARNING] Drop your NPTEL PDFs into the 'pdfs' folder and restart.{CLR_RESET}")
            return
            
        db.insert_chunks(raw_chunks)
    else:
        print(f"{CLR_GREEN}[SYSTEM] Local database active with {row_count} historical slide vectors loaded.{CLR_RESET}")

    # 2. Continuous UI/UX Interaction Prompt Loop
    while True:
        print(f"\n{CLR_BLUE}--------------------------------------------------{CLR_RESET}")
        print("CHOOSE AN OPTION:")
        print("1. Ask a specific conceptual question")
        print("2. Generate a 5-Question NPTEL Mock Test")
        print("3. Exit Program")
        
        choice = input(f"\n{CLR_YELLOW}Select Option (1/2/3): {CLR_RESET}").strip()

        if choice == "3":
            print(f"\n{CLR_GREEN}Shutting down pipeline cleanly. Good luck with your NPTEL Exam!{CLR_RESET}")
            break

        elif choice == "1":
            user_query = input(f"{CLR_YELLOW}Enter your question/concept: {CLR_RESET}").strip()
            if not user_query: continue
            
            print(f"\n[1/2] Computing matrix coordinates locally...")
            matched_contexts = db.query_similarity(user_query, top_k=3)
            
            print(f"[2/2] Connecting to Cloud LLM reasoning engine...")
            answer = ask_cloud_llm(user_query, matched_contexts)
            
            print(f"\n{CLR_GREEN}=== CRITICAL EXAM ANSWER ==={CLR_RESET}")
            print(answer)

        elif choice == "2":
            topic = input(f"{CLR_YELLOW}Enter topic for the Mock Quiz (e.g., 'MQTT vs CoAP'): {CLR_RESET}").strip()
            if not topic: continue
            
            quiz_prompt = (
                f"Generate a strict 5-Question Multiple Choice Quiz (MCQ) testing: '{topic}'. "
                "Include options A, B, C, D. Focus heavily on protocol layers, technical values, and IEEE standards. "
                "Provide the correct answer key with short explanations at the very bottom."
            )
            
            print(f"\n[1/2] Extracting exact textbook slides regarding {topic}...")
            matched_contexts = db.query_similarity(topic, top_k=8)
            
            print(f"[2/2] Generating exam questions via cloud compute...")
            quiz = ask_cloud_llm(quiz_prompt, matched_contexts)
            
            print(f"\n{CLR_GREEN}=== DYNAMIC NPTEL MOCK EXAM ==={CLR_RESET}")
            print(quiz)
            
        else:
            print(f"{CLR_YELLOW}[INVALID] Please input 1, 2, or 3.{CLR_RESET}")

if __name__ == "__main__":
    run_cli_app()
