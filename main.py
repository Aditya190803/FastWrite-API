from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import base64
import os
import tempfile
import requests
from FastWrite import (
    extract_zip, list_code_files, read_file,  # UPDATED
    generate_documentation_groq, generate_documentation_gemini,
    generate_documentation_openai, generate_documentation_openrouter,
    generate_data_flow
)

# FastAPI app setup
app = FastAPI()

# ✅ CORS: Allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to FastAPI!"}

# Request body schema
class RequestBody(BaseModel):
    github_url: str = None
    zip_file: str = None
    llm_provider: str
    llm_model: str
    api_key: str = None
    prompt: str

def fetch_github_zip(github_url: str):
    if not github_url.startswith("https://github.com/"):
        raise ValueError("Invalid GitHub URL")
    if github_url.endswith('.zip'):
        try:
            response = requests.get(github_url, timeout=10)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            raise RuntimeError(f"Failed to fetch GitHub ZIP: {str(e)}")
    base_url = github_url.replace("github.com", "codeload.github.com")
    for branch in ["main", "master"]:
        zip_url = f"{base_url}/zip/refs/heads/{branch}"
        try:
            response = requests.get(zip_url, timeout=10)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            if branch == "master":
                raise RuntimeError(f"Failed to fetch GitHub ZIP for branches 'main' or 'master': {str(e)}")
            continue

def process_zip(zip_data, tmp_dir):
    zip_path = os.path.join(tmp_dir, "temp.zip")
    with open(zip_path, "wb") as f:
        f.write(zip_data)
    extract_zip(zip_path, tmp_dir)
    code_files = list_code_files(tmp_dir)  # ✅ Support all code files
    if not code_files:
        raise ValueError("No code files found in the ZIP")
    main_file_path = os.path.join(tmp_dir, code_files[0])
    return read_file(main_file_path)

@app.post("/generate")
async def generate_documentation(request: RequestBody):
    try:
        llm_provider = request.llm_provider.lower()
        custom_prompt = request.prompt
        api_key = request.api_key

        required_fields = {'llm_provider': llm_provider, 'llm_model': request.llm_model, 'prompt': custom_prompt}
        missing = [k for k, v in required_fields.items() if not v]
        if missing:
            raise HTTPException(status_code=400, detail=f'Missing required fields: {", ".join(missing)}')

        if not (request.github_url or request.zip_file):
            raise HTTPException(status_code=400, detail='Must provide either github_url or zip_file')

        if llm_provider in {"groq", "gemini", "openai", "openrouter"} and not api_key:
            raise HTTPException(status_code=400, detail=f'API key is required for {llm_provider}')

        with tempfile.TemporaryDirectory(dir='/tmp') as tmp_dir:
            if request.github_url:
                zip_data = fetch_github_zip(request.github_url)
            elif request.zip_file:
                try:
                    zip_data = base64.b64decode(request.zip_file)
                except Exception:
                    raise HTTPException(status_code=400, detail='Invalid base64-encoded ZIP file')

            code_content = process_zip(zip_data, tmp_dir)

            try:
                flowchart = generate_data_flow(code_content)
            except Exception as e:
                flowchart = f"Failed to generate data flow diagram: {str(e)}"

            if llm_provider == "groq":
                documentation = generate_documentation_groq(code_content, custom_prompt, api_key, request.llm_model)
            elif llm_provider == "gemini":
                documentation = generate_documentation_gemini(code_content, custom_prompt, api_key, request.llm_model)
            elif llm_provider == "openai":
                documentation = generate_documentation_openai(code_content, custom_prompt, api_key, request.llm_model)
            elif llm_provider == "openrouter":
                documentation = generate_documentation_openrouter(code_content, custom_prompt, api_key, request.llm_model)
            else:
                raise HTTPException(status_code=400, detail=f'Unsupported LLM provider: {llm_provider}')

            return {
                'documentation': documentation,
                'flowchart': flowchart,
                'llm_provider': llm_provider,
                'llm_model': request.llm_model
            }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Internal server error: {str(e)}')
