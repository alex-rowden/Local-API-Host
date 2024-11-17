'''This module will be used to host act as the server hosting the api. Largely FastAPI'''
#python modules
import os
#exteral modules
from fastapi import FastAPI, HTTPException
from typing import List
#my modules
from __version__ import version, title, description
from model_handling import ZephyrLLM

def create_app() -> FastAPI:
    root_path = r"/"
    if os.getenv('ROUTE'):
        root_path = os.getenv('ROUTE')

    app = FastAPI(root_path=root_path)

    # Initialize the ZephyrLLM model
    zephyr_llm = ZephyrLLM()

    @app.post("/generate_text/")
    async def generate_text(prompts: List[str] | str) -> List[str] | str:
        if isinstance(prompts, str):
            prompts = [prompts]

        if not prompts or not all(prompts):
            raise HTTPException(status_code=400, detail="Prompts cannot be empty or None")
        try:
            # Generate text using the ZephyrLLM model
            generated_texts = zephyr_llm.generate_text(prompts)
            return generated_texts
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app
