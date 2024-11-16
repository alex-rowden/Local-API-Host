#first part imports
from typing import Dict, List

#third-party imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

class LLM():
    '''Base LLM Class to handle the tokenizer and inference model for the API'''
    def __init__(self, model_name: str, device_map: str = "auto",
                 torch_dtype: torch.dtype = torch.float16) -> None:
        '''Initialization function for Generic LLM Class to setup a CausalLM model and tokenizer'''

        self.tokenizer: AutoTokenizer= AutoTokenizer.from_pretrained(model_name)
        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map = device_map,
            torch_dtype = torch_dtype
        )
    def generate_text(self, message_chain: List[Dict[str, str]], max_length : int =50,
                      num_return_sequences: int =1, max_new_tokens : int = 4096) -> str:
        '''Function to generate text from a prompt using the LLM'''
        prompt:  List[int] | Dict[str, str] = self.tokenizer.apply_chat_template(
            message_chain,
            tokenize=False,
            and_add_generation_prompt=True
        )
        inputs: BatchEncoding = self.tokenizer(
            prompt,
            return_tensors="pt"
        )
        input_ids: torch.Tensor = inputs.to(self.model.device)
        outputs: torch.Tensor = self.model.generate(
            **input_ids,
            use_cache=True,
            max_new_tokens=max_length, 
            num_return_sequences=num_return_sequences
        )
        output_text: str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text
class SolarLLM(LLM):
    '''LLM Subclass for the Upstage Solar-10.7b-Instruct-v1.0 model'''
    def __init__(self, **kwargs) -> None:
        '''Constructor for SolarLLM. Just calls the superclass constructor
        with the correct model name'''
        super().__init__("Upstage/SOLAR-10.7b-Instruct-v1.0", **kwargs)

if __name__ == "__main__":
    llm: LLM = SolarLLM(device_map = 'cuda:0')
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Give me an idea for a dnd character based off Michael from the Good Place."
        }
    ]

    response: str = llm.generate_text(messages)
    print(response)
