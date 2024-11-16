#first part imports
from typing import Dict, List, AsyncGenerator, Generator
from abc import ABC, abstractmethod
import uuid
#third-party imports
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, TextStreamer
import vllm

class LLM(ABC):
    '''Abstract Base Class for LLM'''
    @abstractmethod
    def __init__(self, model_name: str, device_map: str = "auto",
                torch_dtype: torch.dtype = torch.float16) -> None:
        '''Abstract Initialization function for Generic LLM Class to setup a CausalLM model and
         tokenizer'''
    @abstractmethod
    def generate_text(self, message_chain: List[Dict[str, str]], max_length : int =50,
                      num_return_sequences: int =1, max_new_tokens : int = 4096) -> str:
        '''Abstract Method to generate text from a prompt using the LLM'''

class HFLLM(LLM):
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
            return_tensors="pt",
            padding=True,
            truncation=True, 
            max_length=max_length
        )
        input_ids: torch.Tensor = inputs.to(self.model.device)
        streamer: TextStreamer = TextStreamer(
            self.tokenizer,
            skip_prompt=False,
            skip_special_tokens=True)

        outputs: torch.Tensor = self.model.generate(
            **input_ids,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            num_return_sequences=num_return_sequences
        )
        output_text: str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text

class VLLM(LLM):
    '''LLM Subclass which uses vLLM's inference engine for faster inference
      and lower memory usage'''

    def __init__(self, model_name: str, device_map: str = "auto",
                  dtype: str = 'half', streaming = True) -> None:
        '''Initialization function for vLLM LLM Class to setup a CausalLM model and tokenizer'''
        engine_args: vllm.EngineArgs = vllm.EngineArgs(
            model=model_name,
            dtype=dtype,
            enable_prefix_caching=True
        )
        if streaming:
            self.model = vllm.ASyncLLMEngine.from_engine_args(engine_args)
        else:
            self.model = vllm.LLMEngine.from_engine_args(engine_args)
        self.streaming: bool = streaming
    def message_chain_to_string(self, message_chain: List[Dict[str, str]]) -> str:
        '''Function to convert a message chain to a string'''
        prompt: str = ""
        for message in message_chain:
            prompt += f"{message['role']}: {message['content']}\n"
        return prompt
    def generate_text(
        self,
        message_chain: List[Dict[str, str]],
        max_new_tokens : int = 4096
    ) -> str:
        '''Function to generate text from a prompt using the LLM'''
        prompt: str = self.message_chain_to_string(message_chain)
        request_id: str = str(uuid.uuid().hex)
        sampling_params : vllm.SamplingParams = vllm.SamplingParams(
            temperature=0.2,
            top_p=0.95,
            max_tokens=max_new_tokens
        )

        model_response: vllm.Output = self.model.generate(prompt, sampling_params, request_id)
        if self.streaming:
            async def stream_response(model_response: vllm.Output) -> str:
                '''Function to stream the response from the LLM'''
                response: str = ""
                async for request_out in model_response:
                    
                    response += token.text
                    yield token.text
                return response

            return stream_response(model_response)
        else:
            return model_response.outputs[0].text


        



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
