'''Module to handle models from various sources to be hosted through the API'''
#first part imports
from typing import Dict, List, AsyncGenerator
from abc import ABC, abstractmethod
import uuid
import json
import os
#third-party imports
import torch
from transformers import (AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    TextStreamer,
    BitsAndBytesConfig,
)
if os.name != 'nt':
    import vllm

class LLM(ABC):
    '''Abstract Base Class for LLM'''
    eos_token: str = "<\/s>"
    pad_token: str = "<pad>""
    padding_side: str = "right"
    @abstractmethod
    def __init__(self, model_name: str, device_map: str = "auto",
                torch_dtype: torch.dtype = torch.float16) -> None:
        '''Abstract Initialization function for Generic LLM Class to setup a CausalLM model and
         tokenizer'''
    @abstractmethod
    def generate_text(self, message_chain: List[Dict[str, str]],
                    max_new_tokens : int = 4096, **kwargs) -> str:

        '''Abstract Method to generate text from a prompt using the LLM'''

class HFLLM(LLM):
    '''Base LLM Class to handle the tokenizer and inference model for the API'''
    def __init__(self, model_name: str, device_map: str = "auto",
                 torch_dtype: torch.dtype = torch.float16) -> None:
        '''Initialization function for Generic LLM Class to setup a CausalLM model and tokenizer'''
        bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(
            load_in_4bit=True,
        )

        self.tokenizer: AutoTokenizer= AutoTokenizer.from_pretrained(model_name,
            eos_token = self.eos_token,
            pad_token = self.pad_token,
            padding_side = self.padding_side,
            use_fast = True
        )

        self.model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map = device_map,
            torch_dtype = torch_dtype,
            trust_remote_code = True,
            quantization_config = bnb_config
        )
    def generate_text(self, message_chain: List[Dict[str, str]],
                    max_new_tokens : int = 4098,
                    **kwargs
                    ) -> str:
        '''Function to generate text from a prompt using the LLM'''
        #process kwargs
        num_return_sequences: int = \
            kwargs['num_return_sequences'] if 'num_return_sequences' in kwargs else 1
        max_length: int = kwargs['max_length'] if 'max_length' in kwargs else 4096
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
            skip_prompt=True,
            skip_special_tokens=False)

        outputs: torch.Tensor = self.model.generate(
            **input_ids,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            num_return_sequences=num_return_sequences
        )
        output_text: str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text
if os.name != 'nt':
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
                self.model: vllm.AsyncLLMEngine = vllm.AsyncLLMEngine.from_engine_args(engine_args)
            else:
                self.model: vllm.LLMEngine = vllm.LLMEngine.from_engine_args(engine_args)
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
            max_new_tokens : int = 4096,
            **kwargs
        ) -> str:
            '''Function to generate text from a prompt using the LLM'''
            prompt: str = self.message_chain_to_string(message_chain)
            request_id: str = str(uuid.uuid4().hex)
            sampling_params : vllm.SamplingParams = vllm.SamplingParams(
                temperature=0.2,
                top_p=0.95,
                max_tokens=max_new_tokens
            )
            model_response: vllm.RequestOutput = self.model.generate(prompt,
                sampling_params, request_id)
            if self.streaming:
                async def stream_response(
                        model_response: vllm.RequestOutput
                        ) -> AsyncGenerator[bytes, None]:
                    '''Function to stream the response from the LLM'''
                    async for request_out in model_response:
                        prompt: str = request_out.prompt
                        text_outputs = [
                            prompt + token.text for token in request_out.outputs
                        ]
                        ret: Dict[str, List[str]] = {"text": text_outputs}
                        yield (json.dumps(ret, ensure_ascii=True).encode('utf-8'))
                return stream_response(model_response)
            else:
                return model_response
class SolarLLM(HFLLM):
    '''LLM Subclass for the Upstage Solar-10.7b-Instruct-v1.0 model'''
    def __init__(self, **kwargs) -> None:
        '''Constructor for SolarLLM. Just calls the superclass constructor
        with the correct model name'''
        super().__init__("Upstage/SOLAR-10.7b-Instruct-v1.0", **kwargs)
class ZephyrLLM(HFLLM):
    '''LLM Subclass for the HuggingFaceH4/zephyr-7b-beta model'''
    def __init__(self, **kwargs) -> None:
        '''Constructor for ZephyrLLM. Just calls the superclass constructor
        with the correct model name'''
        if 'torch_dtype' not in kwargs: 
            kwargs['torch_dtype'] = torch.bfloat16
        super().__init__("HuggingFaceH4/zephyr-7b-beta", **kwargs)


if __name__ == "__main__":
    llm: LLM = ZephyrLLM(device_map = 'auto')
    messages: List[Dict[str, str]] = [
        {
            "role": "system", 
            "content": ("You are a DnD expert who specializes in world building and"
                        "character creation. You answer questions completely and in detail.")
        },
        {
            "role": "user",
            "content": ("Help me create a character based on Michael from The Good Place. To"
                        " elaborate. I don't want a character exactly like the Michael."
                        " Instead, I want a character whose job it was to torcher humans in the "
                        "afterlife. At one point, the character was cursed by a deity and made to "
                        "live for a time as a heroic adventurer. The chararcter would be a "
                        "grumbly 40-something man who is very cynical and sarcastic. He would be "
                        "slow to help and quick to judge and mock those the party wanted to help. "
                        "But it wouldn't come from a place of malice, but rather from a place of "
                        "ignorance. Help me build the character based off 5e rules. I want a race, "
                        "background. For a class I'm thinking something intelligence based either "
                        "wizard or artificer but I'm open to suggestions. I'm also open to "
                        "subclasses. I want to make sure the character is balanced and fun to play."
                        " Remember that this character should be evil to start."
            )
        }
    ]

    response: str = llm.generate_text(messages)
    print(response)
