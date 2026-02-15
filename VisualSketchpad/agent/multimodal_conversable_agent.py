# import copy
# from typing import Dict, List, Optional, Tuple, Union

# from autogen.oai.client import OpenAIWrapper
# from autogen.agentchat import Agent, ConversableAgent
# from autogen.agentchat.contrib.img_utils import (
#     gpt4v_formatter,
#     message_formatter_pil_to_b64,
# )
# from autogen.code_utils import content_str

# from autogen._pydantic import model_dump

# DEFAULT_LMM_SYS_MSG = """You are a helpful AI assistant."""
# DEFAULT_MODEL = "gpt-4-turbo"


# class MultimodalConversableAgent(ConversableAgent):
#     DEFAULT_CONFIG = {
#         "model": DEFAULT_MODEL,
#     }

#     def __init__(
#         self,
#         name: str,
#         system_message: Optional[Union[str, List]] = DEFAULT_LMM_SYS_MSG,
#         is_termination_msg: str = None,
#         *args,
#         **kwargs,
#     ):
#         """
#         Args:
#             name (str): agent name.
#             system_message (str): system message for the OpenAIWrapper inference.
#                 Please override this attribute if you want to reprogram the agent.
#             **kwargs (dict): Please refer to other kwargs in
#                 [ConversableAgent](../conversable_agent#__init__).
#         """
#         super().__init__(
#             name,
#             system_message,
#             is_termination_msg=is_termination_msg,
#             *args,
#             **kwargs,
#         )
#         # call the setter to handle special format.
#         self.update_system_message(system_message)
#         self._is_termination_msg = (
#             is_termination_msg
#             if is_termination_msg is not None
#             else (lambda x: content_str(x.get("content")) == "TERMINATE")
#         )

#         # Override the `generate_oai_reply`
#         self.replace_reply_func(ConversableAgent.generate_oai_reply, MultimodalConversableAgent.generate_oai_reply)
#         self.replace_reply_func(
#             ConversableAgent.a_generate_oai_reply,
#             MultimodalConversableAgent.a_generate_oai_reply,
#         )

#     def update_system_message(self, system_message: Union[Dict, List, str]):
#         """Update the system message.

#         Args:
#             system_message (str): system message for the OpenAIWrapper inference.
#         """
#         self._oai_system_message[0]["content"] = self._message_to_dict(system_message)["content"]
#         self._oai_system_message[0]["role"] = "system"

#     @staticmethod
#     def _message_to_dict(message: Union[Dict, List, str]) -> Dict:
#         """Convert a message to a dictionary. This implementation
#         handles the GPT-4V formatting for easier prompts.

#         The message can be a string, a dictionary, or a list of dictionaries:
#             - If it's a string, it will be cast into a list and placed in the 'content' field.
#             - If it's a list, it will be directly placed in the 'content' field.
#             - If it's a dictionary, it is already in message dict format. The 'content' field of this dictionary
#             will be processed using the gpt4v_formatter.
#         """
#         if isinstance(message, str):
#             return {"content": gpt4v_formatter(message, img_format="pil")}
#         if isinstance(message, list):
#             return {"content": message}
#         if isinstance(message, dict):
#             assert "content" in message, "The message dict must have a `content` field"
#             if isinstance(message["content"], str):
#                 message = copy.deepcopy(message)
#                 message["content"] = gpt4v_formatter(message["content"], img_format="pil")
#             try:
#                 content_str(message["content"])
#             except (TypeError, ValueError) as e:
#                 print("The `content` field should be compatible with the content_str function!")
#                 raise e
#             return message
#         raise ValueError(f"Unsupported message type: {type(message)}")

#     def generate_oai_reply(
#         self,
#         messages: Optional[List[Dict]] = None,
#         sender: Optional[Agent] = None,
#         config: Optional[OpenAIWrapper] = None,
#     ) -> Tuple[bool, Union[str, Dict, None]]:
#         """Generate a reply using autogen.oai."""
#         client = self.client if config is None else config
#         if client is None:
#             return False, None
#         if messages is None:
#             messages = self._oai_messages[sender]

#         messages_with_b64_img = message_formatter_pil_to_b64(self._oai_system_message + messages)

#         # TODO: #1143 handle token limit exceeded error
#         response = client.create(context=messages[-1].pop("context", None), messages=messages_with_b64_img)

#         # TODO: line 301, line 271 is converting messages to dict. Can be removed after ChatCompletionMessage_to_dict is merged.
#         extracted_response = client.extract_text_or_completion_object(response)[0]
#         if not isinstance(extracted_response, str):
#             extracted_response = model_dump(extracted_response)
#         return True, extracted_response


import copy
from typing import Dict, List, Optional, Union, Tuple
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from autogen.agentchat import ConversableAgent, Agent
from autogen.code_utils import content_str
from autogen._pydantic import model_dump
from autogen.agentchat.contrib.img_utils import gpt4v_formatter, message_formatter_pil_to_b64

DEFAULT_LMM_SYS_MSG = "You are a helpful AI assistant."
# DEFAULT_MODEL_PATH = "Qwen/Qwen3-VL-32B-Instruct"
DEFAULT_MODEL_PATH = "/rds/general/user/ns1324/home/iso/Qwen3-VL/pretrainedModels/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"


class MultimodalConversableAgent(ConversableAgent):
    DEFAULT_CONFIG = {
        "model": DEFAULT_MODEL_PATH,
    }

    def __init__(
        self,
        model,
        processor,
        name: str,
        system_message: Optional[Union[str, List]] = DEFAULT_LMM_SYS_MSG,
        is_termination_msg: str = None,
        model_path: str = DEFAULT_MODEL_PATH,
        *args,
        **kwargs,
    ):
        # Force llm_config to False to disable OpenAI client
        kwargs['llm_config'] = False
        
        # Initialize parent class
        super().__init__(
            name,
            system_message=system_message,
            is_termination_msg=is_termination_msg,
            *args,
            **kwargs,
        )

        # Create a mock client object for usage tracking compatibility
        class MockClient:
            def __init__(self):
                self.total_usage_summary = {}
                self.actual_usage_summary = {}
            
            def clear_usage_summary(self):
                """Reset usage statistics"""
                self.total_usage_summary = {}
                self.actual_usage_summary = {}
            
            def update_usage(self, input_tokens=0, output_tokens=0):
                """Update token counts (no cost for local models)"""
                if 'total_tokens' not in self.total_usage_summary:
                    self.total_usage_summary['total_tokens'] = 0
                    self.actual_usage_summary['total_tokens'] = 0
                
                total = input_tokens + output_tokens
                self.total_usage_summary['total_tokens'] += total
                self.actual_usage_summary['total_tokens'] += total
            
            def create(self, *args, **kwargs):
                """Mock create method if needed"""
                pass
            
            def __getattr__(self, name):
                """Catch-all for any other methods that might be called"""
                def method(*args, **kwargs):
                    pass
                return method

        self.client = MockClient()

        # # Load local Qwen model & processor
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # print(f"Loading model from {model_path}...")
        # self.model = AutoModelForImageTextToText.from_pretrained(
        #     model_path,
        #     torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        # ).to(device)
        
        # self.processor = AutoProcessor.from_pretrained(model_path)
        # print(f"Model loaded on {device}")
        self.model = model
        self.processor = processor

        # Update system message after parent init
        self.update_system_message(system_message)
        
        # Override termination function if provided
        if is_termination_msg is not None:
            self._is_termination_msg = is_termination_msg

        # Register reply functions - use register_reply instead of replace_reply_func
        self.register_reply(
            [ConversableAgent, None],
            reply_func=MultimodalConversableAgent.generate_oai_reply,
            position=0,
        )

    def update_system_message(self, system_message: Union[Dict, List, str]):
        """Update the system message."""
        if not hasattr(self, '_oai_system_message') or not self._oai_system_message:
            self._oai_system_message = [{"role": "system", "content": ""}]
        
        self._oai_system_message[0]["content"] = self._message_to_dict(system_message)["content"]
        self._oai_system_message[0]["role"] = "system"

    @staticmethod
    def _message_to_dict(message: Union[Dict, List, str]) -> Dict:
        """Convert message to dict format."""
        if isinstance(message, str):
            return {"content": gpt4v_formatter(message, img_format="pil")}
        if isinstance(message, list):
            return {"content": message}
        if isinstance(message, dict):
            assert "content" in message, "The message dict must have a `content` field"
            if isinstance(message["content"], str):
                message = copy.deepcopy(message)
                message["content"] = gpt4v_formatter(message["content"], img_format="pil")
            content_str(message["content"])
            return message
        raise ValueError(f"Unsupported message type: {type(message)}")

    def generate_oai_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config=None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """
        Generate a reply using the local Qwen model.
        This replaces the OpenAI API call but keeps the same method signature.
        """
        if messages is None:
            messages = self._oai_messages[sender]

        # Combine system message with conversation messages
        all_messages = self._oai_system_message + messages
        messages_with_b64_img = message_formatter_pil_to_b64(all_messages)
        print("Prompt going to the model!", messages_with_b64_img)

        try:
            # Prepare inputs for local Qwen model
            inputs = self.processor.apply_chat_template(
                messages_with_b64_img,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            # Generate output
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024)

            # Trim prompt tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # Decode output text
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # Return same structure as OpenAI agent
            if len(output_text) == 1:
                output_text = output_text[0]
            
            return True, output_text
            
        except Exception as e:
            print(f"Error generating reply: {e}")
            return False, None

    async def a_generate_oai_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config=None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """Async version - just calls sync version for now."""
        return self.generate_oai_reply(messages, sender, config)
