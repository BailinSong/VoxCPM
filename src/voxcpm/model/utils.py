from typing import List
import torch
from transformers import PreTrainedTokenizer


def mask_multichar_chinese_tokens(tokenizer: PreTrainedTokenizer):
    """Create a tokenizer wrapper that converts multi-character Chinese tokens to single characters.
    
    This function creates a wrapper around the provided tokenizer that automatically
    splits multi-character Chinese tokens into individual characters. This is useful
    for ensuring consistent tokenization of Chinese text.
    
    Args:
        tokenizer: The base tokenizer to wrap
        
    Returns:
        A CharTokenizerWrapper instance that handles multi-character Chinese tokens
        
    Example:
        >>> from transformers import LlamaTokenizerFast
        >>> tokenizer = LlamaTokenizerFast.from_pretrained("path/to/tokenizer")
        >>> wrapped_tokenizer = mask_multichar_chinese_tokens(tokenizer)
        >>> tokens = wrapped_tokenizer("你好世界")
    """
    # Pre-compute multi-character tokens (length >= 2, pure Chinese characters)
    multichar_tokens = {
        token for token in tokenizer.vocab.keys() 
        if len(token) >= 2 and all("\u4e00" <= c <= "\u9fff" for c in token)
    }

    class CharTokenizerWrapper:
        """Wrapper class for tokenizers that handles multi-character Chinese tokens.
        
        This wrapper automatically splits multi-character Chinese tokens into
        individual characters while preserving the original tokenizer's interface.
        """
        
        def __init__(self, base_tokenizer: PreTrainedTokenizer) -> None:
            """Initialize the wrapper with a base tokenizer.
            
            Args:
                base_tokenizer: The tokenizer to wrap
            """
            self.tokenizer = base_tokenizer
            self.multichar_tokens = multichar_tokens

        def tokenize(self, text: str, **kwargs) -> List[str]:
            """Tokenize text and split multi-character Chinese tokens into single characters.
            
            Args:
                text: Input text to tokenize
                **kwargs: Additional arguments passed to the base tokenizer
                
            Returns:
                List of processed tokens with multi-character Chinese tokens split
                
            Example:
                >>> wrapper = CharTokenizerWrapper(tokenizer)
                >>> tokens = wrapper.tokenize("你好世界")
                >>> # Returns ["你", "好", "世", "界"] instead of ["你好", "世界"]
            """
            if not isinstance(text, str):
                raise TypeError(f"Expected string input, got {type(text)}")
                
            tokens = self.tokenizer.tokenize(text, **kwargs)
            processed = []
            
            for token in tokens:
                # Remove possible subword prefix
                clean_token = token.replace("▁", "")

                if clean_token in self.multichar_tokens:
                    # Split multi-character token into single characters
                    chars = list(clean_token)
                    processed.extend(chars)
                else:
                    processed.append(token)
                    
            return processed

        def __call__(self, text: str, **kwargs) -> List[int]:
            """Call the tokenizer and return token IDs.
            
            This method provides the same interface as the original tokenizer
            but with multi-character Chinese token handling.
            
            Args:
                text: Input text to tokenize
                **kwargs: Additional arguments passed to the base tokenizer
                
            Returns:
                List of token IDs
                
            Raises:
                TypeError: If input is not a string
                ValueError: If tokenization fails
            """
            try:
                tokens = self.tokenize(text, **kwargs)
                result = self.tokenizer.convert_tokens_to_ids(tokens)
                return result
            except Exception as e:
                raise ValueError(f"Tokenization failed: {str(e)}") from e

    return CharTokenizerWrapper(tokenizer)


def _is_directml_available():
    """检查 DirectML 设备是否可用"""
    try:
        import torch
        import platform
        # 检查是否在 Windows 上
        if platform.system() != "Windows":
            return False
        
        # 检查是否有 DirectML 后端支持
        if hasattr(torch.backends, 'directml') and torch.backends.directml.is_available():
            return True
        
        # 尝试创建 DirectML 张量
        try:
            torch.tensor([1.0], device='directml:0')
            return True
        except:
            return False
    except:
        return False

def _is_hip_available():
    """检查 HIP 设备是否可用"""
    try:
        import torch
        # 检查是否有 HIP 后端支持
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return True
        # 检查是否有 AMD GPU 相关的环境变量
        import os
        if os.environ.get('ROCM_PATH') is not None:
            return True
        # 尝试创建 HIP 张量
        try:
            torch.tensor([1.0], device='hip:0')
            return True
        except:
            return False
    except:
        return False


def get_dtype(dtype: str):
    """
    获取数据类型，对于不支持的数据类型自动降级
    """
    # 检查当前设备
    device = None
    if torch.cuda.is_available():
        device = "cuda"
    elif _is_directml_available():
        device = "directml"
    elif _is_hip_available():
        device = "hip"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    # MPS 设备不支持 bfloat16，自动降级为 float32
    if device == "mps" and dtype in ["bfloat16", "bf16"]:
        print(f"Warning: MPS device does not support {dtype}, falling back to float32")
        return torch.float32
    
    # DirectML 设备对 bfloat16 支持有限，建议使用 float32
    if device == "directml" and dtype in ["bfloat16", "bf16"]:
        print(f"Warning: DirectML device has limited support for {dtype}, falling back to float32")
        return torch.float32
    
    # HIP 设备对 bfloat16 支持有限，建议使用 float32
    if device == "hip" and dtype in ["bfloat16", "bf16"]:
        print(f"Warning: HIP device has limited support for {dtype}, falling back to float32")
        return torch.float32
    
    if dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "bf16":
        return torch.bfloat16
    elif dtype == "float16":
        return torch.float16
    elif dtype == "fp16":
        return torch.float16
    elif dtype == "float32":
        return torch.float32
    elif dtype == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
