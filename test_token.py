from llama.model.tokenization_llama import LLaMATokenizer
from llama.model.tokenization_llama_fast import LLaMATokenizerFast

tokenizer = LLaMATokenizerFast.from_pretrained("checkpoints/LLaMA-base")
out_tokens = tokenizer.tokenize("谢谢你的回复！GitHub账号是akemimadoka。")
out_ids = tokenizer.convert_tokens_to_ids(out_tokens)
out_str = tokenizer.convert_tokens_to_string(out_tokens)
print(out_tokens)
print(out_ids)
print(out_str)

tokenizer = LLaMATokenizer.from_pretrained("checkpoints/LLaMA-base")
out_tokens = tokenizer.tokenize("谢谢你的回复！GitHub账号是akemimadoka。")
out_ids = tokenizer.convert_tokens_to_ids(out_tokens)
out_str = tokenizer.convert_tokens_to_string(out_tokens)
print(out_tokens)
print(out_ids)
print(out_str)