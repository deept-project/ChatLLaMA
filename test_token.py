from llama.model.tokenization_llama import LLaMATokenizer
from llama.model.tokenization_llama_fast import LLaMATokenizerFast

# tokenizer = LLaMATokenizerFast.from_pretrained("weights/LLaMA-base")
# out = tokenizer.tokenize("谢谢你的回复！GitHub账号是akemimadoka。")
# print(out)

tokenizer = LLaMATokenizer.from_pretrained("weights/LLaMA-base")
out = tokenizer.tokenize("谢谢你的回复！GitHub账号是akemimadoka。")
print(out)