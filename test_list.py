from llama.data.compressed_list import CompressedList
import tqdm
lst = list()
for i in tqdm.tqdm(list(range(100000000))):
    lst.append(i)
# print(lst)