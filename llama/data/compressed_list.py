
CHUNK_MAX_LEN = 512
import pickle
import sys
from typing import Any, Dict, List, Tuple
import blosc

class uchunk_iterator(object):
    def __init__(self, data) -> None:
        self.data = data
        self.index = 0

    def __next__(self):
        if self.index < len(self.data):
            ret = self.data[self.index]
            self.index += 1
            return ret
        else:
            raise StopIteration

class uchunk(object):
    def __init__(self, data) -> None:
        self.data = data
    
    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return uchunk_iterator(self.data)
    
    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value
    
    def to_compressed(self):
        out = cchunk()
        bin_data = pickle.dumps(self.data)
        out.compressed_data = blosc.compress(bin_data, typesize=1)
        out.item_num = len(self.data)
        return out
    
    def append(self, value):
        self.data.append(value)

class cchunk(object):
    def __init__(self) -> None:
        self.compressed_data = None
        self.item_num = 0

    def __len__(self) -> int:
        return self.item_num

    def __getitem__(self, key):
        if self.compressed_data is None or self.item_num == 0:
            raise IndexError("key out of range")
        if isinstance(key, int):
            items = self.get_udata()
            return items[key]
        else:
            print("error get item")

    def __setitem__(self, key, value):
        if self.compressed_data is None or self.item_num == 0:
            raise IndexError("key out of range")
        if isinstance(key, int):
            items = self.get_udata()
            items[key] = value
            self.set_udata(items)
        else:
            print("error get item")
    
    def __sizeof__(self) -> int:
        return sys.getsizeof(self.compressed_data) + sys.getsizeof(self.item_num)
    
    def to_uncompressed(self):
        items = self.get_udata()
        return uchunk(items)

    def get_udata(self):
        if self.compressed_data is None:
            return []
        else:
            raw_data = blosc.decompress(self.compressed_data)
            items = pickle.loads(raw_data)
            return items
    
    def set_udata(self, items):
        bin_data = pickle.dumps(items)
        self.compressed_data = blosc.compress(bin_data, typesize=1)

    def append(self, value):
        items = self.get_udata()
        items.append(value)
        self.set_udata(items)
        self.item_num += 1

class clist_iterator(object):
    def __init__(self, clist_obj) -> None:
        self.clist_obj = clist_obj
        self.index = 0

    def __next__(self):
        if self.index < len(self.clist_obj):
            ret = self.clist_obj[self.index]
            self.index += 1
            return ret
        else:
            raise StopIteration

class clist(object):
    def __init__(self) -> None:
        self.chunks: List[cchunk] = []
        self.cached_chunk: uchunk = None
        self.cached_chunk_id = -1
        self.total_len = 0
    
    def _write_cache(self):
        if self.cached_chunk_id != -1:
            self.chunks[self.cached_chunk_id] = self.cached_chunk.to_compressed()

    def _load_cache(self, index: int):
        self.cached_chunk = self.chunks[index].to_uncompressed()
        self.cached_chunk_id = index
    
    def _get_chunk(self, chunk_id: int):
        if chunk_id == self.cached_chunk_id:
            return self.cached_chunk
        else:
            self._write_cache()
            self._load_cache(chunk_id)
            return self.cached_chunk

    def get_item(self, key: int):
        chunk_id = key // CHUNK_MAX_LEN
        item_id = key % CHUNK_MAX_LEN
        return self._get_chunk(chunk_id)[item_id]
    
    def set_item(self, key: int, value: Any):
        chunk_id = key // CHUNK_MAX_LEN
        item_id = key % CHUNK_MAX_LEN
        chunk = self._get_chunk(chunk_id)
        chunk[item_id] = value
    
    def __sizeof__(self) -> int:
        total_size = 0
        for chunk in self.chunks:
            total_size += sys.getsizeof(chunk)
        return total_size + sys.getsizeof(self.cached_chunk) + \
            sys.getsizeof(self.cached_chunk_id) + sys.getsizeof(self.total_len)

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get_item(key)
        else:
            print("error get item")

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.set_item(key, value)
        else:
            print("error get item")
    
    def __repr__(self) -> str:
        out = []
        out.append("[")
        for i, compressed_chunk in enumerate(self.chunks):
            chunk = self._get_chunk(i)
            for item in chunk:
                out.append(item.__repr__())
                out.append(", ")
        out.pop(-1)
        out.append("]")
        return "".join(out)
    
    def __iter__(self):
        return clist_iterator(self)

    def append(self, value):
        if len(self.chunks) == 0:
            last_chunk_len = CHUNK_MAX_LEN + 100
        else:
            last_chunk = self._get_chunk(len(self.chunks) - 1)
            last_chunk_len = len(last_chunk)

        if last_chunk_len >= CHUNK_MAX_LEN:
            new_chunk = cchunk()
            self.chunks.append(new_chunk)

            chunk = self._get_chunk(len(self.chunks) - 1)
            chunk.append(value)
        else:
            last_chunk.append(value)
        
        self.total_len += 1
