import delia
print(delia.__file__)
from delia.tools import builtins
print(builtins.__file__)
import asyncio
print(asyncio.run(builtins.search_code("test", ".")))
