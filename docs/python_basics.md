# Python 速查

- 列表推导式：`[x * x for x in range(10) if x % 2 == 0]`
- 字典合并（3.9+）：`merged = a | b`
- 数据类：`@dataclass` 自动生成 `__init__` / `__repr__` / `__eq__`
- 虚拟环境：`python -m venv .venv && source .venv/bin/activate`
- 类型标注：`def f(x: int) -> str: ...`，配合 mypy 静态检查
- GIL：CPU 密集用 multiprocessing，IO 密集用 asyncio 或线程
