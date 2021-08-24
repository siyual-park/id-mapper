from typing import TypeVar, Union, Tuple

T = TypeVar('T')

_scalar_or_tuple_any_t = Union[T, Tuple[T, ...]]
_scalar_or_tuple_1_t = Union[T, Tuple[T]]
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_scalar_or_tuple_3_t = Union[T, Tuple[T, T, T]]
_scalar_or_tuple_4_t = Union[T, Tuple[T, T, T, T]]
_scalar_or_tuple_5_t = Union[T, Tuple[T, T, T, T, T]]
_scalar_or_tuple_6_t = Union[T, Tuple[T, T, T, T, T, T]]

size_any_t = _scalar_or_tuple_any_t[int]
size_1_t = _scalar_or_tuple_1_t[int]
size_2_t = _scalar_or_tuple_2_t[int]
size_3_t = _scalar_or_tuple_3_t[int]
size_4_t = _scalar_or_tuple_4_t[int]
size_5_t = _scalar_or_tuple_5_t[int]
size_6_t = _scalar_or_tuple_6_t[int]
