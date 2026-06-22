from typing import Any, Dict, Optional, Tuple



def get_factory_adder() -> Tuple[Any, Dict[str, Any]]:

    classes_dict = {}
    def _add_class(class_: Any, name: Optional[str]=None) -> Any:
        if name is None:
            name = class_.__name__
        classes_dict[name] = class_
        return class_

    def add_class(class_: Any, name: Optional[str]=None) -> Any:
        if not callable(class_):
            name = class_
            def wrapper(class_: Any) -> Any:
                return _add_class(class_, name)
            return wrapper
        else:
            return _add_class(class_)

    return add_class, classes_dict
