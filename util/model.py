from abc import abstractmethod, ABCMeta


class Model:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_vars(self) -> dict:
        """
        :return: trainable variables and optimizers
        """
        pass
