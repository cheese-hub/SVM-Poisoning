import abc
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np

class Attack(abc.ABC):
    """
    Abstract base class for all attack abstract base classes.
    """

    attack_params: List[str] = []
    # The _estimator_requirements define the requirements an estimator must satisfy to be used as a target for an
    # attack. They should be a tuple of requirements, where each requirement is either a class the estimator must
    # inherit from, or a tuple of classes which define a union, i.e. the estimator must inherit from at least one class
    # in the requirement tuple.
    _estimator_requirements: Optional[Union[Tuple[Any, ...], Tuple[()]]] = None

    def __init__(
        self,
        estimator,
    ):
        """
        :param estimator: An estimator.

        """
        super().__init__()

        if self.estimator_requirements is None:
            raise ValueError("Estimator requirements have not been defined in `_estimator_requirements`.")

        # if not self.is_estimator_valid(estimator, self._estimator_requirements):
        #     raise EstimatorError(self.__class__, self.estimator_requirements, estimator)

        self._estimator = estimator
        # self._summary_writer_arg = summary_writer
        # self._summary_writer: Optional[SummaryWriter] = None

        # if isinstance(summary_writer, SummaryWriter):  # pragma: no cover
        #     self._summary_writer = summary_writer
        # elif summary_writer:
        #     self._summary_writer = SummaryWriterDefault(summary_writer)

        # Attack._check_params(self)

    @property
    def estimator(self):
        """The estimator."""
        return self._estimator

    @property
    def summary_writer(self):
        """The summary writer."""
        return self._summary_writer

    @property
    def estimator_requirements(self):
        """The estimator requirements."""
        return self._estimator_requirements

    def set_params(self, **kwargs) -> None:
        """
        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.

        :param kwargs: A dictionary of attack-specific parameters.
        """
        for key, value in kwargs.items():
            if key in self.attack_params:
                setattr(self, key, value)
        self._check_params()

    # def _check_params(self) -> None:

        # if not isinstance(self._summary_writer_arg, (bool, str, SummaryWriter)):
        #     raise ValueError("The argument `summary_writer` has to be either of type bool or str.")

    @staticmethod
    def is_estimator_valid(estimator, estimator_requirements) -> bool:
        """
        Checks if the given estimator satisfies the requirements for this attack.

        :param estimator: The estimator to check.
        :param estimator_requirements: Estimator requirements.
        :return: True if the estimator is valid for the attack.
        """

        for req in estimator_requirements:
            # A requirement is either a class which the estimator must inherit from, or a tuple of classes and the
            # estimator is required to inherit from at least one of the classes
            if isinstance(req, tuple):
                if all(p not in type(estimator).__mro__ for p in req):
                    return False
            elif req not in type(estimator).__mro__:
                return False
        return True

class PoisoningAttack(Attack):
    """
    Abstract base class for poisoning attack classes
    """

    def __init__(self, classifier) -> None:
        """
        :param classifier: A trained classifier (or none if no classifier is needed)
        """
        super().__init__(classifier)

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y=Optional[np.ndarray], **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate poisoning examples and return them as an array. This method should be overridden by all concrete
        poisoning attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y:  Target labels for `x`. Untargeted attacks set this value to None.
        :return: An tuple holding the (poisoning examples, poisoning labels).
        """
        raise NotImplementedError


class PoisoningAttackWhiteBox(PoisoningAttack):
    """
    Abstract base class for poisoning attack classes that have white-box access to the model (classifier object).
    """

    @abc.abstractmethod
    def poison(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate poisoning examples and return them as an array. This method should be overridden by all concrete
        poisoning attack implementations.

        :param x: An array with the original inputs to be attacked.
        :param y: Correct labels or target labels for `x`, depending if the attack is targeted
               or not. This parameter is only used by some of the attacks.
        :return: An tuple holding the `(poisoning_examples, poisoning_labels)`.
        """
        raise NotImplementedError
