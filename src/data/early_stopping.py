
import math


class EarlyStopping:

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0
    ):
        """
        Initialize early stopping.

        Args:
            patience: Consecutive non-improving epochs allowed.
            min_delta: Minimum improvement required.
        """

        if (
            not isinstance(patience, int)
            or isinstance(patience, bool)
            or patience < 0
        ):
            raise ValueError(
                "patience must be a non-negative integer"
            )

        if (
            not isinstance(min_delta, (int, float))
            or isinstance(min_delta, bool)
            or not math.isfinite(min_delta)
            or min_delta < 0
        ):
            raise ValueError(
                "min_delta must be a finite non-negative number"
            )

        self.patience = patience
        self.min_delta = float(min_delta)

        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, validation_loss: float):
        """
        Update early stopping state using validation loss.

        Returns:
            True if training should stop, otherwise False.
        """

        if (
            not isinstance(validation_loss, (int, float))
            or isinstance(validation_loss, bool)
            or not math.isfinite(validation_loss)
            or validation_loss < 0
        ):
            raise ValueError(
                "validation_loss must be a finite non-negative number"
            )

        validation_loss = float(validation_loss)

        # Check whether meaningful improvement occurred.
        if validation_loss < self.best_loss - self.min_delta:

            self.best_loss = validation_loss
            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def state_dict(self):
        """
        Return the current early stopping state.
        """

        return {
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best_loss": self.best_loss,
            "counter": self.counter,
            "early_stop": self.early_stop
        }

    def load_state_dict(self, state_dict):
        """
        Restore an existing early stopping state.
        """

        required_keys = {
            "patience",
            "min_delta",
            "best_loss",
            "counter",
            "early_stop"
        }

        if not isinstance(state_dict, dict):
            raise TypeError("state_dict must be a dictionary")

        missing_keys = required_keys - state_dict.keys()

        if missing_keys:
            raise ValueError(
                f"Missing state keys: {missing_keys}"
            )

        patience = state_dict["patience"]
        min_delta = state_dict["min_delta"]
        best_loss = state_dict["best_loss"]
        counter = state_dict["counter"]
        early_stop = state_dict["early_stop"]

        if (
            not isinstance(patience, int)
            or isinstance(patience, bool)
            or patience < 0
        ):
            raise ValueError("Invalid patience in state_dict")

        if (
            not isinstance(min_delta, (int, float))
            or isinstance(min_delta, bool)
            or not math.isfinite(min_delta)
            or min_delta < 0
        ):
            raise ValueError("Invalid min_delta in state_dict")

        if (
            not isinstance(best_loss, (int, float))
            or math.isnan(best_loss)
            or best_loss < 0
        ):
            raise ValueError("Invalid best_loss in state_dict")

        if (
            not isinstance(counter, int)
            or isinstance(counter, bool)
            or counter < 0
        ):
            raise ValueError("Invalid counter in state_dict")

        if not isinstance(early_stop, bool):
            raise ValueError("Invalid early_stop in state_dict")

        self.patience = patience
        self.min_delta = float(min_delta)
        self.best_loss = float(best_loss)
        self.counter = counter
        self.early_stop = early_stop
