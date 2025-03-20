from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import torch
from monai.apps import get_logger
from monai.config import IgniteInfo
from monai.handlers.tensorboard_handlers import SummaryWriterX
from monai.utils import min_version, optional_import, CommonKeys
from tensorboardX import SummaryWriter

import lp_inspect as lp

Events, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Events")
if TYPE_CHECKING:
    from ignite.engine import Engine
else:
    Engine, _ = optional_import(
        "ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine", as_type="decorator"
    )

DEFAULT_KEY_VAL_FORMAT = "{}: {:.4f} "

class LinearProbeHandler:
    """
    LinearProbeHandler defines a set of Ignite Event-handlers to do linear probing
    It can either take the direct output from model inference or attach to any of the model's feature layers given by name.

    Note that if ``name`` is None, this class will leverage `engine.logger` as the logger, otherwise,
    ``logging.getLogger(name)`` is used. In both cases, it's important to make sure that the logging level is at least
    ``INFO``. To change the level of logging, please call ``import ignite; ignite.utils.setup_logger(name)``
    (when ``name`` is not None) or ``engine.logger = ignite.utils.setup_logger(engine.logger.name, reset=True)``
    (when ``name`` is None) before running the engine with this handler attached.

    Default behaviors:
        - When EPOCH_COMPLETED, trains a linear probe on the features extracted from the model.

    Usage example::
        import ignite
        import torch
        from monai.handlers import from_engine
        from monai.utils import CommonKeys

        from monai_handlers.LinearProbeHandler import LinearProbeHandler

        trainer = ignite.engine.Engine(lambda eng,batch: batch)
        LinearProbeHandler(**dict(epochs=1), output_transform=from_engine([CommonKeys.PRED, CommonKeys.LABEL])) \
            .attach(trainer)
        samples = [{CommonKeys.PRED: torch.rand((128, 10)), CommonKeys.LABEL: torch.randint(0, 9, (128,)).detach().numpy()}]
        trainer.run(samples, max_epochs=1)
    """

    def __init__(
            self,
            summary_writer: SummaryWriter | SummaryWriterX | None = None,
            output_transform: Callable = lambda x: x[0],
            feature_layer_name : str | None = None,
            global_epoch_transform: Callable = lambda x: x,
            every_n_epochs: int = 1,
            state_attributes: Sequence[str] | None = None,
            name: str | None = "LinearProbeHandler",
            out_dir: str = "out",
            model: torch.nn.Module | None = None,
            **kwargs: Any,
    ) -> None:
        """

        Args:
            output_transform: a callable that is used to transform the
                ``ignite.engine.state.output`` into a scalar to print, or a dictionary of {key: scalar}.
                In the latter case, the output string will be formatted as key: value.
                By default, this value logging happens when every iteration completed.
                The default behavior is to print loss from output[0] as output is a decollated list,
                and we replicated loss value for every item of the decollated list.
                `engine.state` and `output_transform` inherit from the ignite concept:
                https://pytorch.org/ignite/concepts.html#state, explanation and usage example are in the tutorial:
                https://github.com/Project-MONAI/tutorials/blob/master/modules/batch_output_transform.ipynb.
            global_epoch_transform: a callable that is used to customize global epoch number.
                For example, in evaluation, the evaluator engine might want to print synced epoch number
                with the trainer engine.
            state_attributes: expected attributes from `engine.state`, if provided, will extract them
                when epoch completed.
            name: identifier of `logging.logger` to use, if None, defaulting to ``engine.logger``.
            kwargs: additional arguments to be passed to the linear_probe function, see linear_probe.py for more details

        """

        self.output_transform = output_transform
        self.global_epoch_transform = global_epoch_transform
        self.state_attributes = state_attributes
        self.logger = get_logger(name)  # type: ignore
        self.summary_writer = summary_writer
        self.iteration_predictions = []
        self.iteration_labels = []
        self.model = model
        self.feature_layer_name = feature_layer_name
        self.hook_handle = None
        self.every_n_epochs = every_n_epochs
        self.out_dir = out_dir
        self.kwargs = kwargs

        if (self.model and not self.feature_layer_name) or (self.feature_layer_name and not self.model):
            raise ValueError("Either none of or both model and feature_layer_name must be provided.")

    def attach(self, engine: Engine) -> None:
        """
        Register a set of Ignite Event-Handlers to a specified Ignite engine.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        engine.add_event_handler(Events.ITERATION_COMPLETED, self.iteration_completed)
        engine.add_event_handler(Events.EPOCH_COMPLETED, self.epoch_completed)

        if not engine.has_event_handler(self.exception_raised, Events.EXCEPTION_RAISED):
            engine.add_event_handler(Events.EXCEPTION_RAISED, self.exception_raised)

        if self.model and self.feature_layer_name:
            engine.add_event_handler(Events.STARTED, self._register_hook)
            engine.add_event_handler(Events.COMPLETED, self._unregister_hook)

    def epoch_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation epoch completed Event.
        Print epoch level log, default values are from Ignite `engine.state.metrics` dict.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        if not engine.state.epoch % self.every_n_epochs == 0:
            return

        if self.iteration_predictions:
            # initialize a new summarywriter
            # otherwise you won't be able to distinguish scalars from multiple calls to lp_eval
            writer= SummaryWriter(logdir=self.summary_writer.get_logdir() + f"_lp_epoch{engine.state.epoch}")
            lp.lp_eval(data=[{CommonKeys.IMAGE: i, CommonKeys.LABEL: l} for (i,l) in zip(self.iteration_predictions, self.iteration_labels)],
                            writer=writer,
                            out_dir=self.out_dir,
                            **self.kwargs
                            )
            self.iteration_predictions = []
            self.iteration_labels = []
        else:
            self.logger.warning("No features to make UMAP from")

    def iteration_completed(self, engine: Engine) -> None:
        """
        Handler for train or validation/evaluation iteration completed Event.
        Print iteration level log, default values are from Ignite `engine.state.output`.

        Args:
            engine: Ignite Engine, it can be a trainer, validator or evaluator.

        """
        output = self.output_transform(engine.state.output)
        if not isinstance(output, tuple):
            raise ValueError("output_transform must return a tuple of (predictions, labels).")
        predictions = output[0]
        labels = output[1]
        self.iteration_labels.extend(labels)
        # if there is no model or feature_layer_name, we collect the predictions directly from the output
        # otherwise, it will be done in the hook
        if not (self.model or self.feature_layer_name):
            self.iteration_predictions.extend(predictions)

    def exception_raised(self, _engine: Engine, e: Exception) -> None:
        """
        Handler for train or validation/evaluation exception raised Event.
        Print the exception information and traceback. This callback may be skipped because the logic
        with Ignite can only trigger the first attached handler for `EXCEPTION_RAISED` event.

        Args:
            _engine: Ignite Engine, unused argument.
            e: the exception caught in Ignite during engine.run().

        """
        self.logger.exception(f"Exception: {e}")
        raise e

    def _unregister_hook(self, engine: Engine) -> None:
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
            self.logger.debug(f"Hook removed from layer: {self.feature_layer_name}")

    def _register_hook(self):
        """
        Register a forward hook to capture the output of the specified layer.
        """

        def hook(_, __, output):
            self.iteration_predictions.extend(output.clone().detach().cpu().numpy().squeeze())

        for name, module in self.model.named_modules():
            if name == self.feature_layer_name:
                self.hook_handle = module.register_forward_hook(hook)
                self.logger.debug(f"Hook registered to layer: {name}")
                return
        raise ValueError(f"Layer '{self.feature_layer_name}' not found in the model.")
