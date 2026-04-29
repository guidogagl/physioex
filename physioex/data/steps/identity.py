from physioex.data.pipeline import PreprocessingStep, CompiledStep


class Identity(PreprocessingStep):
    """Passthrough; does nothing."""

    def spec(self) -> str:
        return "Identity()"

    def compile(self, fs_in: float) -> CompiledStep:
        return CompiledStep(apply=lambda x: x, fs_out=fs_in)
