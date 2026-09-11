class ShadowTrainerError(RuntimeError):
    """Base error with a stable machine-readable code."""

    code = "shadow_trainer_error"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class ConfigurationError(ShadowTrainerError):
    code = "configuration_error"


class AdmissionError(ShadowTrainerError):
    code = "admission_rejected"


class IntegrityError(ShadowTrainerError):
    code = "integrity_error"


class DependencyError(ShadowTrainerError):
    code = "dependency_error"
