class ProjectBaseError(Exception):
    """Base class for all custom exceptions in the project."""
    def __init__(self, message="An error occurred in the pipeline."):
        super().__init__(message)

class DataIngestionError(ProjectBaseError):
    """Raised when data ingestion fails."""
    def __init__(self, message="Failed during data ingestion."):
        super().__init__(message)

class DataTransformationError(ProjectBaseError):
    """Raised when data transformation fails."""
    def __init__(self, message="Failed during data transformation."):
        super().__init__(message)

class ModelTrainingError(ProjectBaseError):
    """Raised when model training fails."""
    def __init__(self, message="Failed during model training."):
        super().__init__(message)

class PredictionError(ProjectBaseError):
    """Raised when prediction fails."""
    def __init__(self, message="Failed during model prediction."):
        super().__init__(message)
