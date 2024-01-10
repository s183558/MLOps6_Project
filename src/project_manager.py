import os

class ProjectManager():
    """
    Holds general information about the project that different methods and classes might need
    """
    def __init__(self):
        self.project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_folder = os.path.join(self.project_folder, "data")
        self.model_folder = os.path.join(self.project_folder, "models")

        # Removes some warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false" 

    def get_project_folder(self) -> str:
        return self.project_folder
    
    def get_data_folder(self) -> str:
        return self.data_folder
    
    def get_model_folder(self) -> str:
        return self.model_folder
