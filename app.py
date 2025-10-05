from Frontend.interface import GradioApp
from Backend.prepare_model import initialize_model

if __name__ == "__main__":
    model = initialize_model()
    app = GradioApp(model)
    app.launch()