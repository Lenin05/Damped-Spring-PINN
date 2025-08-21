from pinn import create_data, test_model, train_model

PROJECTS = {
    "Create_Data": create_data.run,
    "Train_Model": train_model.run,
    "Test_Model": test_model.run,
}

if __name__ == "__main__":
    project_name = "Create_Data"  # Cambia aquí o pasa por argumento

    try:
        PROJECTS[project_name]()
    except KeyError:
        print(
            f"❌ Project '{project_name}' not found. Available: {list(PROJECTS.keys())}"
        )
