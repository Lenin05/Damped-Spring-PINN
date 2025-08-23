from pinn import create_data, test_model, train_ia_model, train_pinn_model

PROJECTS = {
    "Create_Data": create_data.run,
    "Train_Model_IA": train_ia_model.run,
    "Train_Model_PINN": train_pinn_model.run,
    "Test_Model": test_model.run,
}

if __name__ == "__main__":
    project_name = "Test_Model"  # Cambia aquí o pasa por argumento

    try:
        PROJECTS[project_name]()
    except KeyError:
        print(
            f"❌ Project '{project_name}' not found. Available: {list(PROJECTS.keys())}"
        )
