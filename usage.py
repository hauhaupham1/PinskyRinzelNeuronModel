# In your main script or notebook
from PRmodel_Motoneuron.losses import optimize_model, comprehensive_loss

# Option 1: Perform full optimization
optimized_params = optimize_model(
    yaml_file="PRmodel_Motoneuron/motoneuron.yaml",
    learning_rate=0.01,
    n_iterations=500  # Start with fewer iterations for testing
)

# Option 2: Evaluate current parameters
from PRmodel_Motoneuron.MotoneuronModel import MotoneuronModel

model = MotoneuronModel(yaml_file="PRmodel_Motoneuron/motoneuron.yaml")
current_params = {
    'g_c': model.g_c,
    'g_Na': model.g_Na,
    'g_DR': model.g_DR,
    'g_Ca': model.g_Ca,
    'g_AHP': model.g_AHP,
    'g_L_soma': model.g_L_soma,
    'g_L_dend': model.g_L_dend,
}

total_loss, component_losses = comprehensive_loss(current_params)
print(f"Total loss: {total_loss}")
for key, value in component_losses.items():
    print(f"{key} loss: {value}")