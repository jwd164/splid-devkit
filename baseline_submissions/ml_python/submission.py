"""
    This is the main script that will create the predictions on test data and save 
    a predictions file.
"""
import time
from pathlib import Path
import pickle

import utils
import ew_Validator
import ns_id_Validator
import ns_ik_Validator
import ss_Validator

start_time = time.time()

# INPUT/OUTPUT PATHS WITHIN THE DOCKER CONTAINER
TRAINED_MODEL_DIR = Path('/trained_model/')
TEST_DATA_DIR = Path('/dataset/test/')
TEST_PREDS_FP = Path('/submission/submission.csv')


# Rest of configuration, specific to this submission
delta_column = "Delta_SemimajorAxis"

ss_validator = ss_Validator.SS_Validator()
ew_validator = ew_Validator.EW_Validator()
ns_ik_validator = ns_ik_Validator.NS_IK_Validator()
ns_id_validator = ns_id_Validator.NS_ID_Validator()

feature_cols = [
    "Eccentricity",
    "Semimajor Axis (m)",
    "Inclination (deg)",
    "RAAN (deg)",
    "Argument of Periapsis (deg)",
    "True Anomaly (deg)",
    "Latitude (deg)",
    "Longitude (deg)",
    "Altitude (m)",
    # "X (m)",
    # "Y (m)",
    "Z (m)",
    "Vx (m/s)",
    "Vy (m/s)",
    "Vz (m/s)",
    delta_column,
]

lag_steps = 0

test_data, updated_feature_cols = utils.tabularize_data(
    TEST_DATA_DIR, feature_cols, lag_steps=lag_steps)

print("Data Tabularization Complete")

# Load the trained models (don't use the utils module, use pickle)
model_EW = pickle.load(open(TRAINED_MODEL_DIR / 'model_EW_v3.pkl', 'rb'))
le_EW = pickle.load(open(TRAINED_MODEL_DIR / 'le_EW_v3.pkl', 'rb'))

# Make predictions on the test data for EW
test_data['Predicted_EW'] = le_EW.inverse_transform(
    model_EW.predict(test_data[updated_feature_cols])
)

# Trash the model to free the memory
model_EW = None
le_EW = None
print("EW Predictions Complete")

model_NS = pickle.load(open(TRAINED_MODEL_DIR / 'model_NS_v3.pkl', 'rb'))
le_NS = pickle.load(open(TRAINED_MODEL_DIR / 'le_NS_v3.pkl', 'rb'))

# Make predictions on the test data for NS
test_data['Predicted_NS'] = le_NS.inverse_transform(
    model_NS.predict(test_data[updated_feature_cols])
)

print("NS Predictions Complete")

# Print the first few rows of the test data with predictions for both EW and NS
test_results = utils.convert_classifier_output(test_data)


validated_results = ss_validator.apply_validator(test_results, test_data)

validated_results = ew_validator.apply_validator(validated_results, test_data)


validated_results = ns_ik_validator.apply_validator(validated_results, test_data)


validated_results = ns_id_validator.apply_validator(validated_results, test_data)

sorted_test_results = validated_results.sort_values(by=['ObjectID', 'TimeIndex']).reset_index(drop=True)


# Save the test results to a csv file to be submitted to the challenge
sorted_test_results.to_csv(TEST_PREDS_FP, index=False)
print("Saved predictions to: {}".format(TEST_PREDS_FP))

time.sleep(360) # TEMPORARY FIX TO OVERCOME EVALAI BUG