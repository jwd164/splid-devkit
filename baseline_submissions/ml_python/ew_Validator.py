import pandas as pd

DELTA_COLUMN = "Delta_SemimajorAxis"
MANEUVER_THRESHOLD = 3000

class EW_Validator:
    
    def apply_validator(self, results, data):
        validate_results = results[(results['Node']!='SS') & (results['Direction']=='EW')]
        to_drop = []
        for result_index, result in validate_results.iterrows():
            event_data = data[(data['ObjectID']==result['ObjectID']) & (data['TimeIndex']>(result['TimeIndex']-5)) & (data['TimeIndex']<(result['TimeIndex']+5))]
            #TODO : Should I move the event to the data point?
            fits_threshold = (event_data[DELTA_COLUMN].abs()>MANEUVER_THRESHOLD).any()
            if not fits_threshold:
                to_drop.append(result_index)
                
        for index in to_drop:
            results = results.drop(index)
        return results