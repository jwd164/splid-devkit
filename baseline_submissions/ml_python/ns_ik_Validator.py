import pandas as pd

INCLINATION_DIR = "Inclination Direction"

class NS_IK_Validator:
    
    def apply_validator(self, results, data):
        validate_results = results[(results['Node']=='IK') & (results['Direction']=='NS') ]
        to_drop = []
        for result_index, result in validate_results.iterrows():
            event_data = data[(data['ObjectID']==result['ObjectID']) & (data['TimeIndex']>(result['TimeIndex']-5)) & (data['TimeIndex']<(result['TimeIndex']+5))]
            #TODO : Should I move the event to the data point?
            validated = True
            direction_sum = event_data[INCLINATION_DIR].sum()
            if (direction_sum>=8) | (direction_sum<=-8):
                validated = False
            if not validated:
                to_drop.append(result_index)
                
        for index in to_drop:
            results = results.drop(index)
        return results