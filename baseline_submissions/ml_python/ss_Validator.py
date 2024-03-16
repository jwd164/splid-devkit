import pandas as pd

class SS_Validator:
    def apply_validator(self, results, data):
        mask = (results['Node']=='SS')&(results['TimeIndex']!=0)
        stripped_results = results[~mask]
        merged_results = stripped_results.copy()
        merged_results = merged_results.sort_values(by=['ObjectID', 'TimeIndex']).reset_index(drop=True)
        return merged_results