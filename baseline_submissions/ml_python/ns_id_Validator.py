import pandas as pd


class NS_ID_Validator:
    
    def apply_validator(self, results, data):
        mask = (results['Node']=='ID') & (results['Direction']=='NS')
        validated_results = results[~mask].copy()
        return validated_results