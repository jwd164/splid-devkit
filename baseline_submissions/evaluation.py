import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import mean_squared_error
from fastcore.all import *
import argparse


class NodeDetectionEvaluator:
    def __init__(self, ground_truth, participant, tolerance):
        self.ground_truth = ground_truth.copy()
        self.participant = participant.copy()
        self.tolerance = tolerance

    def evaluate_EW(self, object_id):
        gt_object = self.ground_truth[(self.ground_truth['ObjectID'] == object_id) & \
                          (self.ground_truth['Direction'] == 'EW')].copy()
        p_object = self.participant[(self.participant['ObjectID'] == object_id) & \
                                    (self.participant['Direction'] == 'EW')].copy()
        p_object['matched_EW'] = False
        p_object['classification_EW'] = None
        p_object['distance'] = None
        gt_object['classification_EW'] = None
        gt_object['distance'] = None
        tp = 0
        fp = 0
        fn = 0
        node_wrong =0
        type_wrong = 0
        not_matched = 0
        ew_fp = 0
        ew_ik_fp =0
        ew_id_fp =0

        for gt_idx, gt_row in gt_object.iterrows():
            matching_participant_events = p_object[
                (p_object['TimeIndex'] >= gt_row['TimeIndex'] - self.tolerance) &
                (p_object['TimeIndex'] <= gt_row['TimeIndex'] + self.tolerance) &
                (p_object['Direction'] == gt_row['Direction']) & 
                (p_object['matched_EW'] == False)
            ]

            if len(matching_participant_events) > 0:
                p_idx = matching_participant_events.index[0]
                p_row = matching_participant_events.iloc[0]
                distance = p_row['TimeIndex'] - gt_row['TimeIndex']
                if p_row['Node'] == gt_row['Node'] and p_row['Type'] == gt_row['Type']:
                    tp += 1
                    
                    gt_object.loc[gt_idx, 'classification_EW'] = 'TP'
                    gt_object.loc[gt_idx, 'distance'] = distance
                    p_object.loc[p_idx, 'classification_EW'] = 'TP'
                    p_object.loc[p_idx, 'distance'] = distance
                else:
                    fp += 1
                    gt_object.loc[gt_idx, 'classification_EW'] = 'FP'
                    gt_object.loc[gt_idx, 'distance'] = distance
                    p_object.loc[p_idx, 'classification_EW'] = 'FP'
                    p_object.loc[p_idx, 'distance'] = distance
                    if p_row['Node']!= gt_row['Node']:
                        node_wrong += 1
                    if p_row['Type']!= gt_row['Type']:
                        type_wrong +=1
                    if p_row['Direction'] == 'EW':
                        ew_fp +=1
                    if p_row['Direction'] == 'NS':
                        ns_fp +=1
                        if p_row['Node']=='IK':
                            ns_ik_fp+=1
                        if p_row['Node']=='ID':
                            ns_id_fp +=1
                p_object.loc[matching_participant_events.index[0], 'matched_EW'] = True
            else:
                fn += 1
                gt_object.loc[gt_idx, 'classification_EW'] = 'FN'
                

        additional_fp = p_object[~p_object['matched_EW']].copy()
        ew_add_fp = additional_fp[additional_fp['Direction']=='EW'].copy()
        ew_ik_fp += len(additional_fp[additional_fp['Node']=='IK'])
        ew_id_fp += len(additional_fp[additional_fp['Node']=='ID'])
        fp += len(additional_fp)
        ew_fp += len(ew_add_fp)
        not_matched = len(additional_fp)
        p_object.loc[additional_fp.index, 'classification_EW'] = 'FP'

        return tp, fp, fn, gt_object, p_object, node_wrong, type_wrong, not_matched, ew_ik_fp, ew_id_fp
    
    def evaluate_NS(self, object_id):
        gt_object = self.ground_truth[(self.ground_truth['ObjectID'] == object_id) & \
                          (self.ground_truth['Direction'] == 'NS')].copy()
        p_object = self.participant[(self.participant['ObjectID'] == object_id) & \
                                    (self.participant['Direction'] == 'NS')].copy()
        p_object['matched_NS'] = False
        p_object['classification_NS'] = None
        p_object['distance'] = None
        gt_object['classification_NS'] = None
        gt_object['distance'] = None
        tp = 0
        fp = 0
        fn = 0
        node_wrong =0
        type_wrong = 0
        not_matched = 0
        ns_ik_fp =0
        ns_id_fp =0

        for gt_idx, gt_row in gt_object.iterrows():
            matching_participant_events = p_object[
                (p_object['TimeIndex'] >= gt_row['TimeIndex'] - self.tolerance) &
                (p_object['TimeIndex'] <= gt_row['TimeIndex'] + self.tolerance) &
                (p_object['Direction'] == gt_row['Direction']) & 
                (p_object['matched_NS'] == False)
            ]

            if len(matching_participant_events) > 0:
                p_idx = matching_participant_events.index[0]
                p_row = matching_participant_events.iloc[0]
                distance = p_row['TimeIndex'] - gt_row['TimeIndex']
                if p_row['Node'] == gt_row['Node'] and p_row['Type'] == gt_row['Type']:
                    tp += 1
                    
                    gt_object.loc[gt_idx, 'classification_NS'] = 'TP'
                    gt_object.loc[gt_idx, 'distance'] = distance
                    p_object.loc[p_idx, 'classification_NS'] = 'TP'
                    p_object.loc[p_idx, 'distance'] = distance
                else:
                    fp += 1
                    gt_object.loc[gt_idx, 'classification_NS'] = 'FP'
                    gt_object.loc[gt_idx, 'distance'] = distance
                    p_object.loc[p_idx, 'classification_NS'] = 'FP'
                    p_object.loc[p_idx, 'distance'] = distance
                    if p_row['Node']!= gt_row['Node']:
                        node_wrong += 1
                    if p_row['Type']!= gt_row['Type']:
                        type_wrong +=1
                    if p_row['Direction'] == 'NS':
                        if p_row['Node']=='IK':
                            ns_ik_fp+=1
                        if p_row['Node']=='ID':
                            ns_id_fp +=1
                p_object.loc[matching_participant_events.index[0], 'matched_NS'] = True
            else:
                fn += 1
                gt_object.loc[gt_idx, 'classification_NS'] = 'FN'
                

        additional_fp = p_object[~p_object['matched_NS']].copy()
        ns_add_fp = additional_fp[additional_fp['Direction']=='NS'].copy()
        ns_ik_fp += len(additional_fp[additional_fp['Node']=='IK'])
        ns_id_fp += len(additional_fp[additional_fp['Node']=='ID'])
        fp += len(additional_fp)
        not_matched = len(additional_fp)
        p_object.loc[additional_fp.index, 'classification_NS'] = 'FP'

        return tp, fp, fn, gt_object, p_object, node_wrong, type_wrong, not_matched, ns_ik_fp, ns_id_fp

    def evaluate(self, object_id):
        gt_object = self.ground_truth[(self.ground_truth['ObjectID'] == object_id) & \
                          (self.ground_truth['Direction'] != 'ES')].copy()
        p_object = self.participant[(self.participant['ObjectID'] == object_id) & \
                                    (self.participant['Direction'] != 'ES')].copy()
        p_object['matched'] = False
        p_object['classification'] = None
        p_object['distance'] = None
        gt_object['classification'] = None
        gt_object['distance'] = None
        tp = 0
        fp = 0
        fn = 0
        node_wrong =0
        type_wrong = 0
        not_matched = 0
        ew_fp = 0
        ns_fp = 0
        ns_ik_fp =0
        ns_id_fp =0

        for gt_idx, gt_row in gt_object.iterrows():
            matching_participant_events = p_object[
                (p_object['TimeIndex'] >= gt_row['TimeIndex'] - self.tolerance) &
                (p_object['TimeIndex'] <= gt_row['TimeIndex'] + self.tolerance) &
                (p_object['Direction'] == gt_row['Direction']) & 
                (p_object['matched'] == False)
            ]

            if len(matching_participant_events) > 0:
                p_idx = matching_participant_events.index[0]
                p_row = matching_participant_events.iloc[0]
                distance = p_row['TimeIndex'] - gt_row['TimeIndex']
                if p_row['Node'] == gt_row['Node'] and p_row['Type'] == gt_row['Type']:
                    tp += 1
                    
                    gt_object.loc[gt_idx, 'classification'] = 'TP'
                    gt_object.loc[gt_idx, 'distance'] = distance
                    p_object.loc[p_idx, 'classification'] = 'TP'
                    p_object.loc[p_idx, 'distance'] = distance
                else:
                    fp += 1
                    gt_object.loc[gt_idx, 'classification'] = 'FP'
                    gt_object.loc[gt_idx, 'distance'] = distance
                    p_object.loc[p_idx, 'classification'] = 'FP'
                    p_object.loc[p_idx, 'distance'] = distance
                    if p_row['Node']!= gt_row['Node']:
                        node_wrong += 1
                    if p_row['Type']!= gt_row['Type']:
                        type_wrong +=1
                    if p_row['Direction'] == 'EW':
                        ew_fp +=1
                    if p_row['Direction'] == 'NS':
                        ns_fp +=1
                        if p_row['Node']=='IK':
                            ns_ik_fp+=1
                        if p_row['Node']=='ID':
                            ns_id_fp +=1
                p_object.loc[matching_participant_events.index[0], 'matched'] = True
            else:
                fn += 1
                gt_object.loc[gt_idx, 'classification'] = 'FN'
                

        additional_fp = p_object[~p_object['matched']].copy()
        ew_add_fp = additional_fp[additional_fp['Direction']=='EW'].copy()
        ns_add_fp = additional_fp[additional_fp['Direction']=='NS'].copy()
        ns_ik_add_fp = ns_add_fp[ns_add_fp['Node']=='IK'].copy()
        ns_id_add_fp = ns_add_fp[ns_add_fp['Node']=='ID'].copy()
        fp += len(additional_fp)
        ew_fp += len(ew_add_fp)
        ns_fp += len(ns_add_fp)
        ns_ik_fp += len(ns_ik_add_fp)
        ns_id_fp += len(ns_id_add_fp)
        not_matched = len(additional_fp)
        p_object.loc[additional_fp.index, 'classification'] = 'FP'

        return tp, fp, fn, gt_object, p_object, node_wrong, type_wrong, not_matched, ew_fp, ns_fp, ns_ik_fp, ns_id_fp
    
    def score_EW(self, debug=False):
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_wrong_node = 0
        total_wrong_type = 0
        total_not_matched =0
        total_distances = []
        total_ik_fp =0
        total_id_fp =0

        for object_id in self.ground_truth['ObjectID'].unique():
            _, _, _, gt_object, p_object, wrong_node, wrong_type, not_matched, ew_ik_fp, ew_id_fp= self.evaluate_EW(object_id)
            
            total_tp += len(p_object[p_object['classification_EW'] == 'TP'])
            total_fp += len(p_object[p_object['classification_EW'] == 'FP'])
            total_fn += len(gt_object[gt_object['classification_EW'] == 'FN'])
            total_wrong_type += wrong_type
            total_wrong_node += wrong_node
            total_not_matched += not_matched
            total_ik_fp+=ew_ik_fp
            total_id_fp+=ew_id_fp
            total_distances.extend(
                p_object[p_object['classification_EW'] == 'TP']['distance'].tolist()
            )
        
        if debug:
            print(f"Total TPs: {total_tp}")
            print(f"Total FPs: {total_fp}")
            print(f"Total FNs: {total_fn}")
            print(f"Total Distances: {total_distances}")
            print(f"Total Wrong Nodes: {total_wrong_node}")
            print(f"Total Wrong Types: {total_wrong_type}")
            print(f"Total Not Matched: {total_not_matched}")
            print(f"Total IK FP: {total_ik_fp}")
            print(f"Total ID FP: {total_id_fp}")
            ad_fp = total_not_matched-total_id_fp-total_ik_fp
            print(f"Total AD FP: {ad_fp}")

        precision = total_tp / (total_tp + total_fp) \
            if (total_tp + total_fp) != 0 else 0
        recall = total_tp / (total_tp + total_fn) \
            if (total_tp + total_fn) != 0 else 0
        f2 = (5 * total_tp) / (5 * total_tp + 4 * total_fn + total_fp) \
            if (5 * total_tp + 4 * total_fn + total_fp) != 0 else 0
        rmse = np.sqrt((sum(d ** 2 for d in total_distances) / len(total_distances))) if total_distances else 0

        return precision, recall, f2, rmse
    
    def score_NS(self, debug=False):
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_wrong_node = 0
        total_wrong_type = 0
        total_not_matched =0
        total_distances = []
        total_ik_fp =0
        total_id_fp =0

        for object_id in self.ground_truth['ObjectID'].unique():
            _, _, _, gt_object, p_object, wrong_node, wrong_type, not_matched, ns_ik_fp, ns_id_fp= self.evaluate_NS(object_id)
            
            total_tp += len(p_object[p_object['classification_NS'] == 'TP'])
            total_fp += len(p_object[p_object['classification_NS'] == 'FP'])
            total_fn += len(gt_object[gt_object['classification_NS'] == 'FN'])
            total_wrong_type += wrong_type
            total_wrong_node += wrong_node
            total_not_matched += not_matched
            total_ik_fp+=ns_ik_fp
            total_id_fp+=ns_id_fp
            total_distances.extend(
                p_object[p_object['classification_NS'] == 'TP']['distance'].tolist()
            )
        
        if debug:
            print(f"Total TPs: {total_tp}")
            print(f"Total FPs: {total_fp}")
            print(f"Total FNs: {total_fn}")
            print(f"Total Distances: {total_distances}")
            print(f"Total Wrong Nodes: {total_wrong_node}")
            print(f"Total Wrong Types: {total_wrong_type}")
            print(f"Total Not Matched: {total_not_matched}")
            print(f"Total IK FP: {total_ik_fp}")
            print(f"Total ID FP: {total_id_fp}")
            ad_fp = total_not_matched-total_id_fp-total_ik_fp
            print(f"Total AD FP: {ad_fp}")

        precision = total_tp / (total_tp + total_fp) \
            if (total_tp + total_fp) != 0 else 0
        recall = total_tp / (total_tp + total_fn) \
            if (total_tp + total_fn) != 0 else 0
        f2 = (5 * total_tp) / (5 * total_tp + 4 * total_fn + total_fp) \
            if (5 * total_tp + 4 * total_fn + total_fp) != 0 else 0
        rmse = np.sqrt((sum(d ** 2 for d in total_distances) / len(total_distances))) if total_distances else 0

        return precision, recall, f2, rmse

    def score(self, debug=False):
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_wrong_node = 0
        total_wrong_type = 0
        total_not_matched =0
        total_ew_fp =0
        total_ns_fp =0
        total_distances = []

        for object_id in self.ground_truth['ObjectID'].unique():
            _, _, _, gt_object, p_object, wrong_node, wrong_type, not_matched, ew_fp, ns_fp,ns_ik_fp, ns_id_fp = self.evaluate(object_id)
            
            total_tp += len(p_object[p_object['classification'] == 'TP'])
            total_fp += len(p_object[p_object['classification'] == 'FP'])
            total_fn += len(gt_object[gt_object['classification'] == 'FN'])
            total_wrong_type += wrong_type
            total_wrong_node += wrong_node
            total_not_matched += not_matched
            total_ew_fp += ew_fp
            total_ns_fp += ns_fp
            total_distances.extend(
                p_object[p_object['classification'] == 'TP']['distance'].tolist()
            )
        
        if debug:
            print(f"Total TPs: {total_tp}")
            print(f"Total FPs: {total_fp}")
            print(f"Total FNs: {total_fn}")
            print(f"Total Distances: {total_distances}")
            print(f"Total Wrong Nodes: {total_wrong_node}")
            print(f"Total Wrong Types: {total_wrong_type}")
            print(f"Total Not Matched: {total_not_matched}")
            print(f"Total EW FP: {total_ew_fp}")
            print(f"Total NS FP: {total_ns_fp}")

        precision = total_tp / (total_tp + total_fp) \
            if (total_tp + total_fp) != 0 else 0
        recall = total_tp / (total_tp + total_fn) \
            if (total_tp + total_fn) != 0 else 0
        f2 = (5 * total_tp) / (5 * total_tp + 4 * total_fn + total_fp) \
            if (5 * total_tp + 4 * total_fn + total_fp) != 0 else 0
        rmse = np.sqrt((sum(d ** 2 for d in total_distances) / len(total_distances))) if total_distances else 0

        return precision, recall, f2, rmse


    def plot(self, object_id):
        tp, fp, fn, gt_object, p_object,wrong_nodes, wrong_types, not_matched, ew_fp, ns_fp,ns_ik_fp, ns_id_fp = self.evaluate(object_id)
        tp_distances = p_object[p_object['classification'] == 'TP']['distance']
        mse = np.sqrt(mean_squared_error([0] * len(tp_distances), tp_distances)) if tp_distances.size else 0
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ground_truth_EW = gt_object[gt_object['Direction'] == 'EW']
        ground_truth_NS = gt_object[gt_object['Direction'] == 'NS']
        participant_EW = p_object[p_object['Direction'] == 'EW']
        participant_NS = p_object[p_object['Direction'] == 'NS']
        self._plot_type_timeline(ground_truth_EW, participant_EW, ax1, 'EW')
        self._plot_type_timeline(ground_truth_NS, participant_NS, ax2, 'NS')
        plt.xlabel('TimeIndex')
        title_info = f"Object {object_id}: TPs={tp}, FPs={fp}, FNs={fn}"
        fig.suptitle(title_info, fontsize=10)
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        legend_elements = [
            plt.Line2D([0], [0], color='green', linestyle='dashed', label='True Positive (TP)'),
            plt.Line2D([0], [0], color='blue', linestyle='dashed', label='False Positive (FP)'),
            plt.Line2D([0], [0], color='red', linestyle='dashed', label='False Negative (FN)'),
            patches.Patch(color='grey', alpha=0.2, label='Tolerance Interval')
        ]
        plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.5), ncol=4)
        plt.show()


    def _plot_type_timeline(self, ground_truth_type, participant_type, ax, 
                            type_label):
        for _, row in ground_truth_type.iterrows():
            label = row['Node'] + '-' + row['Type']
            ax.scatter(row['TimeIndex'], 2, color='black')
            ax.text(row['TimeIndex'] + 3, 2.05, label, rotation=45)
            ax.fill_betweenx([1, 2], row['TimeIndex'] - self.tolerance,
                             row['TimeIndex'] + self.tolerance, color='grey',
                             alpha=0.2)
            if row['classification'] == 'TP':
                ax.text(row['TimeIndex'] + self.tolerance + .5, 1.5, 
                        str(row['distance']), 
                        color='black')
                ax.plot([row['TimeIndex'], 
                         row['TimeIndex'] + row['distance']], [2, 1], 
                         color='green', linestyle='dashed')
            elif row['classification'] == 'FP':
                ax.plot([row['TimeIndex'], 
                         row['TimeIndex'] + row['distance']], [2, 1], 
                         color='blue', linestyle='dashed')
            elif row['classification'] == 'FN':
                ax.plot([row['TimeIndex'], 
                         row['TimeIndex']], [2, 2.2], color='red', 
                         linestyle='dashed')

        for _, row in participant_type.iterrows():
            label = row['Node'] + '-' + row['Type']
            ax.scatter(row['TimeIndex'], 1, color='black')
            ax.text(row['TimeIndex'] + 3, 1.05, label, rotation=45)
            if row['classification'] == 'FP' and row['matched'] == False:
                ax.plot([row['TimeIndex'], row['TimeIndex']], [1, 0.8], 
                        color='blue', linestyle='dashed')

        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.yaxis.grid(True)
        ax.set_yticks([1, 2])
        ax.set_yticklabels(['Participant', 'Ground truth'])
        ax.set_title(type_label)


def merge_label_files(label_folder):
    """
    Merges all label files in a given folder into a single pandas DataFrame. 
    The filenames must be in the format <ObjectID>.csv, and the object id will
    be extracted from the filename and added as a column to the DataFrame.

    Args:
        label_folder (str): The path to the folder containing the label files.

    Returns:
        pandas.DataFrame: A DataFrame containing the merged label data.
    """
    label_data = []
    label_folder = Path(label_folder).expanduser()
    for file_path in label_folder.ls():
        df = pd.read_csv(file_path)
        oid_s = os.path.basename(file_path).split('.')[0]  # Extract ObjectID from filename
        df['ObjectID'] = int(oid_s)
        label_data.append(df)

    label_data = pd.concat(label_data)
    label_data = label_data[['ObjectID'] + list(label_data.columns[:-1])]
    return label_data


def run_evaluator(ground_truth_path=None, participant_path=None, plot_object=None):

    if participant_path is None:
        participant_df = pd.read_csv('participant_toy.csv')
    else:
        participant_path = Path(participant_path).expanduser()
        if participant_path.is_dir():
            participant_df = merge_label_files(participant_path)  
        else:
            participant_df = pd.read_csv(participant_path)
    
    if ground_truth_path is None:
        ground_truth_df = pd.read_csv('ground_truth_toy.csv')
    else:
        ground_truth_path = Path(ground_truth_path).expanduser()
        if ground_truth_path.is_dir():
            ground_truth_df = merge_label_files(ground_truth_path)
        else:
            ground_truth_df = pd.read_csv(ground_truth_path)
    

    # Create a NodeDetectionEvaluator instance
    #print(f'Participant submission filepath: {participant_path}')
    #print(f'Ground truth filepath: {ground_truth_path}')
    #print(f'rows in participant: {len(participant_df)}')
    #print(f'rows in ground truth: {len(ground_truth_df)}')  
    evaluator = NodeDetectionEvaluator(ground_truth_df, participant_df, tolerance=6)
    precision, recall, f2, rmse = evaluator.score(debug=True)
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F2: {f2:.2f}')
    print(f'RMSE: {rmse:.2f}')

    # Plot the evaluation for the selected object (if any)
    if plot_object:
        evaluator.plot(object_id=plot_object)
    return precision, recall, f2, rmse

if __name__ == "__main__":
    if 'ipykernel' in sys.modules:
        run_evaluator(plot_object=True)
    else:
        # Parse the arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('--participant', type=str, required=False, 
                            help='Path to the participant file or folder. \
                                If a folder is provided, all the CSV files in the \
                                    folder will be used. If none, the toy example \
                                        will be used.')
        parser.add_argument('--ground_truth', type=str, required=False,
                            help='Path to the ground truth file. If none, the toy \
                                example will be used.')
        parser.add_argument('--plot_object', type=int, required=False,
                    help='Object ID to plot.')
        args = parser.parse_args()
        run_evaluator(args.ground_truth, args.participant, args.plot_object)