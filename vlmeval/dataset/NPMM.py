from typing import Any


from vlmeval.smp import *
from vlmeval.dataset.image_base import ImageBaseDataset
import pandas as pd
from collections import defaultdict
from vlmeval.dataset.utils.NPMM.tsp import validation as tsp_validation
from vlmeval.dataset.utils.NPMM.hamiltonian_cycle import validation as hamiltonian_cycle_validation
from vlmeval.dataset.utils.NPMM.maximum_set import validation as maximum_set_validation
from vlmeval.dataset.utils.NPMM.minimum_cut import validation as minimum_cut_validation
from vlmeval.dataset.utils.NPMM.gcp import validation as gcp_validation
from vlmeval.dataset.utils.NPMM.mcp import validation as mcp_validation

class MMHELIX(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL = {
        'NPMM': '',
    }
    GROUP_LIST = {
        "selection": ["set-cover", "subset-sum", "knapsack"],
        "planning": ["NpTsp", "NpHamiltonianCycle"],
        "graph": ["NpMaximumCliqueProblem", "NpMaximumSet", "NpGcpD"],
        "partition": ["NpMinimumCut"],
        "schedule": ["meeting-schedule"]
    }
    
    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        stats = defaultdict[Any, dict[str, int | float]](lambda: {'total': 0, 'valid': 0, 'ar_accum': 0.0})
        target_tasks = ["NpTsp", "NpHamiltonianCycle", "NpMaximumCliqueProblem", 
                        "NpMinimumCut", "NpMaximumSet", "NpGcpD"]
        for index, data_item in data.iterrows():
            task = data_item.get("task").split("-")[1]
            if task not in target_tasks:
                continue
            prediction = data_item.get('prediction', None)
            reward_model = data_item.get('reward_model', None)   
            ground_truth = reward_model['ground_truth']['ground_truth']
            graph = data_item.get('question')
            stats[task]['total'] += 1
            is_invalid = True
            value = 0
            try:
                if task == "NpTsp":
                    is_invalid, value, msg = tsp_validation(graph, prediction)
                elif task == "NpHamiltonianCycle":
                    is_invalid, value, msg = hamiltonian_cycle_validation(graph, prediction)
                elif task == "NpMaximumCliqueProblem":
                    is_invalid, value, msg = mcp_validation(graph, prediction)
                elif task == "NpMinimumCut":
                    is_invalid, value, msg = minimum_cut_validation(graph, prediction)
                elif task == "NpMaximumSet":
                    is_invalid, value, msg = maximum_set_validation(graph, prediction)
                elif task == "NpGcpD":
                    is_invalid, value, msg = gcp_validation(graph, prediction)
            except Exception as e:
                print(f"Validation error for task {task}, index {index}: {e}")
                is_invalid = True
            if not is_invalid:
                stats[task]['valid'] += 1
                if task in ["NpGcpD", "NpMinimumCut"]:
                    stats[task]['ar_accum'] +=  float(ground_truth) / value
                else:
                    stats[task]['ar_accum'] += value / float(ground_truth) 

        # 计算每个子任务的最终统计结果
        subtask_results = {}
        for task in target_tasks:
            t_stat = stats[task]
            total = t_stat['total']
            if total > 0:
                sr = t_stat['valid'] / total
                ar = t_stat['ar_accum'] / total
            else:
                sr = 0.0
                ar = 0.0
            
            subtask_results[task] = {'SR': sr, 'AR': ar, 'count': total}
        subtask_file = get_intermediate_file_path(eval_file, '_subtask_stats', 'json')
        dump(subtask_results, subtask_file)
        group_stats = []
        
        total_sr = 0
        total_ar = 0
        valid_tasks_count = 0
        
        for group, tasks in self.GROUP_LIST.items():
            relevant_tasks = [t for t in tasks if t in subtask_results and subtask_results[t]['count'] > 0]
            
            if not relevant_tasks:
                continue
            avg_sr = sum(subtask_results[t]['SR'] for t in relevant_tasks) / len(relevant_tasks)
            avg_ar = sum(subtask_results[t]['AR'] for t in relevant_tasks) / len(relevant_tasks)
            
            group_stats.append({
                'Task': group, 
                'SR': avg_sr,
                'AR': avg_ar,
                'num_subtasks': len(relevant_tasks)
            })
            
            total_sr += avg_sr
            total_ar += avg_ar
            valid_tasks_count += 1

        if valid_tasks_count > 0:
            group_stats.append({
                'Task': 'Overall',
                'SR': total_sr / valid_tasks_count,
                'AR': total_ar / valid_tasks_count,
                'num_subtasks': valid_tasks_count
            })

        accuracy_df = pd.DataFrame(group_stats)
        
        score_file = get_intermediate_file_path(eval_file, '_acc', 'csv')
        dump(accuracy_df, score_file)
        
        return accuracy_df
