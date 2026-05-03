"""Task implementations."""
from .base import BaseTask, TaskConfig
from .copy_task import CopyTask
from .delayed_recall import DelayedRecallTask
from .running_sum import RunningSumTask
from .chain_rule import ChainRuleTask


def create_task(task_name: str, config: TaskConfig = None) -> BaseTask:
    """Create a task by name.
    
    Args:
        task_name: One of 'copy', 'recall', 'sum', 'chain'
        config: Optional task configuration
        
    Returns:
        The requested task
    """
    if config is None:
        config = TaskConfig()
    
    task_name = task_name.lower()
    
    if task_name == 'copy':
        return CopyTask(config)
    elif task_name in ('recall', 'delayed_recall'):
        return DelayedRecallTask(config)
    elif task_name in ('sum', 'running_sum'):
        return RunningSumTask(config)
    elif task_name in ('chain', 'chain_rule'):
        return ChainRuleTask(config)
    else:
        raise ValueError(f"Unknown task: {task_name}")


def create_all_tasks(config: TaskConfig = None):
    """Create all tasks for comparison."""
    return {
        'copy': create_task('copy', config),
        'recall': create_task('recall', config),
        'sum': create_task('sum', config),
        'chain': create_task('chain', config)
    }


__all__ = [
    'BaseTask',
    'TaskConfig',
    'CopyTask',
    'DelayedRecallTask',
    'RunningSumTask',
    'ChainRuleTask',
    'create_task',
    'create_all_tasks',
]
