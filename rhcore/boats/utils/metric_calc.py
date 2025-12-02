def reset_metrics(metrics):
    for _, metric in metrics.items(): # _: metric_name
        if hasattr(metric, 'reset'):
            metric.reset()

