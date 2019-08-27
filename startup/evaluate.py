import torch
from ignite.engine.engine import Engine, State, Events
from metric.stat_metric import StatMetric

from utils import prepare_batch


def get_evaluator(args, model, loss_fn, metrics={}):
    def _inference(evaluator, batch):
        model.eval()
        with torch.no_grad():
            net_inputs, target = prepare_batch(args, batch, model.vocab)
            y_pred = model(**net_inputs)
            batch_size = y_pred.shape[0]
            loss, stats = loss_fn(y_pred, target)
            return loss.item(), stats, batch_size, y_pred, target  # TODO: add false_answer metric

    engine = Engine(_inference)

    metrics = {**metrics, **{
        'loss': StatMetric(output_transform=lambda x: (x[0], x[2])),
        'top1_acc': StatMetric(output_transform=lambda x: ((x[3].argmax(dim=-1) == x[4]).float().mean().item(), x[2]))
    }}
    if hasattr(loss_fn, 'get_metric'):
        metrics = {**metrics, **loss_fn.get_metric()}

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def evaluate_once(evaluator, iterator):
    evaluator.run(iterator)
    return evaluator.state


def evaluate(args):
    pass
