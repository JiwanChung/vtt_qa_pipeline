from ignite.engine.engine import Engine, State, Events

from dataset import get_iterator
from model import get_model
from loss import get_loss
from optimizer import get_optimizer
from logger import get_logger, log_results

from utils import prepare_batch
from metric import get_metrics
from evaluate import get_evaluator, evaluate_once
from metric.stat_metric import StatMetric


def get_trainer(args, model, loss_fn, optimizer):
    def update_model(trainer, batch):
        model.train()
        optimizer.zero_grad()
        net_inputs, target = prepare_batch(args, batch, model.vocab)
        y_pred = model(**net_inputs)
        batch_size = y_pred.shape[0]
        loss, stats = loss_fn(y_pred, target)
        loss.backward()
        optimizer.step()
        return loss.item(), stats, batch_size, y_pred.detach(), target.detach()

    trainer = Engine(update_model)

    metrics = {
        'loss': StatMetric(output_transform=lambda x: (x[0], x[2])),
        'top1_acc': StatMetric(output_transform=lambda x: ((x[3].argmax(dim=-1) == x[4]).float().mean().item(), x[2]))
    }
    if hasattr(loss_fn, 'get_metric'):
        metrics = {**metrics, **loss_fn.get_metric()}

    for name, metric in metrics.items():
        metric.attach(trainer, name)

    return trainer


def train(args):
    iters, vocab = get_iterator(args)

    model = get_model(args, vocab)
    loss_fn = get_loss(args, vocab)
    optimizer = get_optimizer(args, model)

    trainer = get_trainer(args, model, loss_fn, optimizer)
    metrics = get_metrics(args, vocab)
    evaluator = get_evaluator(args, model, loss_fn, metrics)

    logger = get_logger(args)
    @trainer.on(Events.STARTED)
    def on_training_started(engine):
        print("Begin Training")

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_iter_results(engine):
        log_results(logger, 'train/iter', engine.state, engine.state.iteration)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate_epoch(engine):
        log_results(logger, 'train/epoch', engine.state, engine.state.epoch)
        state = evaluate_once(evaluator, iterator=iters['val'])
        log_results(logger, 'valid/epoch', state, engine.state.epoch)

    trainer.run(iters['train'], max_epochs=args.max_epochs)
