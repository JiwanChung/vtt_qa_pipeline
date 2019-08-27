from functools import partial
from collections import defaultdict

import torch
from torchtext import data
import nltk

from utils import pad_tensor
from preprocess_image import preprocess_images, get_empty_image_vector


def load_text_data(args, tokenizer):
    vid = InfoField()
    videoType = InfoField()
    q_level_mem = InfoField()
    q_level_logic = InfoField()

    que = data.Field(sequential=True, tokenize=tokenizer, lower=args.lower)
    true_ans = data.Field(sequential=True, tokenize=tokenizer, lower=args.lower,
                          init_token='<sos>', eos_token='<eos>',
                          unk_token='<unk>')

    single_false_ans = data.Field(sequential=True, tokenize=tokenizer, lower=args.lower,
                          init_token='<sos>', eos_token='<eos>',
                          unk_token='<unk>')
    false_ans = data.NestedField(single_false_ans)

    text_data = data.TabularDataset(
        path=args.data_path, format='json',
        fields={
            'vid': ('vid', vid),
            'videoType': ('videoType', videoType),
            'q_level_mem': ('q_level_mem', q_level_mem),
            'q_level_logic': ('q_level_logic', q_level_logic),
            'que': ('que', que),
            'true_ans': ('true_ans', true_ans),
            'false_ans': ('false_ans', false_ans),
        })

    train, test, val = text_data.split(args.split_ratio)
    if args.shots_only:
        remove_scene_questions(train)
        remove_scene_questions(test)
        remove_scene_questions(val)

    train_iter, test_iter, val_iter = data.Iterator.splits(
        (train, test, val), sort_key=lambda x: len(x.true_ans),
        batch_sizes=args.batch_sizes, device=args.device,
        sort_within_batch=True,
    )

    vocab_args = {}
    k = 'vocab_pretrained'
    if hasattr(args, k):
        vocab_args['vectors'] = getattr(args, k)
    que.build_vocab(train.que, train.true_ans,
                    train.false_ans, train.single_false_ans,
                    **vocab_args)
    que.vocab = process_vocab(que.vocab)
    vocab = que.vocab
    true_ans.vocab = vocab
    false_ans.vocab = vocab
    single_false_ans.vocab = vocab

    return {'train': train_iter, 'val': val_iter, 'test': test_iter}, vocab


def process_vocab(vocab):
    vocab.specials = ['<sos>', '<eos>', '<unk>', '<pad>']

    vocab.special_ids = [vocab.stoi[k] for k in vocab.specials]
    for token in vocab.specials:
        setattr(vocab, token[1:-1], token)

    return vocab


def get_tokenizer(args):
    return {
        'nltk': nltk.word_tokenize
    }[args.tokenizer.lower()]


class ImageIterator:
    def __init__(self, args, text_it):
        self.it = text_it
        self.args = args
        self.allow_empty_images = args.allow_empty_images
        self.num_workers = args.num_workers
        self.device = args.device

        self.image_dt = self.load_images(args.image_path, text_it.dataset,
                                         cache=args.cache_image_vectors, device=args.device)
        print(f"total vids: {len(list(self.image_dt))}")

    def __iter__(self):
        for batch in self.it:
            batch.images = [torch.from_numpy(self.image_dt[vid]).to(self.device).split(1) for vid in batch.vid]
            batch.images = pad_tensor(batch.images).squeeze(2)

            yield batch

    def load_images(self, image_path, dataset, cache=True, device=-1):
        images = preprocess_images(self.args, image_path, cache=cache, device=device, num_workers=self.num_workers)
        if self.allow_empty_images:
            for k, v in images.items():
                sample_image = v
                break
            func = partial(get_empty_image_vector, sample_image_size=list(sample_image.shape))
            images = defaultdict(func, images)

        return {ex.vid: images[ex.vid] for ex in dataset}


def get_image_iterator(args, text_it):
    return ImageIterator(args, text_it)


# batch: [len, batch_size]
def get_iterator(args):
    print("Loading Text Data")
    tokenizer = get_tokenizer(args)
    iters, vocab = load_text_data(args, tokenizer)
    print("Loading Image Data")
    image_iters = {}
    for key, it in iters.items():
        image_iters[key] = get_image_iterator(args, it)
    print("Data Loading Done")

    return image_iters, vocab


def remove_scene_questions(dataset):
    li = []
    for example in dataset.examples:
        if not example.vid.endswith("_000"):
            li.append(example)

    dataset.examples = li


class InfoField(data.RawField):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_target = False
