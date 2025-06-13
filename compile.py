import os
from datetime import datetime, timedelta

# https://docs.pytorch.org/docs/stable/torch.compiler_get_started.html


def f(x):
    a = torch.cos(x)
    b = torch.sin(a)
    return b


def test_f():
    print('test f')
    t1 = datetime.now()
    new_fn = torch.compile(f, backend="inductor")
    t2 = datetime.now()
    print(f'compile: {t2 - t1}')
    input_tensor = torch.randn(10000).to(device="cuda:0")
    a = new_fn(input_tensor)
    print(a)


def test_hub():
    print('test hub')
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    t1 = datetime.now()
    opt_model = torch.compile(model, backend="inductor")
    t2 = datetime.now()
    print(f'compile: {t2 - t1}')
    opt_model(torch.randn(1, 3, 64, 64))


def test_pretrained():
    print('test pretrained')
    from transformers import BertTokenizer, BertModel
    # Copy pasted from here https://huggingface.co/bert-base-uncased
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained("bert-base-uncased").to(device="cuda:0")
    t1 = datetime.now()
    model = torch.compile(model, backend="inductor")  # This is the only line of code that we changed
    t2 = datetime.now()
    print(f'compile: {t2 - t1}')
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt').to(device="cuda:0")
    output = model(**encoded_input)
    print(output)


def test_pony():
    print('test pony')
    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained("AstraliteHeart/pony-diffusion")
    start = datetime.now()
    pipe = pipe.to(device="cuda:0")
    pipe = torch.compile(pipe, backend="inductor")
    prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
    first = timedelta(0)
    total = timedelta(0)
    n = 100
    for i in range(n):
        t1 = datetime.now()
        image = pipe(prompt).images[0]
        t2 = datetime.now()
        print(image)
        print(f'predict: {t2 - t1}')
        if i == 0:
            first = t2 - start
        total = total + (t2 - t1)
    print(f'time to first: {first}')
    print(f'average time: {total / n}')


if __name__ == "__main__":
    # https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html
    cache_dir = os.path.join(os.getcwd(), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = cache_dir
    #
    # os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'
    # os.environ['TORCHINDUCTOR_AUTOGRAD_CACHE'] = '1'

    os.environ['TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE'] = '1'
    os.environ['TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE'] = '1'
    os.environ['TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE'] = '1'
    os.environ['TORCHINDUCTOR_REDIS_URL'] = 'redis://:torchpass@localhost'

    # os.environ['TORCHINDUCTOR_FORCE_DISABLE_CACHES'] = '1'

    import torch

    # test_f()
    # test_hub()
    # test_pretrained()
    test_pony()
