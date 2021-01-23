from big_sleep import Imagine
import sys

text = sys.argv[1]

train = Imagine(
    text = text,
    lr = 0.07,
    save_every = 25,
    save_progress = True
)

train()
