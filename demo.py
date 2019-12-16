from parser_setting import args
from trainer.train import Trainer


if __name__ == '__main__':
    # train(args=args)
    trainer = Trainer(args=args)
    trainer.train()
