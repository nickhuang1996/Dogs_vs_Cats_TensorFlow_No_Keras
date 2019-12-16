import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='D:/datasets/dogsvscats/train')
parser.add_argument('--image_label_list_file', type=str, default='D:/datasets/dogsvscats/image_label_list.txt')
parser.add_argument('--train_data_ratio', type=float, default=0.8)
parser.add_argument('--record_dir', type=str, default='D:/datasets/dogsvscats')

parser.add_argument('--checkpoint_dir', type=str, default='D:/weights_results/Dogs_vs_Cats_TensorFlow_No_Keras')
parser.add_argument('--model_name', type=str, default='resnet_v1_50')
parser.add_argument('--log_dir', type=str, default='D:/weights_results/Dogs_vs_Cats_TensorFlow_No_Keras/tensorboard')

parser.add_argument('--image_size', type=tuple, default=(224, 224))

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--pretrained_model', type=str, default='resnet_v1_50')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--decay_rate', type=float, default=0.1)

parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.99)
parser.add_argument('--epsilon', type=float, default=2e-8)

parser.add_argument('--max_step', type=int, default=110)
parser.add_argument('--per_evaluate_step', type=int, default=5)
parser.add_argument('--per_save_checkpoint_step', type=int, default=10)
args = parser.parse_args()
